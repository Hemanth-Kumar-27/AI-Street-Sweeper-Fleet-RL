"""
Multi-agent reinforcement learning for fleet task allocation.

Splitting the coverage workload across a fleet of sweepers is a genuinely
different problem from routing a single vehicle. Routing — given a fixed
set of streets, find the shortest walk covering them all — is exactly
solved by Route Inspection / the Chinese Postman Problem (route_inspection
.py); there's no reason to learn something that's already exactly solvable.

But *deciding which vehicle is responsible for which streets*, so that no
truck ends up driving twice as long as the rest of the fleet, is a
different problem: the "min-max multi-vehicle arc routing problem" (a.k.a.
the multiple postmen problem). It's provably NP-hard — there is no
closed-form or polynomial-exact algorithm the way there is for CPP. That's
the actual gap RL fills here: not routing (CPP already does that
optimally, and still does after this module runs), but the combinatorial,
fairness-constrained assignment decision routing structurally can't make
on its own.

Two-phase approach:
  1. Voronoi seed: pick k geographically spread starting nodes, then assign
     every street edge to whichever start node it's graph-closer to
     (multi-source Dijkstra). This is a classical, deterministic partition
     — no learning — and its whole purpose is to guarantee every vehicle's
     initial territory is spatially contiguous. It is *not* balanced,
     though: real street networks aren't uniform, so the Voronoi cells
     will generally hold very different amounts of driving.
  2. RL rebalancing: a cooperative multi-agent Q-learning policy then
     repeatedly hands a single boundary edge from the currently busiest
     vehicle to whichever neighbouring vehicle it's already touching,
     learning (via a shared Q-ELM — the same fast, closed-form-trained
     approximator used elsewhere in this project) which transfers actually
     shrink the makespan rather than just shuffling the imbalance
     elsewhere. Because every transfer only ever moves an edge between two
     vehicles that already share a boundary node, territories stay
     contiguous throughout — unlike growing partitions from scratch
     edge-by-edge, which (verified empirically while building this) tends
     to box vehicles in and fragment their regions.

Once the RL policy has decided the final partition, each vehicle's
assigned edges are handed back to the exact, classical Route Inspection
solver (route_inspection.solve_circuit) to produce that vehicle's own
optimal sub-route. This mirrors the division of labour already used for
open-area pockets in hybrid_planner.py: RL solves the part of the problem
that has no closed-form answer (who covers what); CPP solves the part
that does (the shortest walk covering it).
"""

from __future__ import annotations

import os as _os
import random as _random
import time as _time
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

import networkx as nx
import numpy as np

from route_inspection import solve_circuit
from sweeper_rl_elm import ELM

NodeKey = Tuple[float, float]
EdgeKey = Tuple[NodeKey, NodeKey, int]

FEATURE_DIM = 5
DEFAULT_EPISODES = 8
DEFAULT_HIDDEN = 30
MAX_VEHICLES = 6
TRANSFERS_PER_VEHICLE = 20  # transfer-step budget scales with k: more vehicles, more boundary to rebalance
DEFAULT_RESTARTS = 5  # independent seed-partition + RL-training attempts; the best (lowest makespan) wins


def _pick_start_nodes(G: nx.MultiGraph, k: int, first_node: NodeKey = None) -> List[NodeKey]:
    """k geographically spread starting nodes via farthest-point sampling —
    the seeds for the Voronoi partition below. Spread-out seeds are what
    make the resulting territories roughly equal-sized and non-interleaved.

    `first_node` picks which node farthest-point sampling grows from (it
    defaults to the graph's first node, which is deterministic but
    arbitrary — just whatever order the graph happened to build nodes in).
    Different starting nodes generally produce differently-shaped initial
    territories, since every other seed is chosen relative to the ones
    before it. train_fleet_assignment uses this to try a handful of
    different starting shapes and keep whichever balances best, since a
    single farthest-point-sampling run has no way to know in advance which
    starting node will lead to the most even split of a lumpy, non-uniform
    street network."""
    nodes = list(G.nodes)
    if not nodes:
        return []
    if k >= len(nodes):
        return [nodes[i % len(nodes)] for i in range(k)]
    coords = {n: G.nodes[n]["latlng"] for n in nodes}
    chosen = [first_node if first_node is not None else nodes[0]]
    for _ in range(k - 1):
        best_n, best_d = None, -1.0
        for n in nodes:
            if n in chosen:
                continue
            d = min((coords[n][0] - coords[c][0]) ** 2 + (coords[n][1] - coords[c][1]) ** 2 for c in chosen)
            if d > best_d:
                best_d, best_n = d, n
        chosen.append(best_n)
    return chosen


def _edge_list(G: nx.MultiGraph):
    return list(G.edges(keys=True, data=True))


def _voronoi_partition(G: nx.MultiGraph, start_nodes: List[NodeKey], k: int):
    """Deterministic, always-contiguous seed partition: each node belongs
    to whichever start node it's graph-distance-closest to; each edge
    belongs to whichever of its two endpoints is more confidently "inside"
    its own territory (smaller distance to its nearest start)."""
    weight_of: Dict[EdgeKey, float] = {}
    for u, v, key, d in _edge_list(G):
        weight_of[(u, v, key)] = d["weight"]

    dist_to_start = [nx.single_source_dijkstra_path_length(G, s, weight="weight") for s in start_nodes]

    owner_of_node: Dict[NodeKey, int] = {}
    for n in G.nodes:
        best_i, best_d = 0, float("inf")
        for i in range(k):
            d = dist_to_start[i].get(n, float("inf"))
            if d < best_d:
                best_d, best_i = d, i
        owner_of_node[n] = best_i

    assigned: List[List[EdgeKey]] = [[] for _ in range(k)]
    for (u, v, key) in weight_of:
        ou, ov = owner_of_node[u], owner_of_node[v]
        du = dist_to_start[ou].get(u, float("inf"))
        dv = dist_to_start[ov].get(v, float("inf"))
        owner = ou if du <= dv else ov
        assigned[owner].append((u, v, key))

    return assigned, weight_of


def _bridge_edges_of(edges: Set[EdgeKey]) -> Set[EdgeKey]:
    """Which of a vehicle's own edges are bridges within its own induced
    subgraph — i.e. edges whose removal would split its territory into
    more pieces. Parallel duplicate edges between the same two nodes are
    never bridges (the other copy still connects them). Used to stop the
    RL rebalancing step from fragmenting a vehicle's region while trying
    to shed load from it."""
    if not edges:
        return set()
    simple = nx.Graph()
    multiplicity: Dict[frozenset, int] = {}
    for (u, v, _key) in edges:
        pair = frozenset((u, v))
        multiplicity[pair] = multiplicity.get(pair, 0) + 1
        simple.add_edge(u, v)
    bridge_pairs = {frozenset(e) for e in nx.bridges(simple)}
    unsafe: Set[EdgeKey] = set()
    for e in edges:
        u, v, _key = e
        pair = frozenset((u, v))
        if multiplicity[pair] == 1 and pair in bridge_pairs:
            unsafe.add(e)
    return unsafe


def _branch_bundle(edges: Set[EdgeKey], bridge_edge: EdgeKey, start_node: NodeKey) -> Tuple[EdgeKey, ...]:
    """`bridge_edge` is a bridge inside a vehicle's own territory (`edges`)
    — cutting it splits the territory into exactly two pieces. Returns the
    bridge edge plus every edge in whichever piece does NOT contain the
    vehicle's own start node: the "branch" that can be handed off as one
    atomic move without leaving the vehicle's remaining trunk disconnected.
    Without this, a heavily-loaded vehicle whose only boundary edges happen
    to be bridges has no safe way to shed any load at all — single-edge
    transfers alone got stuck well short of a good balance in testing."""
    g = nx.MultiGraph()
    for e in edges:
        if e == bridge_edge:
            continue
        u, v, key = e
        g.add_edge(u, v, key=key)
    bu, bv, _ = bridge_edge
    g.add_node(bu)
    g.add_node(bv)
    comp_of: Dict[NodeKey, int] = {}
    for idx, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            comp_of[n] = idx
    start_comp = comp_of.get(start_node)
    bundle = [bridge_edge]
    for e in edges:
        if e == bridge_edge:
            continue
        u, _v, _key = e
        if comp_of.get(u) != start_comp:
            bundle.append(e)
    return tuple(bundle)


def _run_refinement_episode(
    k: int,
    initial_assigned: List[List[EdgeKey]],
    weight_of: Dict[EdgeKey, float],
    start_nodes: List[NodeKey],
    elm: ELM,
    epsilon: float,
    rng: np.random.Generator,
    max_transfers: int,
):
    """Starting from the Voronoi seed partition, repeatedly transfer load
    from the busiest vehicle to a neighbouring vehicle it already shares a
    boundary node with — either a single edge, or (if that edge is a
    bridge within the busy vehicle's own territory) the whole branch beyond
    it, chosen epsilon-greedily from the shared Q-ELM — for up to
    max_transfers steps or until no further transfer touches a shared
    boundary. Returns the resulting assignment plus (features, reward)
    pairs for Q-learning."""
    assigned: List[Set[EdgeKey]] = [set(edges) for edges in initial_assigned]
    owner_of_edge: Dict[EdgeKey, int] = {e: i for i, edges in enumerate(assigned) for e in edges}

    incident: Dict[NodeKey, Set[EdgeKey]] = {}
    for e in weight_of:
        u, v, _ = e
        incident.setdefault(u, set()).add(e)
        incident.setdefault(v, set()).add(e)

    dist = [sum(weight_of[e] for e in assigned[i]) for i in range(k)]
    total_len = sum(weight_of.values()) or 1.0
    target_share = total_len / max(1, k)

    transitions: List[Tuple[np.ndarray, float]] = []

    for _step in range(max_transfers):
        i_max = int(np.argmax(dist))
        edges_i = assigned[i_max]
        if not edges_i:
            break
        unsafe = _bridge_edges_of(edges_i)
        bundle_cache: Dict[EdgeKey, Tuple[EdgeKey, ...]] = {}

        cands: List[Tuple[Tuple[EdgeKey, ...], int, float]] = []
        seen: Set[Tuple[EdgeKey, int]] = set()
        for e in edges_i:
            u, v, _ = e
            for n in (u, v):
                for other_e in incident.get(n, ()):
                    j = owner_of_edge.get(other_e)
                    if j is None or j == i_max or (e, j) in seen:
                        continue
                    seen.add((e, j))
                    if e in unsafe:
                        if e not in bundle_cache:
                            bundle_cache[e] = _branch_bundle(edges_i, e, start_nodes[i_max])
                        bundle = bundle_cache[e]
                    else:
                        bundle = (e,)
                    w = sum(weight_of[be] for be in bundle)
                    cands.append((bundle, j, w))
        if not cands:
            break

        feats = np.empty((len(cands), FEATURE_DIM), dtype=np.float64)
        for idx, (bundle, j, w) in enumerate(cands):
            new_i_max = dist[i_max] - w
            new_j = dist[j] + w
            feats[idx] = [
                w / total_len,
                (dist[i_max] - dist[j]) / target_share,
                (new_i_max - new_j) / target_share,
                dist[i_max] / target_share,
                len(bundle) / max(1, len(edges_i)),
            ]

        if rng.random() < epsilon:
            choice = int(rng.integers(len(cands)))
        else:
            q = elm.predict(feats)[:, 0]
            choice = int(np.argmax(q))

        bundle, j, w = cands[choice]
        old_makespan = max(dist)

        for be in bundle:
            assigned[i_max].discard(be)
            assigned[j].add(be)
            owner_of_edge[be] = j
        dist[i_max] -= w
        dist[j] += w

        new_makespan = max(dist)
        reward = (old_makespan - new_makespan) / target_share
        transitions.append((feats[choice], reward))

    makespan = max(dist) if dist else 0.0
    return [list(s) for s in assigned], dist, makespan, transitions


@dataclass
class FleetAssignment:
    edges_by_vehicle: List[List[EdgeKey]]
    dist_by_vehicle: List[float]
    makespan_m: float
    episodes_run: int
    train_ms: float
    history: list = field(default_factory=list)


def _train_one_restart(
    G: nx.MultiGraph,
    k: int,
    start_nodes: List[NodeKey],
    episodes: int,
    hidden_dim: int,
    seed: int,
    max_transfers: int,
):
    """One full seed-partition + Q-ELM training run from a given set of
    starting nodes, ending in a greedy (epsilon=0) rollout. Pulled out of
    train_fleet_assignment so it can be repeated from several different
    starting territories (see DEFAULT_RESTARTS) and the best of them kept —
    a single farthest-point-sampling seed has no way to know in advance
    which starting shape will end up easiest to balance evenly."""
    rng = np.random.default_rng(seed)
    initial_assigned, weight_of = _voronoi_partition(G, start_nodes, k)

    elm = ELM(FEATURE_DIM, hidden_dim, 1, seed=seed)
    elm.fit(np.zeros((1, FEATURE_DIM)), np.zeros((1, 1)), lam=1.0)  # dummy init, same pattern as train_q_elm

    buffer_X: List[np.ndarray] = []
    buffer_Y: List[List[float]] = []
    epsilon, eps_min = 0.9, 0.05
    eps_decay = (eps_min / 0.9) ** (1.0 / max(1, episodes - 1))
    history = []

    for ep in range(episodes):
        _, _, makespan, transitions = _run_refinement_episode(
            k, initial_assigned, weight_of, start_nodes, elm, epsilon, rng, max_transfers
        )
        for feats, reward in transitions:
            buffer_X.append(feats)
            buffer_Y.append([reward])
        if len(buffer_X) > 8000:
            buffer_X, buffer_Y = buffer_X[-8000:], buffer_Y[-8000:]
        if buffer_X:
            elm.fit(np.array(buffer_X), np.array(buffer_Y), lam=1.0)
        epsilon = max(eps_min, epsilon * eps_decay)
        history.append({"episode": ep, "makespan_m": makespan, "epsilon": epsilon})

    assigned, dist_by_vehicle, makespan, _ = _run_refinement_episode(
        k, initial_assigned, weight_of, start_nodes, elm, 0.0, rng, max_transfers
    )
    return assigned, dist_by_vehicle, makespan, history


def _routed_makespan(G: nx.MultiGraph, edges_by_vehicle: List[List[EdgeKey]]) -> Tuple[float, List[float]]:
    """The real per-vehicle route length once each vehicle's assigned edges
    are handed to the exact Route Inspection solver — this is what
    actually gets driven, and can diverge noticeably from the raw
    sum-of-edge-weights makespan tracked during RL training. Stitching a
    vehicle's territory into a single circuit (route_inspection.solve_
    circuit, via bridge_components()/_eulerize_min()) adds extra distance
    that depends on that territory's own shape — how many odd-degree nodes
    it has, whether it's split into disconnected pieces — not just its raw
    street length. A vehicle that "looks" fairly loaded during training can
    still end up with a noticeably longer real route than the rest of the
    fleet purely from this stitching overhead. Restart selection scores
    candidates on this real, solved makespan rather than the training-time
    proxy, so it actually optimizes the number the user sees on screen."""
    lengths: List[float] = []
    for edges in edges_by_vehicle:
        if not edges:
            lengths.append(0.0)
            continue
        sub = nx.MultiGraph()
        for (u, v, key) in edges:
            data = G.get_edge_data(u, v, key)
            sub.add_node(u, **G.nodes[u])
            sub.add_node(v, **G.nodes[v])
            sub.add_edge(u, v, key=key, **data)
        result = solve_circuit(sub, need_naive=False, need_both_curbs=False)
        lengths.append(result["optimal"]["length_m"] if result else 0.0)
    return (max(lengths) if lengths else 0.0), lengths


def _restart_worker(args):
    """One full restart (seed-partition + Q-ELM training + real routed
    -makespan scoring), packaged as a single picklable, top-level call so it
    can be run in a separate process. Restarts share no state with each
    other, so running them in parallel instead of one after another changes
    nothing about the result -- train_fleet_assignment still compares every
    restart's routed makespan in the same fixed order and keeps the same
    winner it would have sequentially; only the wall-clock time changes."""
    G, k, start_nodes, episodes, hidden_dim, restart_seed, max_transfers, r = args
    assigned, dist_by_vehicle, train_makespan, history = _train_one_restart(
        G, k, start_nodes, episodes, hidden_dim, restart_seed, max_transfers
    )
    routed_makespan, _routed_lengths = _routed_makespan(G, assigned)
    return r, assigned, dist_by_vehicle, train_makespan, routed_makespan, history


def train_fleet_assignment(
    G: nx.MultiGraph,
    k: int,
    episodes: int = DEFAULT_EPISODES,
    hidden_dim: int = DEFAULT_HIDDEN,
    seed: int = 42,
    max_transfers: int = 0,
    num_restarts: int = DEFAULT_RESTARTS,
) -> FleetAssignment:
    """Voronoi-seed the partition, then train the multi-agent Q-ELM
    rebalancing policy over many simulated refinement episodes, and run one
    final greedy (epsilon=0) rollout for the assignment actually returned.

    Because real street networks are lumpy rather than uniform, the shape
    of the *initial* Voronoi partition — which depends entirely on which
    node farthest-point sampling happens to start from — can make a real
    difference to how evenly the RL step is able to balance things
    afterwards (rebalancing can only move edges between vehicles that
    already share a boundary, so a bad starting shape can leave one
    vehicle boxed in with no good neighbour to hand load to). Rather than
    committing to one starting shape, this runs num_restarts independent
    attempts — the first from the graph's default starting node (so
    behaviour is unchanged if num_restarts=1), the rest from a handful of
    different randomly-chosen starting nodes — and keeps whichever attempt
    produced the lowest makespan. This costs roughly num_restarts times the
    compute of a single attempt, which is still fast (each attempt is a
    handful of ELM-scored episodes over a small feature vector), in
    exchange for meaningfully more consistent load balancing."""
    t0 = _time.time()
    k = max(1, min(k, MAX_VEHICLES))
    if max_transfers <= 0:
        max_transfers = TRANSFERS_PER_VEHICLE * k
    num_restarts = max(1, num_restarts)

    # A single restart's cost scales with the size of the street graph (each
    # transfer step scans the busiest vehicle's edges), so blindly repeating
    # it DEFAULT_RESTARTS times on a very large fetched area could make one
    # planning request take unreasonably long. Only scale restarts down when
    # the caller left it at the default — an explicit num_restarts always
    # wins.
    if num_restarts == DEFAULT_RESTARTS:
        num_edges = G.number_of_edges()
        if num_edges > 900:
            num_restarts = 2
        elif num_edges > 500:
            num_restarts = 3

    nodes = list(G.nodes)
    picker_rng = np.random.default_rng(seed)

    # Decide every restart's starting nodes up front, sequentially, so the
    # picker_rng draw order — and therefore which starting shape each
    # restart gets — is identical to running restarts one at a time. Only
    # the independent training/scoring work below (which shares no state
    # between restarts) is actually parallelized.
    restart_args = []
    for r in range(num_restarts):
        if r == 0 or len(nodes) <= k:
            start_nodes = _pick_start_nodes(G, k)
        else:
            first_node = nodes[int(picker_rng.integers(len(nodes)))]
            start_nodes = _pick_start_nodes(G, k, first_node=first_node)
        restart_seed = seed + r * 7919  # distinct, deterministic per restart
        restart_args.append((G, k, start_nodes, episodes, hidden_dim, restart_seed, max_transfers, r))

    # Restarts are embarrassingly parallel (each trains its own ELM from its
    # own seed partition, with no shared state) — run them across processes
    # when there's more than one to do and more than one CPU to do it with,
    # falling back to the original sequential loop if that ever fails for
    # any reason (e.g. an environment that disallows subprocesses). Either
    # way, results are combined below in the same fixed restart order as
    # before, so which restart "wins" is unaffected by execution order.
    results = [None] * num_restarts
    max_workers = min(num_restarts, _os.cpu_count() or 1)
    ran_in_parallel = False
    if max_workers > 1:
        try:
            with _ProcessPoolExecutor(max_workers=max_workers) as ex:
                for r, assigned, dist_by_vehicle, train_makespan, routed_makespan, history in ex.map(
                    _restart_worker, restart_args
                ):
                    results[r] = (assigned, dist_by_vehicle, train_makespan, routed_makespan, history)
            ran_in_parallel = True
        except Exception:
            ran_in_parallel = False
    if not ran_in_parallel:
        for args in restart_args:
            r, assigned, dist_by_vehicle, train_makespan, routed_makespan, history = _restart_worker(args)
            results[r] = (assigned, dist_by_vehicle, train_makespan, routed_makespan, history)

    best_assigned = None
    best_dist = None
    best_routed_makespan = float("inf")
    restart_history = []
    for r in range(num_restarts):
        assigned, dist_by_vehicle, train_makespan, routed_makespan, history = results[r]
        restart_history.append({
            "restart": r,
            "train_makespan_m": train_makespan,
            "routed_makespan_m": routed_makespan,
        })
        if routed_makespan < best_routed_makespan:
            best_assigned, best_dist, best_routed_makespan = assigned, dist_by_vehicle, routed_makespan

    train_ms = (_time.time() - t0) * 1000.0

    return FleetAssignment(
        edges_by_vehicle=best_assigned,
        dist_by_vehicle=best_dist,
        makespan_m=best_routed_makespan,
        episodes_run=episodes,
        train_ms=train_ms,
        history=restart_history,
    )


def random_fleet_assignment(G: nx.MultiGraph, k: int, seed: int = 20260827) -> FleetAssignment:
    """Baseline for live, on-screen comparison: every street edge is thrown
    to one of the k vehicles completely at random, with no notion of
    geography, contiguity, or fairness at all — this is what "random
    sweeping" looks like with zero planning. Each vehicle's random chunk is
    still handed to the exact Route Inspection solver, so the only thing
    this measures is how bad the *assignment* is, not the routing. Exists
    purely so a demo can show a concrete, real "before vs. after" distance
    saved by fleet balancing, rather than asserting the improvement in the
    abstract. A fixed default seed keeps the comparison reproducible across
    repeated plans of the same area."""
    t0 = _time.time()
    k = max(1, min(k, MAX_VEHICLES))
    rng = _random.Random(seed)
    edges = list(G.edges(keys=True))
    assigned: List[List[EdgeKey]] = [[] for _ in range(k)]
    for e in edges:
        assigned[rng.randrange(k)].append(e)
    dist_by_vehicle = [
        sum(G.get_edge_data(u, v, key)["weight"] for (u, v, key) in edges_i)
        for edges_i in assigned
    ]
    # Every caller of this function re-solves each vehicle's real route via
    # build_vehicle_routes() anyway (to get an actual path to draw/animate),
    # and derives its own makespan from those real, solved lengths — this
    # dataclass's makespan_m field is never read downstream. So skip the
    # expensive routed solve (_routed_makespan calls the exact CPP solver,
    # including minimum-weight matching, once per vehicle) here and use the
    # free raw territory-weight max instead; it costs nothing extra and
    # avoids solving every vehicle's circuit twice for a number nobody uses.
    makespan_m = max(dist_by_vehicle) if dist_by_vehicle else 0.0
    train_ms = (_time.time() - t0) * 1000.0
    return FleetAssignment(
        edges_by_vehicle=assigned,
        dist_by_vehicle=dist_by_vehicle,
        makespan_m=makespan_m,
        episodes_run=0,
        train_ms=train_ms,
        history=[],
    )


def build_vehicle_routes(
    G: nx.MultiGraph, assignment: FleetAssignment, start_latlng=None, both_curbs: bool = False
) -> List[dict]:
    """Hand each vehicle's RL-assigned edges back to the exact Route
    Inspection solver for that vehicle's own optimal circuit.

    start_latlng is a single (lat, lng) applied to every vehicle (the
    normal case — each vehicle starts wherever is nearest in its own
    territory), or None to let the solver pick.

    both_curbs=True routes each vehicle over its own assigned streets
    twice (once per direction/curb) instead of once. Note this doesn't
    require any change to the balancing logic above: the Voronoi partition
    and RL rebalancing only ever compare *relative* loads between vehicles
    (ratios to a shared target share), and scaling every street's weight
    by the same factor (here, driving it twice instead of once) doesn't
    change which split is balanced — so a single fleet_rl run already
    produces the right assignment for either mode."""
    vehicles = []
    for i, edges in enumerate(assignment.edges_by_vehicle):
        this_start = start_latlng
        if not edges:
            vehicles.append({"path": [], "lengthM": 0.0, "steps": 0})
            continue
        sub = nx.MultiGraph()
        for (u, v, key) in edges:
            data = G.get_edge_data(u, v, key)
            sub.add_node(u, **G.nodes[u])
            sub.add_node(v, **G.nodes[v])
            sub.add_edge(u, v, key=key, **data)
        # Only ever one of these two is actually used below — request just
        # that one so solve_circuit skips the other entirely. This matters
        # a lot for both_curbs=True (the app's default): that variant needs
        # no matching at all, so this avoids running the expensive
        # min-weight-matching "optimal" solve for every vehicle for nothing.
        result = solve_circuit(sub, this_start, need_optimal=not both_curbs, need_naive=False, need_both_curbs=both_curbs)
        if result is None:
            vehicles.append({"path": [], "lengthM": 0.0, "steps": 0})
        else:
            chosen = result["both_curbs"] if both_curbs else result["optimal"]
            vehicles.append({
                "path": chosen["path"],
                "lengthM": chosen["length_m"],
                "steps": len(chosen["path"]) - 1,
            })
    return vehicles
