"""
Street-network coverage via Route Inspection (the "Chinese Postman Problem").

Why this replaces the grid+RL-ELM pipeline for street-following sweeps:
rasterizing a street network into a grid and having an agent (RL-ELM or
otherwise) learn to cover it is fragile — cell size vs. road-buffer width
has to be tuned per area, coarse grids fragment connectivity, and the
result is an approximate blob rather than a path that actually sits on the
street centerlines. None of that is a good fit for "cover every street with
the shortest possible route."

Route Inspection is the exact right tool for this: build a graph directly
from the real street geometry (nodes = intersections/endpoints, edges =
street segments, weighted by real-world length), then find the shortest
closed walk that traverses every edge at least once. This is the same
class of algorithm used for real mail delivery, snow-plowing, and street
-sweeping route optimization.

Algorithm (classical, exact for this problem class):
  1. Build a weighted multigraph straight from the fetched street
     polylines — no rasterization at all, so the path always sits exactly
     on the streets.
  2. If the street data forms more than one connected piece (common with
     real OSM extracts — dead ends, a street barely clipped by the query
     box, etc.), bridge the pieces with the shortest possible connector
     edges so a single continuous route is possible.
  3. Nodes with odd degree can't be part of a closed walk that uses every
     edge exactly once — you'd get stuck. Fix this by finding the minimum
     -total-length way to duplicate some edges (a "T-join" via minimum
     weight perfect matching over the odd-degree nodes' shortest paths) so
     every node ends up with even degree. This is the Chinese Postman
     Problem and is solved *exactly* here (not a heuristic).
  4. Once every node has even degree the graph is Eulerian — walk an
     Eulerian circuit, which visits every edge exactly once, giving the
     shortest possible route that covers every street at least once.

For comparison we also report a trivial "naive" route (drive out-and-back
on every street with no optimization, i.e. every edge duplicated) — this is
always a safe upper bound and gives an honest before/after number.

This is deliberately the ONLY classical (non-learned) piece of the
pipeline: it's a well-posed, exactly-solvable combinatorial problem, so
there's no reason to spend a learning method on it. RL (sweeper_rl_elm.py)
is reserved for the part of the problem this can't solve — open 2D areas
with no graph structure — see hybrid_planner.py.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from geo import make_projector, point_in_polygon

LatLng = Tuple[float, float]

NODE_PRECISION = 6  # ~0.11m at the equator; merges shared OSM endpoints


def _key(p: LatLng) -> Tuple[float, float]:
    return (round(p[0], NODE_PRECISION), round(p[1], NODE_PRECISION))


def build_road_graph(
    road_ways_latlng: Sequence[Sequence[LatLng]],
    area_latlng: Sequence[LatLng],
    obstacles_latlng: Optional[Sequence[Sequence[LatLng]]] = None,
) -> nx.MultiGraph:
    """Build a weighted multigraph directly from street polylines. Nodes are
    keyed by rounded (lat, lng) so shared endpoints between ways (real
    intersections) merge into a single graph node automatically — no
    spatial-index snapping needed since OSM data already shares exact
    coordinates at junctions."""
    obstacles_latlng = obstacles_latlng or []
    ref_lat = sum(p[0] for p in area_latlng) / len(area_latlng)
    to_xy, _ = make_projector(ref_lat)
    area_xy = [to_xy(lat, lng) for lat, lng in area_latlng]
    obstacles_xy = [[to_xy(lat, lng) for lat, lng in poly] for poly in obstacles_latlng]

    G = nx.MultiGraph()
    for points in road_ways_latlng:
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            mx, my = to_xy((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            if not point_in_polygon((mx, my), area_xy):
                continue
            if any(point_in_polygon((mx, my), obs) for obs in obstacles_xy):
                continue
            k1, k2 = _key(p1), _key(p2)
            if k1 == k2:
                continue
            x1, y1 = to_xy(*p1)
            x2, y2 = to_xy(*p2)
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist < 0.05:  # skip near-duplicate points
                continue
            G.add_node(k1, latlng=p1)
            G.add_node(k2, latlng=p2)
            G.add_edge(k1, k2, weight=dist)
    return G


def bridge_components(G: nx.MultiGraph) -> nx.MultiGraph:
    """If the street data isn't one connected piece, connect the pieces with
    the shortest possible connector edges (nearest-node-to-nearest-node)
    until the whole graph is traversable in one route. Marked with
    transit=True so the frontend can render them distinctly (a real vehicle
    driving a short unmapped/disconnected link).

    The nearest-pair search below is the exact same "cheap planar approx"
    distance used before (raw lat/lng hypot, picking the true minimum over
    every merged-node/other-node pair) -- it's just done as one vectorized
    numpy distance-matrix computation per candidate component instead of a
    pure-Python double loop, so the chosen bridge (and therefore the whole
    route) is identical, only faster on real OSM extracts that fragment
    into many small disconnected pieces."""
    G = G.copy()
    comps = [list(c) for c in nx.connected_components(G)]
    if len(comps) <= 1:
        return G

    coords = {n: G.nodes[n]["latlng"] for n in G.nodes}

    comps.sort(key=len, reverse=True)
    merged: List = list(comps[0])
    remaining: List[List] = [list(c) for c in comps[1:]]
    while remaining:
        merged_xy = np.array([coords[n] for n in merged], dtype=np.float64)
        best = None  # (dist, node_in_merged, node_in_other, other_idx)
        for idx, comp in enumerate(remaining):
            comp_xy = np.array([coords[n] for n in comp], dtype=np.float64)
            diff = merged_xy[:, None, :] - comp_xy[None, :, :]
            d2 = np.einsum("ijk,ijk->ij", diff, diff)
            i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
            d = math.sqrt(float(d2[i, j]))
            if best is None or d < best[0]:
                best = (d, merged[i], comp[j], idx)
        _, a, b, idx = best
        (alat, alng) = coords[a]
        (blat, blng) = coords[b]
        ref_lat = (alat + blat) / 2
        to_xy, _ = make_projector(ref_lat)
        x1, y1 = to_xy(alat, alng)
        x2, y2 = to_xy(blat, blng)
        G.add_edge(a, b, weight=math.hypot(x2 - x1, y2 - y1), transit=True)
        merged.extend(remaining.pop(idx))
    return G


def _eulerize_min(G: nx.MultiGraph) -> nx.MultiGraph:
    """Exact Chinese Postman augmentation: duplicate the minimum *real
    -distance* total length of edges needed to make every node even degree.

    (networkx's built-in nx.eulerize() matches on hop-count, not distance —
    fine for uniform grids but wrong here since street segments vary wildly
    in length. This does the same matching approach but weighted by actual
    metres, which is the textbook Chinese Postman solution.)"""
    G = nx.MultiGraph(G)
    odd = [n for n, d in G.degree() if d % 2 == 1]
    if not odd:
        return G

    # One single-source Dijkstra per odd node (distances to every other node
    # at once) instead of one Dijkstra per PAIR of odd nodes — O(k) full
    # searches instead of O(k^2) targeted ones, which matters once a real
    # street network has a few hundred dead-ends/T-junctions.
    #
    # (A target-limited early-stopping variant was tried here and verified
    # exactly correct, but benchmarked no faster — and sometimes slightly
    # slower — on this project's own grid-shaped test networks, where the
    # odd-degree nodes are the whole perimeter rather than a small, localized
    # minority, so early-stopping rarely triggers before most of the graph
    # is explored anyway. Kept as the plain networkx call since it's no
    # slower here and simpler.)
    odd_set = set(odd)
    Gp = nx.Graph()
    for m in odd:
        lengths, paths = nx.single_source_dijkstra(G, m, weight="weight")
        for n in odd_set:
            if n <= m or n not in lengths:
                continue
            Gp.add_edge(m, n, weight=-lengths[n], path=paths[n])  # negate: matching maximizes

    matching = nx.max_weight_matching(Gp, maxcardinality=True)
    for m, n in matching:
        path = Gp[m][n]["path"]
        for a, b in zip(path, path[1:]):
            edge_data = min(G.get_edge_data(a, b).values(), key=lambda d: d["weight"])
            G.add_edge(a, b, **edge_data)
    return G


def _eulerize_naive(G: nx.MultiGraph) -> nx.MultiGraph:
    """Trivial always-valid baseline: duplicate every single edge once.
    Doubling every edge doubles every node's degree, so it's automatically
    even — no matching needed. This is what "drive out onto every street
    and straight back" costs, used purely as an honest upper-bound
    comparison for the optimized route."""
    G2 = nx.MultiGraph(G)
    for u, v, data in list(G.edges(data=True)):
        G2.add_edge(u, v, **data)
    return G2


def _to_directed_both_curbs(G: nx.MultiGraph) -> "nx.MultiDiGraph":
    """Every street segment becomes two directed edges — one for each
    direction of travel, i.e. one pass per curb. Real curb-to-curb
    sweeping needs a pass in each direction (a truck's brush only reaches
    one side per pass); simply duplicating an *undirected* edge doesn't
    guarantee that — an Eulerian circuit is free to traverse both copies
    in the same direction if the rest of the graph allows it, which would
    sweep one curb twice and skip the other entirely. Directed edges rule
    that out structurally.

    Because every directed edge is balanced by its exact reverse, every
    node's in-degree equals its out-degree automatically — so the result
    is always Eulerian with no augmentation/matching step needed at all
    (unlike the single-pass case, which needs the odd-degree matching in
    _eulerize_min). That's not a shortcut, it's just the math: once every
    street is required to be driven exactly twice, any route that does so
    has exactly the same total length (2x the street network), so there
    is nothing left to optimize for a single vehicle — see solve_circuit.
    """
    D = nx.MultiDiGraph()
    D.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        D.add_edge(u, v, **data)
        D.add_edge(v, u, **data)
    return D


def _circuit_path_directed(G: "nx.MultiDiGraph", start_key) -> Tuple[List[LatLng], float]:
    if start_key not in G:
        start_key = next(iter(G.nodes))
    if G.number_of_edges() == 0 or not nx.is_eulerian(G):
        latlng = G.nodes[start_key]["latlng"]
        return [latlng], 0.0
    circuit = list(nx.eulerian_circuit(G, source=start_key, keys=True))
    path = [G.nodes[circuit[0][0]]["latlng"]]
    total = 0.0
    for u, v, k in circuit:
        path.append(G.nodes[v]["latlng"])
        total += G.edges[u, v, k]["weight"]
    return path, total


def _circuit_path(G: nx.MultiGraph, start_key) -> Tuple[List[LatLng], float]:
    if start_key not in G:
        start_key = next(iter(G.nodes))
    if not nx.is_eulerian(G):
        # Shouldn't happen after eulerizing a connected graph, but guard
        # against degenerate single-node/no-edge cases.
        latlng = G.nodes[start_key]["latlng"]
        return [latlng], 0.0
    circuit = list(nx.eulerian_circuit(G, source=start_key, keys=True))
    path = [G.nodes[circuit[0][0]]["latlng"]]
    total = 0.0
    for u, v, k in circuit:
        path.append(G.nodes[v]["latlng"])
        total += G.edges[u, v, k]["weight"]
    return path, total


def solve_circuit(
    G: nx.MultiGraph,
    start_latlng: Optional[LatLng] = None,
    need_optimal: bool = True,
    need_naive: bool = True,
    need_both_curbs: bool = True,
) -> Optional[dict]:
    """Given an already-built street multigraph (possibly disconnected,
    possibly just a subset of a larger graph), bridge any disconnected
    pieces and solve Chinese Postman on it exactly, returning the requested
    circuit variant(s).

    This is the piece shared by the single-vehicle pipeline
    (solve_route_inspection, below) and per-vehicle fleet sub-routes after
    RL-based task allocation (fleet_rl.py) — routing itself is always
    solved exactly and classically; only *which streets a given vehicle
    is responsible for* is ever a learned decision.

    The "optimal" (single-pass) variant is the only one that needs the
    minimum-weight matching step (_eulerize_min) — by far the most
    expensive part of this function on larger graphs. Callers that only
    ever look at "both_curbs" (the app's default sweeping mode) or only at
    "naive" should pass need_optimal=False so that matching never runs at
    all; this matters a lot when solve_circuit is called many times in a
    loop (once per vehicle, once per random-sweeping trial, once per
    Q-ELM restart). All three default to True so existing callers that
    want the full comparison keep working unchanged.

    Returns None if G has no edges.
    """
    G = G.copy()
    G.remove_nodes_from(list(nx.isolates(G)))
    if G.number_of_edges() == 0:
        return None

    G = bridge_components(G)

    if start_latlng is not None:
        start_key = min(
            G.nodes,
            key=lambda n: (G.nodes[n]["latlng"][0] - start_latlng[0]) ** 2
            + (G.nodes[n]["latlng"][1] - start_latlng[1]) ** 2,
        )
    else:
        start_key = next(iter(G.nodes))

    original_length = sum(d["weight"] for _, _, d in G.edges(data=True))
    num_streets = G.number_of_edges()

    result = {"original_length_m": original_length, "num_streets": num_streets}

    if need_optimal:
        G_opt = _eulerize_min(G)
        opt_path, opt_length = _circuit_path(G_opt, start_key)
        result["optimal"] = {"path": opt_path, "length_m": opt_length}

    if need_naive:
        G_naive = _eulerize_naive(G)
        naive_path, naive_length = _circuit_path(G_naive, start_key)
        result["naive"] = {"path": naive_path, "length_m": naive_length}

    if need_both_curbs:
        G_both = _to_directed_both_curbs(G)
        both_path, both_length = _circuit_path_directed(G_both, start_key)
        result["both_curbs"] = {"path": both_path, "length_m": both_length}

    return result


def solve_route_inspection(
    road_ways_latlng: Sequence[Sequence[LatLng]],
    area_latlng: Sequence[LatLng],
    obstacles_latlng: Optional[Sequence[Sequence[LatLng]]] = None,
    start_latlng: Optional[LatLng] = None,
) -> Optional[dict]:
    """Full pipeline: build graph from street polylines -> solve_circuit.
    Returns None if there's no usable street graph (e.g. no roads in area).
    """
    G = build_road_graph(road_ways_latlng, area_latlng, obstacles_latlng)
    return solve_circuit(G, start_latlng)


def path_length_m(path_latlng: Sequence[LatLng]) -> float:
    """Real-world length of a lat/lng polyline, in metres."""
    if len(path_latlng) < 2:
        return 0.0
    ref_lat = sum(p[0] for p in path_latlng) / len(path_latlng)
    to_xy, _ = make_projector(ref_lat)
    total = 0.0
    prev = to_xy(*path_latlng[0])
    for p in path_latlng[1:]:
        cur = to_xy(*p)
        total += math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        prev = cur
    return total
