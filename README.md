# Sweeper Coverage Web App (Python / Flask)

Interactive map tool: draw the area to sweep and click **Plan**. Streets are
always fetched automatically from OpenStreetMap — there's no manual
tracing. Two modes, because they're genuinely different problems:

- **Street**: solved with **Route Inspection** — the classical "Chinese
  Postman Problem" — directly on the real street graph
  (`route_inspection.py`, using `networkx`). No rasterization at all, so
  the route always sits exactly on the streets. This is classical graph
  theory, not learned: it's an exactly-solvable problem, so there's no
  reason to spend a learning method on it.
- **Street + open area**: Route Inspection still solves the streets
  exactly, then any open pockets it can't reach (a parking lot, plaza,
  courtyard with no street network) are covered by a **Q-ELM**
  reinforcement-learning agent (`sweeper_rl_elm.py`) and spliced into the
  route as detours (`hybrid_planner.py`). This is where the RL earns its
  place — it solves the part of the problem Route Inspection structurally
  cannot: open 2D coverage with no predefined graph.

Either mode can be split across a **fleet of sweepers** ("Number of
sweeper machines" in the UI, 1–6). Routing a single vehicle is always
exact (CPP) — but *deciding which vehicle covers which streets* so no
truck drives twice as long as the rest of the fleet is the "min-max
multi-vehicle arc routing problem," which is NP-hard and has no
closed-form solution. That's solved here with cooperative multi-agent
reinforcement learning (`fleet_rl.py`), not a fixed geometric split: a
Voronoi seed partition (deterministic, guarantees each vehicle's territory
is spatially contiguous) is refined by a shared Q-ELM policy that
repeatedly hands a boundary street from the busiest vehicle to a
neighbour, learning which transfers actually shrink the makespan (the
longest single vehicle's shift — the fleet's real bottleneck). Each
vehicle's final street assignment is then handed back to the exact CPP
solver for its own optimal route.

**Sweep both curbs.** Real street sweeping needs a pass in each
direction — a truck's brush only reaches the curb on one side per pass,
so a two-sided street has to be driven twice. This is on by default ("🧹
Sweep both curbs" toggle in the UI). It's modeled as a directed graph:
every street becomes two directed edges (one per direction), which
automatically balances every intersection's in/out-degree — so the graph
is already Eulerian with no odd-degree matching step needed. That also
means there's no "optimal vs naive" story left for a single vehicle once
both curbs are required: any route covering every street twice has
exactly the same total length. The comparison shown instead is single
pass (misses a curb) vs. both curbs covered — the honest extra cost of
doing it correctly. The fleet-balancing RL is unaffected either way: it
only compares relative loads between vehicles, and doubling every
street's weight uniformly doesn't change which split is balanced.

## Run it

```
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## How it works

- `route_inspection.py` — builds a weighted graph straight from street
  polylines (nodes = intersections/endpoints, edges = real street
  segments), bridges any disconnected pieces, then solves the Chinese
  Postman Problem exactly: minimum-weight perfect matching (via
  `networkx.max_weight_matching`) over the odd-degree nodes' shortest
  paths, duplicates those edges, and walks the resulting Eulerian circuit.
  Also computes a trivial "naive" route (every street doubled, no
  optimization) as an honest comparison baseline, and a "both curbs" route
  (every street as two directed edges, one per direction) for correct
  curb-to-curb coverage.
- `hybrid_planner.py` — rasterizes the drawn area, subtracts the street
  corridor, and flood-fills what's left into "open pockets"; each pocket is
  covered with the Q-ELM agent and spliced into the CPP route as an
  out-and-back detour.
- `fleet_rl.py` — multi-agent Q-ELM fleet task allocation: Voronoi-seeds a
  contiguous per-vehicle partition of the street graph, then trains a
  shared Q-ELM policy across many simulated rebalancing episodes to decide
  which boundary streets to hand off between vehicles to minimize the
  makespan. Each vehicle's final edge set is routed exactly via
  `route_inspection.solve_circuit`.
- `geo.py` — turns a lat/lng polygon into a grid, used by the hybrid
  engine's pocket-finding step.
- `sweeper_rl_elm.py` — the RL algorithm: `ELM` (Extreme Learning
  Machine), `CoverageGridEnv` (the RL environment), `train_q_elm`
  (Q-learning with the ELM refit after every episode), `rollout_greedy`,
  and `fallback_complete` (guarantees 100% final coverage of a pocket even
  across disconnected regions, by bridging unreachable cells with a direct
  transit hop instead of giving up on them).
- `app.py` — Flask routes: `/api/preview` (street-count/length estimate),
  `/api/plan` (routes to the street or hybrid engine, returns paths as
  lat/lng lists plus mode-appropriate metrics), and `/api/fetch_roads`
  (road data — see below for where it comes from).
- `osm_offline.py` — loads road geometry from a locally-saved `.osm` file
  in `data/`, so the app can auto-fetch streets with **zero live network
  calls** at demo/runtime. See "Auto-fetch reliability" below.
- `static/app.js` — the map UI (drawing the area, mode switching,
  animation, an animated toy-sweeper marker during playback). No algorithm
  code in it; every "Plan" click is just a fetch to `/api/plan`.

## Auto-fetch reliability: offline road extracts

Overpass API mirrors are community-run and frequently blocked by corporate
or campus firewalls — that's a network-policy issue, not a bug in this app,
and no amount of retrying or adding mirrors fixes it reliably. The real fix:
prepare a **local `.osm` file** for the operating area once, ahead of time,
on any machine with normal internet access, and drop it in `data/`.

1. On a machine that can reach the internet normally, open
   https://www.openstreetmap.org, pan/zoom to the operating area, click
   **Export** → adjust the box → **Export** — downloads `map.osm` directly
   (no signup, no API key). For large regions, use
   https://extract.bbbike.org instead.
2. Save it as `data/roads.osm` in this folder.
3. That's it — `/api/fetch_roads` reads this file first from then on.
   No internet call, no manual tracing, every time the operator draws an
   area and clicks Plan. Full details in `osm_offline.py`.

If no file is present, auto-fetch falls back to live Overpass mirrors as
before (and will report per-mirror errors if the server's network blocks
those too).
