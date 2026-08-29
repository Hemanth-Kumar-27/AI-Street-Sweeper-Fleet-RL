"""
Flask backend for the sweeper coverage path planner.

Serves a Leaflet map (templates/index.html) where you draw a sweep area and
click "Plan". Streets are always auto-fetched from OpenStreetMap — no
manual tracing. Two modes, because they're genuinely different problems:

- "Street": solved with Route Inspection / the Chinese Postman Problem
  (route_inspection.py), directly on the real street graph. No
  rasterization involved, so the path always sits exactly on the streets.
  This is classical graph theory, not learned — it's an exactly-solvable
  problem, so there's no reason to spend a learning method on it.
- "Street + open area": Route Inspection still solves the streets exactly,
  then any open pockets it can't reach (a parking lot, plaza, courtyard
  with no street network) are covered by a Q-ELM reinforcement-learning
  agent (sweeper_rl_elm.py) and spliced into the route as detours
  (hybrid_planner.py). This is where the RL actually earns its place: it
  solves the part of the problem — open 2D coverage with no predefined
  graph — that Route Inspection structurally cannot.

Either mode can be split across a fleet of sweepers ("Number of sweeper
machines" in the UI). Routing a single vehicle is always exact (CPP), but
*deciding which vehicle covers which streets* so no truck drives twice as
long as the rest of the fleet is a different, NP-hard problem with no
closed-form solution — solved here with cooperative multi-agent
reinforcement learning (fleet_rl.py), not a fixed geometric split.

Run with:
    pip install -r requirements.txt
    python app.py
then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import matplotlib
matplotlib.use("Agg")  # headless — this process never opens a display
import matplotlib.pyplot as plt
import networkx as nx
from flask import Flask, jsonify, render_template, request

import fleet_rl
import hybrid_planner as hp
import osm_offline
import route_inspection as ri

try:
    import seaborn as sns
    sns.set_theme(style="darkgrid")
except ImportError:
    # seaborn is optional — matplotlib ships an equivalent built-in style,
    # so the chart still looks right even if seaborn isn't installed.
    plt.style.use("seaborn-v0_8-darkgrid")

app = Flask(__name__)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
HIGHWAY_TYPES = "motorway|trunk|primary|secondary|tertiary|unclassified|residential|service|living_street|pedestrian"

FLEET_COLORS = ["#23c98f", "#3d8bff", "#ff9f43", "#c44dff", "#ff5d7a", "#2ec4b6"]


def num_sweepers(data: dict) -> int:
    """How many sweeper machines to split the route across. 1 means no
    fleet planning at all — the single-vehicle CPP route is used as-is."""
    try:
        n = int(data.get("numSweepers", 1))
    except (TypeError, ValueError):
        n = 1
    return max(1, min(n, fleet_rl.MAX_VEHICLES))


def both_curbs_flag(data: dict) -> bool:
    """Whether every street must be driven once in each direction (both
    curbs swept), not just covered at least once. Defaults on — real
    curb-to-curb sweeping needs both passes; off reproduces the original
    single-pass behaviour for comparison."""
    return bool(data.get("bothCurbs", True))


def build_fleet_response(vehicles: list, assignment: fleet_rl.FleetAssignment) -> dict:
    out = []
    for i, v in enumerate(vehicles):
        out.append({
            "id": i + 1,
            "color": FLEET_COLORS[i % len(FLEET_COLORS)],
            "path": v["path"],
            "lengthM": v["lengthM"],
            "steps": v["steps"],
        })
    lengths = [v["lengthM"] for v in out]
    return {
        "numVehicles": len(out),
        "vehicles": out,
        "makespanM": max(lengths) if lengths else 0.0,
        "totalLengthM": sum(lengths) if lengths else 0.0,
        "trainMs": assignment.train_ms,
        "episodesRun": assignment.episodes_run,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/preview", methods=["POST"])
def preview():
    data = request.get_json(force=True)
    area = data["area"]
    ways = data.get("roadWays") or []
    obstacles = data.get("obstacles") or []

    try:
        G = ri.build_road_graph(ways, area, obstacles)
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": str(e)}), 400
    G.remove_nodes_from(list(nx.isolates(G)))
    total_m = sum(d["weight"] for _, _, d in G.edges(data=True))
    return jsonify({
        "numStreets": G.number_of_edges(),
        "streetLengthM": total_m,
    })


@app.route("/api/fetch_roads", methods=["POST"])
def fetch_roads():
    """Road data source, tried in order:
    1. A locally-saved .osm export in data/ (osm_offline.py) — zero live
       network calls, works even when every Overpass mirror is firewalled.
       This is the primary path for a real deployment: prepare the extract
       once, ahead of time, and the operator never touches the network.
    2. Live Overpass mirrors, proxied server-side — used only if no offline
       file has been prepared yet.
    """
    data = request.get_json(force=True)
    area = data["area"]

    offline_ways = osm_offline.load_offline_roads(area)
    if offline_ways is not None:
        return jsonify({"ok": True, "ways": offline_ways, "endpoint": "offline extract (data/*.osm)", "offline": True})

    lats = [p[0] for p in area]
    lngs = [p[1] for p in area]
    pad = 0.0003
    s, n = min(lats) - pad, max(lats) + pad
    w, e = min(lngs) - pad, max(lngs) + pad
    query = f'[out:json][timeout:20];(way["highway"~"^({HIGHWAY_TYPES})$"]({s},{w},{n},{e}););out geom;'

    attempts = []
    for endpoint in OVERPASS_ENDPOINTS:
        url = endpoint + "?data=" + urllib.parse.quote(query)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "capstone-sweeper-planner/1.0"})
            with urllib.request.urlopen(req, timeout=12) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            ways = [
                [[g["lat"], g["lon"]] for g in el["geometry"]]
                for el in payload.get("elements", [])
                if el.get("type") == "way" and el.get("geometry") and len(el["geometry"]) > 1
            ]
            return jsonify({"ok": True, "ways": ways, "endpoint": endpoint})
        except Exception as e:  # noqa: BLE001
            attempts.append(f"{endpoint}: {e}")

    if osm_offline.offline_file_available():
        attempts.append(
            "An offline extract exists in data/ but doesn't cover this area — "
            "either draw within the exported neighbourhood, or export a new "
            "map.osm covering this area and add it to data/."
        )
    else:
        attempts.append(
            "No offline extract found either — add one at data/roads.osm "
            "(see osm_offline.py for how to export it) to avoid live fetches entirely."
        )
    return jsonify({"ok": False, "attempts": attempts}), 502


def plan_street(ways, area, start, k=1, both_curbs=True, obstacles=None):
    """Street mode: solve the real street graph directly with Route
    Inspection (Chinese Postman) — no grid, the route always sits exactly
    on the streets. If k > 1, the street edges are also split across a
    fleet via multi-agent RL task allocation (fleet_rl.py).

    both_curbs=True (default): every street must be driven once in each
    direction, so both curbs actually get swept. Once that's a hard
    requirement, the classic CPP odd-degree matching isn't needed at all —
    duplicating every street automatically makes the graph Eulerian — so
    there's no "optimal vs naive" story left for a single vehicle; every
    valid route covering each street twice has the exact same length. The
    comparison shown instead is honest and still meaningful: single pass
    (misses a curb) vs. both curbs covered (the real requirement)."""
    t0 = time.time()
    G = ri.build_road_graph(ways, area, obstacles)
    # "optimal" is always shown (either as baseline or as the improved
    # route); "naive"/"both_curbs" are mutually exclusive depending on the
    # toggle below, so only the one actually needed is computed.
    result = ri.solve_circuit(G, start, need_naive=not both_curbs, need_both_curbs=both_curbs)
    solve_ms = (time.time() - t0) * 1000.0

    if result is None:
        return jsonify({"error": "No usable streets found in this area — try a different location."}), 400

    original_m = result["original_length_m"]

    if both_curbs:
        baseline, improved = result["optimal"], result["both_curbs"]
        labels = {
            "baseline": "Single pass (each street once — misses one curb)",
            "improved": "Both curbs covered (each street driven once per direction)",
        }
    else:
        baseline, improved = result["naive"], result["optimal"]
        labels = {
            "baseline": "Naive (out-and-back on every street)",
            "improved": "Optimal route (Chinese Postman / Route Inspection)",
        }

    response = {
        "engine": "street",
        "bothCurbs": both_curbs,
        "labels": labels,
        "streets": {"numStreets": result["num_streets"], "streetLengthM": original_m},
        "baseline": {
            "path": baseline["path"],
            "steps": len(baseline["path"]) - 1,
            "lengthM": baseline["length_m"],
            "timeMs": solve_ms,
        },
        "rl": {
            "path": improved["path"],
            "steps": len(improved["path"]) - 1,
            "lengthM": improved["length_m"],
            "overheadPct": max(0.0, (improved["length_m"] - original_m) / original_m * 100) if original_m > 0 else 0.0,
            "timeMs": solve_ms,
        },
    }

    if k > 1:
        assignment = fleet_rl.train_fleet_assignment(G, k)
        vehicles = fleet_rl.build_vehicle_routes(G, assignment, start, both_curbs=both_curbs)
        response["fleet"] = build_fleet_response(vehicles, assignment)

        # Random-assignment baseline, computed alongside the real plan so
        # the UI can show a live "random sweeping vs. fleet balancing"
        # distance-saved comparison — see fleet_rl.random_fleet_assignment.
        random_assignment = fleet_rl.random_fleet_assignment(G, k)
        random_vehicles = fleet_rl.build_vehicle_routes(G, random_assignment, start, both_curbs=both_curbs)
        response["fleetRandom"] = build_fleet_response(random_vehicles, random_assignment)

    return jsonify(response)


def _fleet_with_pockets(G, assignment, start, both_curbs, pockets):
    """Shared by the Q-ELM and random-assignment fleets in hybrid mode:
    solve each vehicle's assigned streets, then stitch on whichever open
    pockets fall nearest that vehicle's own route."""
    vehicles = fleet_rl.build_vehicle_routes(G, assignment, start, both_curbs=both_curbs)
    vehicle_paths = [v["path"] for v in vehicles]
    pockets_by_vehicle = [[] for _ in vehicles]
    for p in pockets:
        pockets_by_vehicle[hp.nearest_vehicle(vehicle_paths, p.path_latlng[0])].append(p)
    for i, v in enumerate(vehicles):
        if pockets_by_vehicle[i]:
            stitched, _ = hp.stitch_route(v["path"], pockets_by_vehicle[i])
            v["path"] = stitched
            v["lengthM"] = ri.path_length_m(stitched)
            v["steps"] = len(stitched) - 1
    return build_fleet_response(vehicles, assignment)


def plan_hybrid(ways, area, start, k=1, both_curbs=True, obstacles=None):
    """Street + open area mode: Route Inspection (CPP) covers the street
    network exactly; any open pockets it doesn't reach (parking lots,
    plazas, courtyards) are covered with Q-ELM reinforcement learning and
    spliced into the route as out-and-back detours. This is the honest
    answer to "why use RL if CPP solves streets exactly" — RL does the
    part CPP structurally can't (2D open-area coverage with no predefined
    graph), not the part it can.

    both_curbs applies only to the street portion (an open plaza has no
    "curb sides" to double-cover, so open-pocket coverage is unaffected).

    If k > 1, the street edges are additionally split across a fleet via
    multi-agent RL (fleet_rl.py), and each open pocket is stitched onto
    whichever vehicle's route currently passes nearest it."""
    if not ways:
        return jsonify({"error": "No streets found in this area — try a different location."}), 400

    t0 = time.time()
    G = ri.build_road_graph(ways, area, obstacles)
    # Hybrid mode never looks at "naive"; only one of optimal/both_curbs is
    # actually used as street_route below, depending on the toggle.
    cpp_result = ri.solve_circuit(G, start, need_optimal=not both_curbs, need_naive=False, need_both_curbs=both_curbs)
    if cpp_result is None:
        return jsonify({"error": "No usable streets found in this area — try a different location."}), 400

    street_route = cpp_result["both_curbs"] if both_curbs else cpp_result["optimal"]

    grid, pocket_masks = hp.find_open_pockets(area, obstacles or [], ways)
    pockets = [hp.solve_pocket(mask, grid) for mask in pocket_masks]
    stitched_path, _ = hp.stitch_route(street_route["path"], pockets)
    solve_ms = (time.time() - t0) * 1000.0

    original_m = cpp_result["original_length_m"]
    streets_only_m = street_route["length_m"]
    combined_m = ri.path_length_m(stitched_path)

    response = {
        "engine": "hybrid",
        "bothCurbs": both_curbs,
        "labels": {
            "baseline": "Streets only, both curbs" if both_curbs else "Streets only (Chinese Postman)",
            "improved": "Streets (both curbs) + open areas" if both_curbs else "Streets + open areas (CPP + RL-ELM)",
        },
        "streets": {"numStreets": cpp_result["num_streets"], "streetLengthM": original_m},
        "pockets": {
            "count": len(pockets),
            "totalCells": sum(p.cells for p in pockets),
            "totalLengthM": sum(p.length_m for p in pockets),
            "avgCoverage": (sum(p.coverage for p in pockets) / len(pockets)) if pockets else None,
        },
        "baseline": {
            "path": street_route["path"],
            "steps": len(street_route["path"]) - 1,
            "lengthM": streets_only_m,
            "timeMs": solve_ms,
        },
        "rl": {
            "path": stitched_path,
            "steps": len(stitched_path) - 1,
            "lengthM": combined_m,
            "timeMs": solve_ms,
        },
    }

    if k > 1:
        assignment = fleet_rl.train_fleet_assignment(G, k)
        response["fleet"] = _fleet_with_pockets(G, assignment, start, both_curbs, pockets)

        # Random-assignment baseline (see plan_street for why) — pockets are
        # stitched on the same way so the comparison stays apples-to-apples.
        random_assignment = fleet_rl.random_fleet_assignment(G, k)
        response["fleetRandom"] = _fleet_with_pockets(G, random_assignment, start, both_curbs, pockets)

    return jsonify(response)


@app.route("/api/plan", methods=["POST"])
def plan():
    data = request.get_json(force=True)
    area = data["area"]
    ways = data.get("roadWays") or []
    obstacles = data.get("obstacles") or []
    start_latlng = data.get("start")
    start = tuple(start_latlng) if start_latlng else None
    k = num_sweepers(data)
    both_curbs = both_curbs_flag(data)

    mode = data.get("mode", "street")
    if mode == "hybrid":
        return plan_hybrid(ways, area, start, k, both_curbs, obstacles)
    return plan_street(ways, area, start, k, both_curbs, obstacles)


def _render_batch_chart_png(rows: list, summary: dict) -> str:
    """Bar-per-random-draw chart with a flat reference line for the one
    Q-ELM plan, rendered server-side with matplotlib/seaborn and returned as
    a base64 PNG data URI — the frontend just drops this straight into an
    <img> tag, so there's no client-side charting library or CDN involved."""
    trials = [r["trial"] for r in rows]
    random_km = [r["randomMakespanM"] / 1000.0 for r in rows]
    qelm_km = summary["qelmMakespanM"] / 1000.0
    avg_random_km = summary["avgRandomMakespanM"] / 1000.0
    avg_saved_km = max(0.0, avg_random_km - qelm_km)
    avg_saved_pct = summary["avgSavedPct"]

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=130)
    ax.bar(trials, random_km, color="#8a8f99", label="Random sweeping (per draw)", zorder=2)
    ax.axhline(qelm_km, color="#23c98f", linewidth=2.5,
               label=f"Q-ELM fleet balancing ({qelm_km:.2f} km)", zorder=3)
    ax.set_xlabel("Random-sweeping trial")
    ax.set_ylabel("Longest-truck distance (km)")
    ax.set_title(
        f"Random sweeping ({summary['numTrials']} draws) vs. Q-ELM — fleet of {summary['fleetSize']}\n"
        f"Average distance saved: {avg_saved_km:.2f} km ({avg_saved_pct:.1f}%)",
        fontsize=12,
    )
    # Headroom above the tallest bar/line so the legend sits in empty space
    # instead of covering any bars, regardless of how tall they happen to be.
    top = max(random_km + [qelm_km]) if random_km else qelm_km
    ax.set_ylim(0, top * 1.3)
    ax.legend(loc="upper center", ncol=2, frameon=True, framealpha=0.9, fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.route("/api/compare_random_batch", methods=["POST"])
def compare_random_batch():
    """Run one real Q-ELM fleet plan against many independent random-sweeping
    draws on the exact same area and fleet size, so the "distance saved"
    number shown live in the UI can be backed by a whole distribution of
    random outcomes, not just whichever single random seed happened to be
    used on screen. Returns per-trial rows, summary stats, and a ready-to-
    display chart image."""
    data = request.get_json(force=True)
    area = data["area"]
    ways = data.get("roadWays") or []
    obstacles = data.get("obstacles") or []
    start_latlng = data.get("start")
    start = tuple(start_latlng) if start_latlng else None
    k = num_sweepers(data)
    both_curbs = both_curbs_flag(data)
    try:
        num_trials = max(1, min(int(data.get("numTrials", 30)), 200))
    except (TypeError, ValueError):
        num_trials = 30

    if k <= 1:
        return jsonify({"error": "Pick a fleet size of 2 or more sweepers before running this comparison."}), 400

    G = ri.build_road_graph(ways, area, obstacles)
    if G.number_of_edges() == 0:
        return jsonify({"error": "No usable streets found in this area — try a different location."}), 400

    assignment = fleet_rl.train_fleet_assignment(G, k)
    qelm_vehicles = fleet_rl.build_vehicle_routes(G, assignment, start, both_curbs=both_curbs)
    qelm_lengths = [v["lengthM"] for v in qelm_vehicles]
    qelm_makespan = max(qelm_lengths) if qelm_lengths else 0.0

    rows = []
    for i in range(num_trials):
        seed = 1000 + i  # distinct, reproducible per trial
        random_assignment = fleet_rl.random_fleet_assignment(G, k, seed=seed)
        random_vehicles = fleet_rl.build_vehicle_routes(G, random_assignment, start, both_curbs=both_curbs)
        random_lengths = [v["lengthM"] for v in random_vehicles]
        random_makespan = max(random_lengths) if random_lengths else 0.0
        saved_m = max(0.0, random_makespan - qelm_makespan)
        saved_pct = (saved_m / random_makespan * 100.0) if random_makespan else 0.0
        rows.append({
            "trial": i + 1,
            "seed": seed,
            "qelmMakespanM": qelm_makespan,
            "randomMakespanM": random_makespan,
            "savedM": saved_m,
            "savedPct": saved_pct,
        })

    saved_pcts = [r["savedPct"] for r in rows]
    summary = {
        "numTrials": num_trials,
        "fleetSize": k,
        "qelmMakespanM": qelm_makespan,
        "avgRandomMakespanM": sum(r["randomMakespanM"] for r in rows) / len(rows),
        "avgSavedPct": sum(saved_pcts) / len(saved_pcts),
        "minSavedPct": min(saved_pcts),
        "maxSavedPct": max(saved_pcts),
    }
    chart_png = _render_batch_chart_png(rows, summary)
    return jsonify({"rows": rows, "summary": summary, "chartPngBase64": chart_png})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
