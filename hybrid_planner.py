"""
Hybrid coverage planner: Route Inspection (Chinese Postman) for the street
network + Q-ELM reinforcement learning for open pockets attached to it,
stitched into one continuous route.

Why this exists: a real neighborhood is rarely purely streets or purely
open space — it's usually a street network with open surfaces branching
off it (a parking lot, a plaza, a courtyard). Route Inspection alone can't
cover those; it only knows about street edges, not 2D open areas. A pure
open-area grid sweep alone would ignore the actual street topology. This
is also the honest answer to "why use RL at all if CPP solves streets
exactly": RL-ELM does something CPP fundamentally cannot — learn a
coverage policy over an open 2D region with no predefined graph — so the
two methods aren't competing for the same job, they're solving two
different sub-problems that both show up in the same real environment.

Pipeline:
  1. Rasterize the whole drawn area, and separately rasterize the street
     corridor (same road-buffer logic as the old grid engine, reused here
     purely to classify "close to a street" vs. not).
  2. Whatever's inside the area, not an obstacle, and NOT in the street
     corridor is a candidate "open pocket". Flood-fill 4-connected
     components; small ones are rasterization noise and are dropped.
  3. Each surviving pocket is covered with the same Q-ELM trainer +
     boustrophedon-guaranteed fallback used in open-area mode.
  4. Each pocket's coverage path is spliced into the CPP route as an
     out-and-back detour at the nearest point along that route — a real
     vehicle can't teleport into a parking lot, it has to drive off the
     road and back onto it.
"""

from __future__ import annotations

import math
import time as _time
from collections import deque
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

import geo
import sweeper_rl_elm as rl

LatLng = Tuple[float, float]

POCKET_CELL_SIZE = 1.5        # metres — fixed resolution for open-pocket coverage
DEFAULT_CORRIDOR_WIDTH = 4.0  # metres — how close to a street counts as "on it"
MIN_POCKET_CELLS = 12         # filters rasterization noise from becoming a "pocket"
MAX_POCKETS = 5               # caps total RL training time if the area is very open
POCKET_EPISODES = 50


def _label_components(mask: np.ndarray, target: int = 1) -> List[set]:
    """4-connected component labeling via BFS flood fill."""
    rows, cols = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: List[set] = []
    for r0 in range(rows):
        for c0 in range(cols):
            if mask[r0, c0] != target or seen[r0, c0]:
                continue
            comp = set()
            q = deque([(r0, c0)])
            seen[r0, c0] = True
            while q:
                r, c = q.popleft()
                comp.add((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] == target and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def find_open_pockets(
    area_latlng: Sequence[LatLng],
    obstacles_latlng: Sequence[Sequence[LatLng]],
    street_ways_latlng: Sequence,
    cell_size: float = POCKET_CELL_SIZE,
    corridor_width: float = DEFAULT_CORRIDOR_WIDTH,
    min_pocket_cells: int = MIN_POCKET_CELLS,
    max_pockets: int = MAX_POCKETS,
) -> Tuple[geo.Grid, List[np.ndarray]]:
    """Rasterize the drawn area, subtract the street corridor, and return
    connected components of what's left as separate pocket masks (largest
    first, capped at max_pockets)."""
    plain_ways = [w for w in street_ways_latlng if len(w) >= 2]

    full_grid = geo.rasterize(area_latlng, obstacles_latlng, cell_size)
    if plain_ways:
        street_grid = geo.rasterize_roads(area_latlng, obstacles_latlng, plain_ways, cell_size, corridor_width)
        street_mask = street_grid.mask
    else:
        street_mask = np.zeros_like(full_grid.mask)

    open_mask = ((full_grid.mask == 1) & (street_mask != 1)).astype(int)
    components = [c for c in _label_components(open_mask, target=1) if len(c) >= min_pocket_cells]
    components.sort(key=len, reverse=True)
    components = components[:max_pockets]

    pockets = []
    for comp in components:
        mask = np.zeros_like(full_grid.mask)
        for (r, c) in comp:
            mask[r, c] = rl.DIRTY
        pockets.append(mask)
    return full_grid, pockets


@dataclass
class PocketResult:
    path_latlng: List[LatLng]
    length_m: float
    cells: int
    coverage: float
    time_ms: float


def _path_length_m(grid: geo.Grid, path_latlng: List[LatLng]) -> float:
    total = 0.0
    for i in range(len(path_latlng) - 1):
        x1, y1 = grid._to_xy(*path_latlng[i])
        x2, y2 = grid._to_xy(*path_latlng[i + 1])
        total += math.hypot(x2 - x1, y2 - y1)
    return total


def solve_pocket(mask: np.ndarray, grid: geo.Grid, episodes: int = POCKET_EPISODES) -> PocketResult:
    """RL-ELM coverage for one open pocket — the same trainer used in
    open-area mode, converted to a lat/lng path."""
    t0 = _time.time()
    dirty_count = int(np.sum(mask == rl.DIRTY))
    hidden_dim = max(50, min(120, dirty_count // 4 or 50))
    result = rl.train_q_elm(mask, episodes=episodes, hidden_dim=hidden_dim)
    rollout = rl.rollout_greedy(mask, result.elm)
    covered = rollout["covered"]
    path = list(rollout["path"])
    if rollout["coverage"] < 0.999:
        fb = rl.fallback_complete(mask, covered, path[-1])
        covered = fb["covered"]
        path += fb["extra_path"][1:]
    coverage = float(np.sum(covered & (mask == rl.DIRTY))) / dirty_count
    path_latlng = [list(grid.cell_center_latlng(r, c)) for (r, c) in path]
    return PocketResult(
        path_latlng=path_latlng,
        length_m=_path_length_m(grid, path_latlng),
        cells=dirty_count,
        coverage=coverage,
        time_ms=(_time.time() - t0) * 1000.0,
    )


def _nearest_index(path_latlng: List[LatLng], target: LatLng) -> int:
    best_i, best_d = 0, float("inf")
    tx, ty = target
    for i, (lat, lng) in enumerate(path_latlng):
        d = (lat - tx) ** 2 + (lng - ty) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


def stitch_route(main_path: List[LatLng], pockets: List[PocketResult]) -> Tuple[List[LatLng], float]:
    """Insert each pocket's coverage path as an out-and-back detour at the
    point along the main CPP route nearest to that pocket, in the order
    those points occur along the route (so indices stay valid without an
    offset table)."""
    if not pockets:
        return list(main_path), 0.0

    insertions = sorted(
        ((_nearest_index(main_path, p.path_latlng[0]), p) for p in pockets),
        key=lambda t: t[0],
    )

    stitched: List[LatLng] = []
    detour_length_m = 0.0
    last_idx = 0
    for idx, pocket in insertions:
        stitched.extend(main_path[last_idx:idx + 1])
        anchor = main_path[idx]
        stitched.extend(pocket.path_latlng)  # drive in and cover the pocket
        stitched.append(anchor)              # drive back out to the same road point
        detour_length_m += pocket.length_m
        last_idx = idx + 1
    stitched.extend(main_path[last_idx:])
    return stitched, detour_length_m


def nearest_vehicle(vehicle_paths: List[List[LatLng]], target: LatLng) -> int:
    """Which vehicle's route currently passes closest to `target` — used to
    hand each open pocket to whichever sweeper is already nearest it when
    the fleet has been split across multiple vehicles, rather than always
    stitching every pocket onto a single route."""
    best_i, best_d = 0, float("inf")
    tx, ty = target
    for i, path in enumerate(vehicle_paths):
        if not path:
            continue
        idx = _nearest_index(path, target)
        lat, lng = path[idx]
        d = (lat - tx) ** 2 + (lng - ty) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i
