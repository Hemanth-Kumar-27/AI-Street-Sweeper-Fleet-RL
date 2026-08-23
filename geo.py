"""
Lat/lng <-> local-metre projection and grid rasterization.

Python port of the geometry half of algo.js (the client-side engine used in
Sweeper_RL_ELM_Path_Planner.html): turns a drawn polygon (and optional
obstacles / street centerlines) into the same rows x cols mask that
sweeper_rl_elm.py's CoverageGridEnv expects (0=outside, 1=dirty/walkable,
2=obstacle).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

LatLng = Tuple[float, float]


def make_projector(ref_lat: float) -> Tuple[Callable[[float, float], Tuple[float, float]], Callable[[float, float], LatLng]]:
    """Equirectangular projection around a reference latitude — matches the
    approximation used in algo.js closely enough for city-block-scale areas."""
    R = 6378137.0  # metres
    lat_rad = math.radians(ref_lat)
    m_per_deg_lat = math.radians(1.0) * R
    m_per_deg_lng = math.radians(1.0) * R * math.cos(lat_rad)

    def to_xy(lat: float, lng: float) -> Tuple[float, float]:
        return lng * m_per_deg_lng, lat * m_per_deg_lat

    def to_latlng(x: float, y: float) -> LatLng:
        return y / m_per_deg_lat, x / m_per_deg_lng

    return to_xy, to_latlng


def point_in_polygon(pt: Tuple[float, float], poly: Sequence[Tuple[float, float]]) -> bool:
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def dist_to_segment(px, py, ax, ay, bx, by) -> float:
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


@dataclass
class Grid:
    rows: int
    cols: int
    cell_size: float
    min_x: float
    min_y: float
    mask: np.ndarray  # rows x cols, values 0/1/2
    _to_latlng: Callable[[float, float], LatLng]
    _to_xy: Callable[[float, float], Tuple[float, float]]

    @property
    def dirty_count(self) -> int:
        return int(np.sum(self.mask == 1))

    def cell_center_latlng(self, r: int, c: int) -> LatLng:
        x = self.min_x + (c + 0.5) * self.cell_size
        y = self.min_y + (r + 0.5) * self.cell_size
        return self._to_latlng(x, y)

    def latlng_to_nearest_cell(self, latlng: LatLng) -> Tuple[int, int]:
        x, y = self._to_xy(latlng[0], latlng[1])
        c = int((x - self.min_x) // self.cell_size)
        r = int((y - self.min_y) // self.cell_size)
        r = max(0, min(self.rows - 1, r))
        c = max(0, min(self.cols - 1, c))
        return r, c


def _grid_dims(area_xy, cell_size, max_cells):
    xs = [p[0] for p in area_xy]
    ys = [p[1] for p in area_xy]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    cs = cell_size
    cols = max(1, math.ceil((max_x - min_x) / cs))
    rows = max(1, math.ceil((max_y - min_y) / cs))
    while rows * cols > max_cells:
        cs *= 1.25
        cols = max(1, math.ceil((max_x - min_x) / cs))
        rows = max(1, math.ceil((max_y - min_y) / cs))
    return rows, cols, cs, min_x, min_y


def rasterize(
    area_latlng: Sequence[LatLng],
    obstacles_latlng: Optional[Sequence[Sequence[LatLng]]] = None,
    cell_size: float = 1.5,
    max_cells: int = 2200,
) -> Grid:
    obstacles_latlng = obstacles_latlng or []
    ref_lat = sum(p[0] for p in area_latlng) / len(area_latlng)
    to_xy, to_latlng = make_projector(ref_lat)
    area_xy = [to_xy(lat, lng) for lat, lng in area_latlng]
    obstacles_xy = [[to_xy(lat, lng) for lat, lng in poly] for poly in obstacles_latlng]

    rows, cols, cs, min_x, min_y = _grid_dims(area_xy, cell_size, max_cells)

    mask = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        cy = min_y + (r + 0.5) * cs
        for c in range(cols):
            cx = min_x + (c + 0.5) * cs
            if point_in_polygon((cx, cy), area_xy):
                blocked = any(point_in_polygon((cx, cy), obs) for obs in obstacles_xy)
                mask[r, c] = 2 if blocked else 1

    return Grid(rows, cols, cs, min_x, min_y, mask, to_latlng, to_xy)


def rasterize_roads(
    area_latlng: Sequence[LatLng],
    obstacles_latlng: Optional[Sequence[Sequence[LatLng]]],
    road_ways_latlng: Optional[Sequence[Sequence[LatLng]]],
    cell_size: float,
    road_width: float,
    max_cells: int = 2200,
) -> Grid:
    """Only marks cells within road_width/2 of a road/street centerline as
    sweepable. Falls back to full-area rasterize() if no ways are given."""
    if not road_ways_latlng:
        return rasterize(area_latlng, obstacles_latlng, cell_size, max_cells)

    obstacles_latlng = obstacles_latlng or []
    ref_lat = sum(p[0] for p in area_latlng) / len(area_latlng)
    to_xy, to_latlng = make_projector(ref_lat)
    area_xy = [to_xy(lat, lng) for lat, lng in area_latlng]
    obstacles_xy = [[to_xy(lat, lng) for lat, lng in poly] for poly in obstacles_latlng]
    roads_xy = [[to_xy(lat, lng) for lat, lng in way] for way in road_ways_latlng]

    rows, cols, cs, min_x, min_y = _grid_dims(area_xy, cell_size, max_cells)
    half_width = road_width / 2 + cs * 0.4

    # Bucket road segments into grid-cell bins for fast nearest-segment lookup.
    bins: dict = {}
    for way in roads_xy:
        for i in range(len(way) - 1):
            ax, ay = way[i]
            bx, by = way[i + 1]
            x0, x1 = min(ax, bx) - half_width, max(ax, bx) + half_width
            y0, y1 = min(ay, by) - half_width, max(ay, by) + half_width
            bc0, bc1 = int((x0 - min_x) // cs), int((x1 - min_x) // cs)
            br0, br1 = int((y0 - min_y) // cs), int((y1 - min_y) // cs)
            for br in range(br0, br1 + 1):
                for bc in range(bc0, bc1 + 1):
                    bins.setdefault((bc, br), []).append((ax, ay, bx, by))

    mask = np.zeros((rows, cols), dtype=int)
    for r in range(rows):
        cy = min_y + (r + 0.5) * cs
        for c in range(cols):
            cx = min_x + (c + 0.5) * cs
            if not point_in_polygon((cx, cy), area_xy):
                continue
            if any(point_in_polygon((cx, cy), obs) for obs in obstacles_xy):
                continue
            segs = bins.get((c, r), [])
            if any(dist_to_segment(cx, cy, ax, ay, bx, by) <= half_width for (ax, ay, bx, by) in segs):
                mask[r, c] = 1

    return Grid(rows, cols, cs, min_x, min_y, mask, to_latlng, to_xy)
