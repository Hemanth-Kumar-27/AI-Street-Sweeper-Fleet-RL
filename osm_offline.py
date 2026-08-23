"""
Offline OpenStreetMap road-network loader.

Fixes the "operator won't sit and trace streets" problem without depending
on any live Overpass API call at demo/runtime. Overpass mirrors can be
blocked by a network's firewall — that's a network-policy issue, not
something fixable in app code. The reliable workaround: export the road
geometry for the operating area ONCE, ahead of time, from any machine that
can reach the internet (openstreetmap.org itself, not a community mirror),
save the file into data/, and the backend loads roads from disk from then
on. Zero manual tracing, zero live network dependency at runtime.

How to get the file (one-time setup, not something the operator ever does):
  1. Go to https://www.openstreetmap.org in a browser that has normal
     internet access.
  2. Pan/zoom to the area the sweeper will operate in.
  3. Click "Export" (top toolbar) -> "Manually select a different area" to
     fine-tune the box if needed -> "Export" button. This downloads a
     `map.osm` file directly (no signup, no API key).
     - If the area is too large, the page will instead suggest Overpass API
       or other bulk-download sources (e.g. https://extract.bbbike.org) —
       use one of those for bigger regions.
  4. Save/rename the downloaded file into this folder as data/roads.osm
     (any *.osm file in data/ is picked up automatically; the first match
     wins if there are several).
  5. Restart the Flask app (or just replan — the loader re-reads the file
     whenever its modified time changes).

From then on, "Street" and "Street + open area" modes in the UI read this
local file first and only fall back to a live Overpass call if no offline
file is present.
"""

from __future__ import annotations

import glob
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence, Tuple

LatLng = Tuple[float, float]
BBox = Tuple[float, float, float, float]  # (min_lat, min_lon, max_lat, max_lon)

HIGHWAY_TYPES = re.compile(
    r"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|"
    r"service|living_street|pedestrian)$"
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# abspath -> (mtime, file_bbox, [(bbox, [(lat, lon), ...]), ...])
_cache: Dict[str, Tuple[float, Optional[BBox], List[Tuple[BBox, List[LatLng]]]]] = {}


def _parse_osm_file(path: str) -> Tuple[Optional[BBox], List[Tuple[BBox, List[LatLng]]]]:
    """Parse a standard OSM XML export into (file_bbox, [(way_bbox, coords), ...]).

    file_bbox is the area this *extract itself* covers (from the <bounds>
    tag OpenStreetMap's Export writes at the top of the file, falling back
    to the union of all node coordinates if that tag is missing) — distinct
    from any individual street's bbox. Needed to tell "this file has no
    streets in the drawn area because there really are none" apart from
    "this file doesn't cover the drawn area at all", which used to be
    treated identically and silently returned zero streets for areas
    outside the exported extract. Streams with iterparse + elem.clear() so
    large extracts don't blow up memory."""
    nodes: Dict[str, LatLng] = {}
    ways: List[Tuple[BBox, List[LatLng]]] = []
    file_bbox: Optional[BBox] = None

    context = ET.iterparse(path, events=("end",))
    for _, elem in context:
        if elem.tag == "bounds":
            try:
                file_bbox = (
                    float(elem.get("minlat")), float(elem.get("minlon")),
                    float(elem.get("maxlat")), float(elem.get("maxlon")),
                )
            except (TypeError, ValueError):
                pass
            elem.clear()
        elif elem.tag == "node":
            nid = elem.get("id")
            lat = elem.get("lat")
            lon = elem.get("lon")
            if nid is not None and lat is not None and lon is not None:
                nodes[nid] = (float(lat), float(lon))
            elem.clear()
        elif elem.tag == "way":
            tags = {}
            refs = []
            for child in elem:
                if child.tag == "nd":
                    refs.append(child.get("ref"))
                elif child.tag == "tag" and child.get("k"):
                    tags[child.get("k")] = child.get("v")
            highway = tags.get("highway")
            if highway and HIGHWAY_TYPES.match(highway):
                coords = [nodes[r] for r in refs if r in nodes]
                if len(coords) >= 2:
                    lats = [c[0] for c in coords]
                    lons = [c[1] for c in coords]
                    ways.append(((min(lats), min(lons), max(lats), max(lons)), coords))
            elem.clear()

    if file_bbox is None and nodes:
        all_lats = [p[0] for p in nodes.values()]
        all_lons = [p[1] for p in nodes.values()]
        file_bbox = (min(all_lats), min(all_lons), max(all_lats), max(all_lons))

    return file_bbox, ways


def _find_offline_file() -> Optional[str]:
    if not os.path.isdir(DATA_DIR):
        return None
    candidates = sorted(
        f for f in glob.glob(os.path.join(DATA_DIR, "*.osm")) if os.path.isfile(f)
    )
    return candidates[0] if candidates else None


def offline_file_available() -> bool:
    return _find_offline_file() is not None


def load_offline_roads(area_latlng: Sequence[LatLng], pad: float = 0.0006) -> Optional[List[List[LatLng]]]:
    """Return road ways overlapping the drawn area's bbox from a locally
    saved .osm file in data/, or None if no such file exists OR the drawn
    area falls entirely outside what the file covers (caller should fall
    back to a live fetch in either case).

    Those are deliberately treated the same: an offline extract only ever
    covers one exported neighbourhood, and a query for anywhere else isn't
    "genuinely zero streets", it's "wrong file for this area" — silently
    returning an empty result for that used to look identical to a real
    empty area and block the live fallback that should have kicked in.
    A real "this area is covered by the file but genuinely has no streets"
    case (e.g. query drawn entirely inside a park within the extract) still
    correctly returns an empty list, not None.

    Each way is a plain [[lat, lng], ...] polyline."""
    path = _find_offline_file()
    if not path:
        return None

    mtime = os.path.getmtime(path)
    cached = _cache.get(path)
    if cached is None or cached[0] != mtime:
        file_bbox, ways = _parse_osm_file(path)
        _cache[path] = (mtime, file_bbox, ways)
    else:
        _, file_bbox, ways = cached

    lats = [p[0] for p in area_latlng]
    lons = [p[1] for p in area_latlng]
    s, n = min(lats) - pad, max(lats) + pad
    w, e = min(lons) - pad, max(lons) + pad

    if file_bbox is not None:
        blat0, blon0, blat1, blon1 = file_bbox
        if blat1 < s - pad or blat0 > n + pad or blon1 < w - pad or blon0 > e + pad:
            return None  # drawn area isn't covered by this extract at all — try live

    result = []
    for (blat0, blon0, blat1, blon1), coords in ways:
        if blat1 < s or blat0 > n or blon1 < w or blon0 > e:
            continue
        result.append([[lat, lon] for lat, lon in coords])
    return result
