# Offline road data goes here

Drop a `.osm` file in this folder (e.g. `roads.osm`) and the app will read
road geometry from it with zero live network calls — this is what makes
"Auto-fetch streets" work reliably even when Overpass mirrors are blocked
by a firewall.

## How to get the file (one-time, not something the operator ever does)

1. On any machine/browser with normal internet access, go to
   https://www.openstreetmap.org
2. Pan/zoom to the area the sweeper will operate in.
3. Click **Export** (top toolbar). Use "Manually select a different area"
   to fine-tune the box, then click the **Export** button — this downloads
   a `map.osm` file directly. No signup, no API key.
   - If OSM says the area is too large for direct export, it will suggest
     Overpass API or another bulk source (e.g. https://extract.bbbike.org)
     — use one of those instead for big regions.
4. Move/rename the downloaded file into this folder, e.g. `roads.osm`.
5. Restart the Flask app (or just re-plan — the loader re-reads the file
   whenever its modified time changes).

From then on, "Auto-fetch streets from OpenStreetMap" mode reads this file
first and only tries a live Overpass call if no `.osm` file is found here.
