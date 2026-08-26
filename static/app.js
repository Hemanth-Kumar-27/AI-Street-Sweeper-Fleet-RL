(function () {
  "use strict";

  // ---------------- Map setup ----------------
  const map = L.map("map", { zoomControl: true }).setView([12.9716, 77.5946], 18);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 21,
    attribution: "&copy; OpenStreetMap contributors",
  }).addTo(map);

  // ---------------- Backend helpers ----------------
  async function api(path, payload) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      const err = new Error(data.error || `Request to ${path} failed (HTTP ${res.status})`);
      err.data = data;
      throw err;
    }
    return data;
  }

  // ---------------- State ----------------
  let mode = "idle"; // idle | drawing-area | picking-start | drawing-obstacle
  let areaPoints = [];
  let areaLayer = null;
  let areaVertexLayers = [];
  let areaFinalized = false;

  let startPoint = null;  // [lat, lng] or null — optional route start/end
  let startMarker = null;

  let obstacles = [];           // array of polygons, each an array of [lat,lng]
  let obstacleLayers = [];      // finalized obstacle polygon layers
  let currentObstaclePoints = [];
  let obstacleVertexLayers = [];
  let obstaclePreviewLine = null;

  let lastLabels = { baseline: "Baseline", improved: "Improved path" };

  let previewLine = null;

  let roadWays = null; // fetched from /api/fetch_roads
  let roadLayers = [];
  let roadFetchInFlight = false;

  let baselineResult = null;
  let rlResult = null;
  let fleetResult = null;   // { numVehicles, vehicles:[{id,color,path,lengthM,steps}], makespanM, totalLengthM, trainMs, episodesRun }
  let baseLineLayer = null;
  let fleetLineLayers = []; // one polyline per vehicle, parallel to fleetResult.vehicles

  let animTimer = null;
  let animEntries = [];     // one entry per simultaneously-animating vehicle/path: {pathLatLngs, cumDist, totalDist, dist, seg, marker}

  // ---------------- DOM refs ----------------
  const $ = (id) => document.getElementById(id);
  const btnDrawArea = $("btnDrawArea");
  const btnFinishArea = $("btnFinishArea");
  const btnUndoArea = $("btnUndoArea");
  const areaStatus = $("areaStatus");
  const btnSetStart = $("btnSetStart");
  const btnClearStart = $("btnClearStart");
  const startHint = $("startHint");
  const btnDrawObstacle = $("btnDrawObstacle");
  const btnFinishObstacle = $("btnFinishObstacle");
  const btnUndoObstacle = $("btnUndoObstacle");
  const btnClearObstacles = $("btnClearObstacles");
  const obstacleHint = $("obstacleHint");
  const obstacleStatus = $("obstacleStatus");

  const roadStatus = $("roadStatus");
  const bothCurbsToggle = $("bothCurbsToggle");

  const btnPlan = $("btnPlan");
  const progressBar = $("progressBar");
  const planStatus = $("planStatus");
  const btnReset = $("btnReset");

  const numSweepersSlider = $("numSweepersSlider");
  const numSweepersVal = $("numSweepersVal");

  const resultsCard = $("resultsCard");
  const metricsTable = $("metricsTable");
  const toggleBase = $("toggleBase");
  const animSelect = $("animSelect");
  const btnPlay = $("btnPlay");
  const btnPause = $("btnPause");
  const btnStopAnim = $("btnStopAnim");
  const speedSlider = $("speedSlider");
  const speedVal = $("speedVal");
  const animBadge = $("animBadge");

  const geoInput = $("geoInput");
  const geoBtn = $("geoBtn");

  // ---------------- Geocoding (Nominatim) ----------------
  geoBtn.addEventListener("click", doGeocode);
  geoInput.addEventListener("keydown", (e) => { if (e.key === "Enter") doGeocode(); });
  function doGeocode() {
    const q = geoInput.value.trim();
    if (!q) return;
    geoBtn.disabled = true;
    geoBtn.textContent = "…";
    fetch("https://nominatim.openstreetmap.org/search?format=json&limit=1&q=" + encodeURIComponent(q))
      .then((r) => r.json())
      .then((res) => {
        if (res && res[0]) map.setView([parseFloat(res[0].lat), parseFloat(res[0].lon)], 19);
        else alert("Location not found.");
      })
      .catch(() => alert("Search failed (network unavailable)."))
      .finally(() => { geoBtn.disabled = false; geoBtn.textContent = "Go"; });
  }

  // ---------------- Road network fetching (proxied through Flask) ----------------
  function updateRoadStatusVisibility() {
    roadStatus.style.display = (roadFetchInFlight || roadWays !== null) ? "block" : "none";
  }

  function clearRoadLayers() {
    for (const l of roadLayers) map.removeLayer(l);
    roadLayers = [];
  }

  async function fetchRoads() {
    if (!areaFinalized) return;
    roadFetchInFlight = true;
    roadWays = null;
    clearRoadLayers();
    roadStatus.style.display = "block";
    roadStatus.textContent = "Asking the Python backend to fetch road data from OpenStreetMap…";
    try {
      const res = await api("/api/fetch_roads", { area: areaPoints });
      if (!res.ok) {
        const detail = (res.attempts || []).join("<br>");
        roadStatus.innerHTML = `Backend couldn't reach OpenStreetMap.<br><span style="opacity:.7;">Details: ${detail}</span> <a href="#" id="retryRoadsLink" style="color:var(--accent);">Retry</a>`;
        const retryLink = $("retryRoadsLink");
        if (retryLink) retryLink.addEventListener("click", (ev) => { ev.preventDefault(); fetchRoads(); });
        roadWays = null;
      } else {
        roadWays = res.ways;
        for (const points of roadWays) {
          roadLayers.push(L.polyline(points, { color: "#ffffff", weight: 1, opacity: 0.5 }).addTo(map));
        }
        roadStatus.textContent = roadWays.length
          ? `Found ${roadWays.length} road segment(s) via ${res.endpoint}.`
          : "No mapped roads found in this area — try a different location.";
      }
    } catch (e) {
      roadWays = null;
      // /api/fetch_roads returns HTTP 502 (not just {ok:false}) when every
      // mirror fails, so the shared api() helper throws before the `!res.ok`
      // branch above ever runs — the per-mirror attempts list was getting
      // silently swallowed here. err.data (set by api()) still has it.
      const attempts = e.data && e.data.attempts;
      roadStatus.innerHTML = attempts
        ? `Backend couldn't reach OpenStreetMap.<br><span style="opacity:.7;">Details: ${attempts.join("<br>")}</span> <a href="#" id="retryRoadsLink" style="color:var(--accent);">Retry</a>`
        : `Couldn't reach the backend's road-fetch endpoint.<br><span style="opacity:.7;">Details: ${e.message}</span> <a href="#" id="retryRoadsLink" style="color:var(--accent);">Retry</a>`;
      const retryLink = $("retryRoadsLink");
      if (retryLink) retryLink.addEventListener("click", (ev) => { ev.preventDefault(); fetchRoads(); });
    } finally {
      roadFetchInFlight = false;
      updateGridPreview();
    }
  }

  updateRoadStatusVisibility();

  bothCurbsToggle.addEventListener("change", resetPlanResultsOnly);

  numSweepersSlider.addEventListener("input", () => {
    numSweepersVal.textContent = numSweepersSlider.value;
  });

  // ---------------- Drawing: area ----------------
  btnDrawArea.addEventListener("click", () => {
    resetPlanResultsOnly();
    mode = "drawing-area";
    areaPoints = [];
    clearLayerList(areaVertexLayers);
    if (areaLayer) { map.removeLayer(areaLayer); areaLayer = null; }
    clearRoadLayers();
    roadWays = null;
    roadStatus.style.display = "none";
    areaFinalized = false;
    clearStartPoint();
    resetObstacles();
    btnDrawArea.classList.add("active");
    btnFinishArea.disabled = true;
    btnUndoArea.disabled = true;
    areaStatus.textContent = "Click points on the map to trace the area (need ≥ 3).";
    updatePlanButton();
  });

  btnUndoArea.addEventListener("click", () => {
    if (mode !== "drawing-area" || !areaPoints.length) return;
    areaPoints.pop();
    const layer = areaVertexLayers.pop();
    if (layer) map.removeLayer(layer);
    redrawPreview(areaPoints, "#3aa0ff");
    btnFinishArea.disabled = areaPoints.length < 3;
    btnUndoArea.disabled = areaPoints.length === 0;
    areaStatus.textContent = areaPoints.length + " point(s) placed.";
  });

  btnFinishArea.addEventListener("click", () => {
    if (areaPoints.length < 3) return;
    mode = "idle";
    btnDrawArea.classList.remove("active");
    areaFinalized = true;
    if (previewLine) { map.removeLayer(previewLine); previewLine = null; }
    areaLayer = L.polygon(areaPoints, { color: "#3aa0ff", weight: 2, fillOpacity: 0.08 }).addTo(map);
    btnFinishArea.disabled = true;
    btnUndoArea.disabled = true;
    areaStatus.textContent = "Area set (" + areaPoints.length + " vertices). Fetching streets…";
    updatePlanButton();
    fetchRoads();
  });

  // ---------------- Map click router ----------------
  map.on("click", (e) => {
    const ll = [e.latlng.lat, e.latlng.lng];
    if (mode === "drawing-area") {
      areaPoints.push(ll);
      const v = L.circleMarker(ll, { radius: 4, color: "#3aa0ff", fillColor: "#3aa0ff", fillOpacity: 1 }).addTo(map);
      areaVertexLayers.push(v);
      redrawPreview(areaPoints, "#3aa0ff");
      btnFinishArea.disabled = areaPoints.length < 3;
      btnUndoArea.disabled = false;
      areaStatus.textContent = areaPoints.length + " point(s) placed.";
    } else if (mode === "picking-start") {
      setStartPoint(ll);
      mode = "idle";
      btnSetStart.classList.remove("active");
      startHint.textContent = "Start point set — click \"Set start point\" again to move it.";
    } else if (mode === "drawing-obstacle") {
      currentObstaclePoints.push(ll);
      const v = L.circleMarker(ll, { radius: 4, color: "#ff5c6c", fillColor: "#ff5c6c", fillOpacity: 1 }).addTo(map);
      obstacleVertexLayers.push(v);
      redrawObstaclePreview();
      btnFinishObstacle.disabled = currentObstaclePoints.length < 3;
      btnUndoObstacle.disabled = false;
    }
  });

  // ---------------- Start point (optional) ----------------
  function setStartPoint(ll) {
    startPoint = ll;
    if (startMarker) map.removeLayer(startMarker);
    startMarker = L.marker(ll, {
      icon: L.divIcon({
        className: "",
        html: '<div style="font-size:26px; line-height:1; transform:translate(-3px,-22px);">📍</div>',
        iconSize: [26, 26],
        iconAnchor: [13, 26],
      }),
    }).addTo(map);
    btnClearStart.disabled = false;
  }

  function clearStartPoint() {
    startPoint = null;
    if (startMarker) { map.removeLayer(startMarker); startMarker = null; }
    btnClearStart.disabled = true;
    btnSetStart.classList.remove("active");
    if (mode === "picking-start") mode = "idle";
    startHint.textContent = "Optional — click a point on the map for where the sweeper begins/ends its route. Leave unset and the solver picks one for you.";
  }

  btnSetStart.addEventListener("click", () => {
    if (btnSetStart.disabled) return;
    mode = "picking-start";
    btnSetStart.classList.add("active");
    startHint.textContent = "Click a point on the map to set the start location…";
  });

  btnClearStart.addEventListener("click", clearStartPoint);

  // ---------------- Obstacles (optional) ----------------
  function redrawObstaclePreview() {
    if (obstaclePreviewLine) { map.removeLayer(obstaclePreviewLine); obstaclePreviewLine = null; }
    if (currentObstaclePoints.length < 2) return;
    obstaclePreviewLine = L.polygon(currentObstaclePoints, { color: "#ff5c6c", weight: 2, fillOpacity: 0.08, dashArray: "4,4" }).addTo(map);
  }

  function updateObstacleStatus() {
    if (!obstacles.length) { obstacleStatus.style.display = "none"; return; }
    obstacleStatus.style.display = "block";
    obstacleStatus.textContent = obstacles.length + " obstacle(s) marked.";
  }

  function resetObstacles() {
    obstacles = [];
    clearLayerList(obstacleLayers);
    clearLayerList(obstacleVertexLayers);
    if (obstaclePreviewLine) { map.removeLayer(obstaclePreviewLine); obstaclePreviewLine = null; }
    currentObstaclePoints = [];
    btnClearObstacles.disabled = true;
    btnFinishObstacle.disabled = true;
    btnUndoObstacle.disabled = true;
    btnDrawObstacle.classList.remove("active");
    if (mode === "drawing-obstacle") mode = "idle";
    updateObstacleStatus();
    obstacleHint.textContent = "Optional — trace any no-go zones inside the area (construction, closed lot). Streets and open-area coverage route around them.";
  }

  btnDrawObstacle.addEventListener("click", () => {
    if (btnDrawObstacle.disabled) return;
    mode = "drawing-obstacle";
    currentObstaclePoints = [];
    clearLayerList(obstacleVertexLayers);
    if (obstaclePreviewLine) { map.removeLayer(obstaclePreviewLine); obstaclePreviewLine = null; }
    btnDrawObstacle.classList.add("active");
    btnFinishObstacle.disabled = true;
    btnUndoObstacle.disabled = true;
    obstacleHint.textContent = "Click points to trace the obstacle boundary (need ≥ 3), then Finish.";
  });

  btnUndoObstacle.addEventListener("click", () => {
    if (mode !== "drawing-obstacle" || !currentObstaclePoints.length) return;
    currentObstaclePoints.pop();
    const layer = obstacleVertexLayers.pop();
    if (layer) map.removeLayer(layer);
    redrawObstaclePreview();
    btnFinishObstacle.disabled = currentObstaclePoints.length < 3;
    btnUndoObstacle.disabled = currentObstaclePoints.length === 0;
  });

  btnFinishObstacle.addEventListener("click", () => {
    if (currentObstaclePoints.length < 3) return;
    obstacles.push(currentObstaclePoints.slice());
    const layer = L.polygon(currentObstaclePoints, { color: "#ff5c6c", weight: 2, fillOpacity: 0.25, fillColor: "#ff5c6c" }).addTo(map);
    obstacleLayers.push(layer);
    clearLayerList(obstacleVertexLayers);
    if (obstaclePreviewLine) { map.removeLayer(obstaclePreviewLine); obstaclePreviewLine = null; }
    currentObstaclePoints = [];
    mode = "idle";
    btnDrawObstacle.classList.remove("active");
    btnFinishObstacle.disabled = true;
    btnUndoObstacle.disabled = true;
    btnClearObstacles.disabled = false;
    updateObstacleStatus();
    obstacleHint.textContent = "Obstacle marked. You can mark another, or clear all.";
    updateGridPreview();
  });

  btnClearObstacles.addEventListener("click", () => {
    resetObstacles();
    updateGridPreview();
  });

  function redrawPreview(points, color) {
    if (previewLine) { map.removeLayer(previewLine); previewLine = null; }
    if (points.length < 2) return;
    previewLine = L.polygon(points, { color, weight: 2, fillOpacity: 0.05, dashArray: "4,4" }).addTo(map);
  }

  function clearLayerList(list) {
    for (const l of list) map.removeLayer(l);
    list.length = 0;
  }

  // ---------------- Street preview (server-side) ----------------
  async function updateGridPreview() {
    if (!areaFinalized) { roadStatus.textContent = "Draw an area to see the street network."; return; }
    if (roadFetchInFlight) { roadStatus.textContent = "Waiting on road data…"; return; }
    if (!roadWays || !roadWays.length) return;
    const payload = { area: areaPoints, roadWays: roadWays || [], obstacles: obstacles };
    try {
      const r = await api("/api/preview", payload);
      roadStatus.textContent = `Street network: ${r.numStreets} segments, ${r.streetLengthM.toFixed(0)} m total — will solve exactly (Route Inspection).`;
    } catch (e) {
      roadStatus.textContent = "Couldn't reach the Python backend: " + e.message;
    }
  }

  // ---------------- Plan button state ----------------
  function updatePlanButton() {
    btnPlan.disabled = !areaFinalized;
    btnSetStart.disabled = !areaFinalized;
    btnDrawObstacle.disabled = !areaFinalized;
    if (areaFinalized) planStatus.textContent = "Ready to plan.";
  }

  // ---------------- Reset ----------------
  btnReset.addEventListener("click", () => location.reload());

  function resetPlanResultsOnly() {
    if (baseLineLayer) { map.removeLayer(baseLineLayer); baseLineLayer = null; }
    clearFleetLayers();
    stopAnim();
    resultsCard.style.display = "none";
    baselineResult = null;
    rlResult = null;
    fleetResult = null;
  }

  function clearFleetLayers() {
    for (const l of fleetLineLayers) map.removeLayer(l);
    fleetLineLayers = [];
  }

  // ---------------- Planning (server-side) ----------------
  btnPlan.addEventListener("click", runPlanning);

  async function runPlanning() {
    resetPlanResultsOnly();
    btnPlan.disabled = true;
    progressBar.style.width = "10%";

    if (roadFetchInFlight) {
      planStatus.textContent = "Still fetching road data — try again in a moment.";
      btnPlan.disabled = false;
      progressBar.style.width = "0%";
      return;
    }

    if (!(roadWays && roadWays.length)) {
      planStatus.textContent = "No road data available in this area — try a different location.";
      btnPlan.disabled = false;
      progressBar.style.width = "0%";
      return;
    }

    const k = parseInt(numSweepersSlider.value, 10) || 1;
    const fleetNote = k > 1 ? ` Splitting across ${k} sweepers with multi-agent RL…` : "";
    planStatus.textContent = "Sending to the Python backend — solving the street network exactly (Route Inspection)…" + fleetNote;
    progressBar.style.width = "35%";

    const payload = { area: areaPoints, mode: "street", roadWays: roadWays || [], numSweepers: k, bothCurbs: bothCurbsToggle.checked, start: startPoint || null, obstacles: obstacles };

    let result;
    try {
      result = await api("/api/plan", payload);
    } catch (e) {
      planStatus.textContent = "Planning failed: " + e.message;
      progressBar.style.width = "0%";
      btnPlan.disabled = false;
      return;
    }

    progressBar.style.width = "100%";
    baselineResult = result.baseline;
    rlResult = result.rl;
    fleetResult = result.fleet || { numVehicles: 1, vehicles: [{ id: 1, color: "#23c98f", path: rlResult.path, lengthM: rlResult.lengthM, steps: rlResult.steps }], makespanM: rlResult.lengthM, totalLengthM: rlResult.lengthM };
    const fleetDoneNote = fleetResult.numVehicles > 1
      ? ` Split across ${fleetResult.numVehicles} sweepers via multi-agent RL — makespan ${Math.round(fleetResult.makespanM)} m.`
      : "";
    planStatus.textContent = "Done — solved the street network exactly (Chinese Postman / Route Inspection)." + fleetDoneNote;
    renderResults(result);
    btnPlan.disabled = false;
  }

  // ---------------- Results rendering ----------------
  function renderResults(result) {
    resultsCard.style.display = "block";

    const labels = result.labels || { baseline: "Baseline", improved: "Improved path" };
    lastLabels = labels;
    $("baseLabel").textContent = labels.baseline;

    baseLineLayer = L.polyline(baselineResult.path, { color: "#ff9f43", weight: 3, opacity: 0.9 }).addTo(map);

    clearFleetLayers();
    for (const v of fleetResult.vehicles) {
      fleetLineLayers.push(L.polyline(v.path, { color: v.color, weight: 3, opacity: 0.9, dashArray: "1,6" }).addTo(map));
    }
    renderFleetLegend(labels);
    renderAnimOptions(labels);

    map.fitBounds(L.polygon(areaPoints).getBounds(), { padding: [30, 30] });
    applyToggles();

    const streets = result.streets || {};
    const overheadPct = rlResult.overheadPct ?? 0;
    const baseOverheadPct = Math.max(0, (baselineResult.lengthM / streets.streetLengthM - 1) * 100);
    const noteRow = result.bothCurbs
      ? `<tr><td>Why no "optimized" number here</td><td colspan="2">Once every street must be driven twice, any route covering both curbs has the same total length (2× the street network) — there's nothing left to optimize for one truck. The fleet split below still is optimized (RL balances the workload across trucks).</td></tr>`
      : "";
    metricsTable.innerHTML = `
      <tr><th>Metric</th><th style="color:var(--base-color)">${labels.baseline}</th><th style="color:var(--rl-color)">${labels.improved}</th></tr>
      <tr><td>Streets covered</td><td colspan="2">${streets.numStreets} segments, ${streets.streetLengthM.toFixed(0)} m of real street</td></tr>
      <tr><td>Route length</td><td>${baselineResult.lengthM.toFixed(0)} m</td><td>${rlResult.lengthM.toFixed(0)} m</td></tr>
      <tr><td>Overhead vs. real street length</td><td>${baseOverheadPct.toFixed(0)}%</td><td>${overheadPct.toFixed(0)}%</td></tr>
      <tr><td>Compute time (Python)</td><td colspan="2">${rlResult.timeMs.toFixed(0)} ms (solved once for both)</td></tr>
      ${noteRow}
    `;

    metricsTable.innerHTML += `<tbody id="fleetRowsBody">${fleetRowsHtml()}</tbody>`;
    animBadge.textContent = "0 / 0 m";
  }

  // HTML for the fleet-specific metrics rows (split, makespan, train time).
  // Pulled out of renderResults so the mid-route replan handler can refresh
  // just these rows after a rebalance, without re-deriving the whole table.
  function fleetRowsHtml() {
    if (!fleetResult || fleetResult.numVehicles <= 1) return "";
    const perVehicle = fleetResult.vehicles.map((v) => `${Math.round(v.lengthM)} m`).join(", ");
    return `
      <tr><td>Fleet split (${fleetResult.numVehicles} sweepers, RL task allocation)</td><td colspan="2">${perVehicle}</td></tr>
      <tr><td>Makespan (longest shift)</td><td colspan="2"><b>${Math.round(fleetResult.makespanM)} m</b> — the real bottleneck for how long the job takes with ${fleetResult.numVehicles} machines working in parallel</td></tr>
      <tr><td>Fleet RL train time</td><td colspan="2">${Math.round(fleetResult.trainMs)} ms</td></tr>
    `;
  }

  function renderFleetMetricsRows() {
    const tbody = $("fleetRowsBody");
    if (tbody) tbody.innerHTML = fleetRowsHtml();
  }

  // Per-vehicle legend: one color-swatched toggle row per sweeper. At
  // numVehicles === 1 this renders identically to the old single-row
  // legend (same green as before).
  function renderFleetLegend(labels) {
    const legend = $("fleetLegend");
    if (fleetResult.numVehicles <= 1) {
      const v = fleetResult.vehicles[0];
      legend.innerHTML = `<div class="toggle-row"><span><span class="swatch" style="background:${v.color};"></span><span>${labels.improved}</span></span><input type="checkbox" class="fleetToggle" data-vid="0" checked/></div>`;
    } else {
      legend.innerHTML = `<div class="hint" style="margin:0 0 4px;">${labels.improved} — ${fleetResult.numVehicles} sweepers (RL task allocation)</div>` +
        fleetResult.vehicles.map((v, i) =>
          `<div class="toggle-row"><span><span class="swatch" style="background:${v.color};"></span><span>Sweeper ${v.id} — ${Math.round(v.lengthM)} m</span></span><input type="checkbox" class="fleetToggle" data-vid="${i}" checked/></div>`
        ).join("");
    }
    for (const cb of legend.querySelectorAll(".fleetToggle")) {
      cb.addEventListener("change", applyToggles);
    }
  }

  function renderAnimOptions(labels) {
    let html = `<option value="base">Animate: ${labels.baseline}</option>`;
    if (fleetResult.numVehicles > 1) {
      html += `<option value="all" selected>Animate: all ${fleetResult.numVehicles} sweepers at once</option>`;
      html += fleetResult.vehicles.map((v, i) => `<option value="v${i}">Animate: Sweeper ${v.id} only</option>`).join("");
    } else {
      html += `<option value="v0" selected>Animate: ${labels.improved}</option>`;
    }
    animSelect.innerHTML = html;
  }

  toggleBase.addEventListener("change", applyToggles);
  function applyToggles() {
    if (baseLineLayer) {
      if (toggleBase.checked) { if (!map.hasLayer(baseLineLayer)) baseLineLayer.addTo(map); } else map.removeLayer(baseLineLayer);
    }
    document.querySelectorAll(".fleetToggle").forEach((cb) => {
      const vid = parseInt(cb.getAttribute("data-vid"), 10);
      const layer = fleetLineLayers[vid];
      if (!layer) return;
      if (cb.checked) { if (!map.hasLayer(layer)) layer.addTo(map); } else map.removeLayer(layer);
    });
  }

  // ---------------- Animation ----------------
  // A full playback at the slowest setting (1×) always takes this many
  // seconds, regardless of how long the actual route is — so "1×" reads as
  // a proper slow crawl on both a short traced street and a long fetched
  // network, instead of racing through long routes.
  const SLOWEST_PLAYBACK_SECONDS = 60;
  const ANIM_TICK_MS = 40;

  speedSlider.addEventListener("input", () => { speedVal.textContent = speedSlider.value + "×"; });

  function computeCumulativeDistances(latlngs) {
    const cum = [0];
    for (let i = 1; i < latlngs.length; i++) {
      cum.push(cum[i - 1] + L.latLng(latlngs[i - 1]).distanceTo(latlngs[i]));
    }
    return cum;
  }

  // Interpolated position `dist` metres along the path. `segRef` is a
  // mutable {i} pointer that only moves forward, so repeated calls with
  // increasing `dist` (the normal playback direction) stay O(1) amortized
  // instead of re-scanning the whole path every tick.
  function pointAtDistance(latlngs, cumDist, dist, segRef) {
    const total = cumDist[cumDist.length - 1];
    dist = Math.max(0, Math.min(dist, total));
    while (segRef.i < cumDist.length - 2 && cumDist[segRef.i + 1] < dist) segRef.i++;
    while (segRef.i > 0 && cumDist[segRef.i] > dist) segRef.i--;
    const i = segRef.i;
    const segStart = cumDist[i], segEnd = cumDist[i + 1];
    const segLen = segEnd - segStart;
    const t = segLen > 1e-9 ? (dist - segStart) / segLen : 0;
    const a = latlngs[i], b = latlngs[i + 1];
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
  }

  function makeSweeperIcon(color) {
    return L.divIcon({
      className: "",
      html: `<div class="sweeper-toy" style="--sweeper-color:${color};">
               <div class="shadow"></div>
               <div class="heading">
                 <div class="wheel tl"></div><div class="wheel tr"></div>
                 <div class="wheel bl"></div><div class="wheel br"></div>
                 <div class="light l"></div><div class="light r"></div>
                 <div class="body"></div>
               </div>
             </div>`,
      iconSize: [44, 54],
      iconAnchor: [22, 27],
    });
  }

  // Rotates the little heading arrow to face the direction of travel,
  // using screen-space pixel deltas so it looks right regardless of map
  // zoom/rotation.
  function updateSweeperHeading(marker, fromLatLng, toLatLng) {
    if (!marker || !fromLatLng || !toLatLng) return;
    const p1 = map.latLngToContainerPoint(fromLatLng);
    const p2 = map.latLngToContainerPoint(toLatLng);
    const dx = p2.x - p1.x, dy = p2.y - p1.y;
    if (Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) return; // no movement, keep last heading
    const angleDeg = (Math.atan2(dx, -dy) * 180) / Math.PI; // 0deg = up (nose points up by default)
    const el = marker.getElement();
    if (!el) return;
    const headingEl = el.querySelector(".heading");
    if (headingEl) headingEl.style.transform = `rotate(${angleDeg}deg)`;
  }

  // Which path(s) to animate, given the current dropdown selection: the
  // baseline run, a single sweeper, or the whole fleet moving at once
  // (each with its own toy-sweeper marker, all ticking together).
  function animSelection() {
    const which = animSelect.value;
    if (which === "base") return [{ path: baselineResult.path, color: "#ff9f43" }];
    if (which === "all") return fleetResult.vehicles.map((v) => ({ path: v.path, color: v.color }));
    const m = /^v(\d+)$/.exec(which);
    if (m && fleetResult.vehicles[+m[1]]) {
      const v = fleetResult.vehicles[+m[1]];
      return [{ path: v.path, color: v.color }];
    }
    return [{ path: fleetResult.vehicles[0].path, color: fleetResult.vehicles[0].color }];
  }

  btnPlay.addEventListener("click", () => {
    if (!baselineResult || !fleetResult) return;
    if (!animTimer) {
      if (!animEntries.length) {
        for (const { path, color } of animSelection()) {
          if (!path || path.length < 2) continue;
          const cumDist = computeCumulativeDistances(path);
          const marker = L.marker(path[0], { icon: makeSweeperIcon(color), interactive: false }).addTo(map);
          updateSweeperHeading(marker, path[0], path[1]);
          animEntries.push({ pathLatLngs: path, cumDist, totalDist: cumDist[cumDist.length - 1], dist: 0, seg: { i: 0 }, marker });
        }
      }
      startAnimLoop();
    }
    btnPlay.disabled = true;
    btnPause.disabled = false;
    btnStopAnim.disabled = false;
  });

  btnPause.addEventListener("click", () => {
    clearAnimTimer();
    btnPlay.disabled = false;
    btnPause.disabled = true;
  });

  btnStopAnim.addEventListener("click", stopAnim);
  animSelect.addEventListener("change", stopAnim);

  function startAnimLoop() {
    clearAnimTimer();
    if (!animEntries.length) return;
    animTimer = setInterval(() => {
      const speedMultiplier = parseFloat(speedSlider.value);
      let allDone = true;
      let sumDist = 0, sumTotal = 0;

      for (const e of animEntries) {
        sumTotal += e.totalDist;
        if (e.totalDist <= 0) continue;
        // Constant-speed motion regardless of how densely each path's
        // points are spaced — always metres-per-second, not points-per-tick.
        const metersPerSecond = (e.totalDist / SLOWEST_PLAYBACK_SECONDS) * speedMultiplier;
        const prevLatLng = pointAtDistance(e.pathLatLngs, e.cumDist, e.dist, e.seg);
        e.dist = Math.min(e.totalDist, e.dist + metersPerSecond * (ANIM_TICK_MS / 1000));
        const curLatLng = pointAtDistance(e.pathLatLngs, e.cumDist, e.dist, e.seg);
        e.marker.setLatLng(curLatLng);
        updateSweeperHeading(e.marker, prevLatLng, curLatLng);
        sumDist += e.dist;
        if (e.dist < e.totalDist) allDone = false;
      }

      animBadge.textContent = animEntries.length > 1
        ? `${Math.round(sumDist)} / ${Math.round(sumTotal)} m combined (${animEntries.length} sweepers)`
        : `${Math.round(sumDist)} / ${Math.round(sumTotal)} m`;

      if (allDone) {
        clearAnimTimer();
        btnPlay.disabled = false;
        btnPause.disabled = true;
      }
    }, ANIM_TICK_MS);
  }

  function clearAnimTimer() {
    if (animTimer) { clearInterval(animTimer); animTimer = null; }
  }

  function stopAnim() {
    clearAnimTimer();
    for (const e of animEntries) map.removeLayer(e.marker);
    animEntries = [];
    btnPlay.disabled = false;
    btnPause.disabled = true;
    btnStopAnim.disabled = true;
    animBadge.textContent = "0 / 0 m";
  }
})();
