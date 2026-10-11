/* Kelp page: map of kelp beds, quarterly canopy chart (ours vs Kelpwatch), latest canopy overlay. */
(function () {
  const map = L.map("kmap", { scrollWheelZoom: false }).setView([34.6, -119.6], 6);
  L.tileLayer(BC.ESRI, { attribution: BC.ESRI_ATTR, maxZoom: 16 }).addTo(map);
  const panel = document.getElementById("kpanel");
  let overlay = null;

  function chart(series) {
    const s = series.filter(r => r.coverage >= 0.5);
    if (s.length < 2) return "<p class='small-note'>Not enough clear quarters yet.</p>";
    const W = 420, H = 190, L0 = 46, R = 14, T = 10, B = 24;
    const t = r => r.year + (r.quarter - 1) / 4;
    const x0 = t(s[0]), x1 = t(s[s.length - 1]);
    const ymax = Math.max(...s.map(r => Math.max(r.canopy_ha + (r.plus_minus_ha || 0), r.kelpwatch_ha || 0))) * 1.1 || 1;
    const X = v => L0 + (W - L0 - R) * (v - x0) / (x1 - x0 || 1), Y = v => T + (H - T - B) * (1 - v / ymax);
    const ours = s.map(r => `${X(t(r))},${Y(r.canopy_ha)}`).join(" ");
    const kwp = s.filter(r => r.kelpwatch_ha != null);
    const kwl = kwp.map(r => `${X(t(r))},${Y(r.kelpwatch_ha)}`).join(" ");
    const band = s.some(r => r.plus_minus_ha) ? `<polygon fill="#d6a026" opacity=".15" points="${s.map(r => `${X(t(r))},${Y(r.canopy_ha + r.plus_minus_ha)}`).join(" ")} ${s.slice().reverse().map(r => `${X(t(r))},${Y(Math.max(0, r.canopy_ha - r.plus_minus_ha))}`).join(" ")}"/>` : "";
    const years = [...new Set(s.map(r => r.year))].filter((y, i, a) => a.length < 8 || y % 2 === 0)
      .map(y => `<text x="${X(y)}" y="${H - 6}" text-anchor="middle">${y}</text>`).join("");
    const ticks = [0, ymax / 2, ymax].map(v => `<text x="${L0 - 6}" y="${Y(v) + 4}" text-anchor="end">${BC.fmt(v)}</text><line x1="${L0}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/>`).join("");
    return `<svg class="hist" viewBox="0 0 ${W} ${H}" role="img" aria-label="Kelp canopy area by quarter">${ticks}${years}${band}
      ${kwl ? `<polyline points="${kwl}" fill="none" stroke="#6f7e81" stroke-width="1.5" stroke-dasharray="4 3"/>` : ""}
      <polyline points="${ours}" fill="none" stroke="#d6a026" stroke-width="2"/></svg>
      <div class="map-legend" style="position:static;box-shadow:none;padding:4px 0;display:flex;gap:14px">
        <span><i style="background:#d6a026"></i>BlueCarbon-AI (Sentinel-2)</span><span><i style="background:#6f7e81"></i>Kelpwatch (Landsat)</span></div>`;
  }

  function show(site, cal) {
    if (overlay) map.removeLayer(overlay);
    overlay = L.imageOverlay(`data/kelp/${site.name}/canopy_latest.png`, site.bounds).addTo(map);
    map.fitBounds(site.bounds, { padding: [20, 20] });
    const last = site.series.filter(r => r.coverage >= 0.5).slice(-1)[0];
    const peak = site.series.reduce((m, r) => r.coverage >= 0.5 && r.canopy_ha > (m ? m.canopy_ha : -1) ? r : m, null);
    const ho = (cal.held_out || {}).site_quarter_ha || {};
    panel.innerHTML = `<span class="badge${site.held_out ? "" : " train"}">${site.held_out ? "Left out of calibration" : "Calibration site"}</span>
      <h1>${BC.esc(site.title)}</h1><p class="where">${BC.esc(site.region)}, ${BC.esc(site.species)}</p>
      <div class="kpis">
        <div><b>${last ? BC.fmt(last.canopy_ha) : "–"}</b><small>hectares of canopy, ${last ? `${last.year} Q${last.quarter}` : ""}</small></div>
        <div><b>${peak ? BC.fmt(peak.canopy_ha) : "–"}</b><small>peak quarter since ${site.series[0].year}${peak ? ` (${peak.year} Q${peak.quarter})` : ""}</small></div>
      </div>
      <h2>Canopy every quarter (ha)</h2>${chart(site.series)}
      <p class="small-note">On kelp beds left out of calibration, quarterly canopy agrees with Kelpwatch with r² ${ho.r2 ?? "–"}
        (error ±${ho.rmse ?? "–"} ha, bias ${ho.bias ?? "–"} ha). Quarters with under half the area clear of cloud are not shown.</p>`;
  }

  fetch("data/kelp.json").then(r => r.ok ? r.json() : null).then(doc => {
    if (!doc || !doc.sites || !doc.sites.length) {
      panel.innerHTML = `<span class="badge">In calibration</span><h1>First results soon</h1>
        <p class="desc">Seven California kelp beds, from Point Loma to Mendocino, are being mapped every quarter since 2016
        and calibrated against Kelpwatch. Results, accuracy and reports appear here when the first run finishes.</p>`;
      return;
    }
    doc.sites.forEach(s => {
      L.circleMarker([(s.bounds[0][0] + s.bounds[1][0]) / 2, (s.bounds[0][1] + s.bounds[1][1]) / 2],
        { radius: 7, weight: 2, color: "#fff", fillColor: s.held_out ? "#6fd3bf" : "#d6a026", fillOpacity: 1 })
        .bindTooltip(s.title).on("click", () => show(s, doc.calibration)).addTo(map);
    });
    show(doc.sites[0], doc.calibration);
  });
})();
