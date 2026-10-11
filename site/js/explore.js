/* Site map: list, Leaflet map with image overlays, details panel. Deep link: explore.html#<site id>. */
(function () {
  const map = L.map("map", { zoomControl: true, attributionControl: true }).setView([20, 0], 2);
  map.zoomControl.setPosition("topright");
  L.tileLayer(BC.ESRI, { attribution: BC.ESRI_ATTR, maxZoom: 17 }).addTo(map);

  const SINGLE = [["Habitats", "classes"], ["Blue carbon only", "bluecarbon"], ["Satellite", null], ["Infrared", "falsecolor"]];
  const CHANGE = [["After", "classes_t1"], ["Before", "classes_t0"], ["What changed", "change"], ["Satellite", null], ["Infrared", "falsecolor_t1"]];
  let doc, C, site, view = 0, base = null, over = null, outline = null;
  const opacity = document.getElementById("opacity");

  const layersFor = s => s.kind === "change" ? CHANGE
    : SINGLE.concat(s.images.confidence ? [["Confidence", "confidence"]] : []);

  function drawLayers() {
    const views = layersFor(site), [, key] = views[view];
    const bounds = site.bounds;
    if (base) map.removeLayer(base);
    if (over) map.removeLayer(over);
    over = null;
    const isIR = key && key.startsWith("falsecolor");
    const baseKey = isIR ? key : site.kind === "change" ? (key === "classes_t0" ? "rgb_t0" : "rgb_t1") : "rgb";
    base = L.imageOverlay(site.images[baseKey], bounds).addTo(map);
    if (key && !isIR && site.images[key]) over = L.imageOverlay(site.images[key], bounds, { opacity: +opacity.value }).addTo(map);
    document.querySelectorAll("#layers button").forEach((b, i) => b.setAttribute("aria-pressed", i === view));
    legend(key);
  }

  function legend(key) {
    const el = document.getElementById("legend");
    if (key === "confidence") {
      el.innerHTML = '<span><i style="background:#f2a65a"></i>Model unsure</span><span><i style="background:#f24c5a"></i>Least sure</span><span>Clear = confident</span>';
    } else if (key === "change") {
      el.innerHTML = '<span><i style="background:#22c55e"></i>Blue carbon gained</span><span><i style="background:#ef4444"></i>Blue carbon lost</span>';
    } else if (key && !key.startsWith("falsecolor")) {
      const blueOnly = key === "bluecarbon";
      el.innerHTML = doc.classes.filter(c => (site.areas_ha[c.key] || 0) >= 1 && (!blueOnly || c.blue_carbon))
        .map(c => `<span><i style="background:${c.color}"></i>${c.name}</span>`).join("");
    } else {
      el.innerHTML = key ? "<span>Infrared view: healthy plants show red</span>" : "<span>Sentinel-2, natural colour</span>";
    }
    el.style.display = el.innerHTML ? "grid" : "none";
  }

  // Blue carbon area per epoch (Landsat history): one line per habitat, a band for the 95% interval.
  function historyChart(h) {
    const ep = h.epochs.filter(e => e.ha);
    if (ep.length < 2) return "";
    const keys = BC.BLUE.filter(k => Math.max(...ep.map(e => e.ha[k] || 0)) >= 1);
    if (!keys.length) return "";
    const W = 340, H = 170, L = 46, R = 18, T = 10, B = 24;
    const x0 = ep[0].year, x1 = ep[ep.length - 1].year;
    const ymax = Math.max(...ep.flatMap(e => keys.map(k => (e.ha[k] || 0) + (e.ci[k] || 0)))) * 1.08 || 1;
    const X = y => L + (W - L - R) * (y - x0) / (x1 - x0), Y = v => T + (H - T - B) * (1 - v / ymax);
    const ticks = [0, ymax / 2, ymax].map(v => `<text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${BC.fmt(v)}</text>
      <line x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}" class="grid"/>`).join("");
    const years = ep.map(e => `<text x="${X(e.year)}" y="${H - 6}" text-anchor="middle">${e.year}</text>`).join("");
    const lines = keys.map(k => {
      const c = C[k].color;
      const band = ep.map(e => `${X(e.year)},${Y((e.ha[k] || 0) + (e.ci[k] || 0))}`).join(" ") + " " +
        ep.slice().reverse().map(e => `${X(e.year)},${Y(Math.max(0, (e.ha[k] || 0) - (e.ci[k] || 0)))}`).join(" ");
      const pts = ep.map(e => `${X(e.year)},${Y(e.ha[k] || 0)}`).join(" ");
      return `<polygon points="${band}" fill="${c}" opacity=".15"/><polyline points="${pts}" fill="none" stroke="${c}" stroke-width="2"/>` +
        ep.map(e => `<circle cx="${X(e.year)}" cy="${Y(e.ha[k] || 0)}" r="2.6" fill="${c}"><title>${C[k].name} ${e.year}: ${BC.fmt(e.ha[k] || 0)} ha</title></circle>`).join("");
    }).join("");
    const iou = (h.model && h.model.iou) || {};
    return `<h2>Habitat since ${x0}</h2>
      <svg class="hist" viewBox="0 0 ${W} ${H}" role="img" aria-label="Blue carbon habitat area by year">${ticks}${years}${lines}</svg>
      <div class="map-legend" style="position:static;box-shadow:none;padding:4px 0;display:flex;gap:14px">${keys.map(k =>
        `<span><i style="background:${C[k].color}"></i>${C[k].name}</span>`).join("")}</div>
      <p class="small-note">Hectares from Landsat (${h.scale_m || 30} m, 3-year composites), corrected for the history model's errors; shaded = 95% range.
      History model on unseen estuaries: ${keys.map(k => `${C[k].name.toLowerCase()} IoU ${iou[k] != null ? iou[k].toFixed(2) : "–"}`).join(", ")}. Coarser than the main map: use it for trends.</p>`;
  }

  function panel() {
    const s = site, c = s.carbon, a = s.areas_ha;
    const badge = s.kind === "change" ? '<span class="badge">Change over time</span>'
      : s.held_out ? '<span class="badge">Never seen in training</span>' : '<span class="badge train">Training site</span>';
    const rows = doc.classes.filter(k => (a[k.key] || 0) >= 0.5).sort((x, y) => (y.blue_carbon - x.blue_carbon) || (a[y.key] - a[x.key]))
      .map(k => `<tr class="${k.blue_carbon ? "bc" : ""}"><td><i style="background:${k.color}"></i>${k.name}</td><td class="num">${BC.fmt(a[k.key])} ha</td></tr>`).join("");
    const soil = BC.BLUE.map(k => c.classes[k]).filter(Boolean).filter(v => v.area_ha >= 1);
    const measured = soil.filter(v => v.soil.source === "measured");
    const soilNote = measured.length
      ? `Soil carbon from ${Math.max(...measured.map(v => v.soil.cores))} measured cores within ${Math.max(...measured.map(v => v.soil.radius_km))} km (Smithsonian Coastal Carbon Library)${measured.length < soil.length ? "; IPCC values for the rest" : ""}.`
      : "Soil carbon from IPCC global default values; no measured cores nearby.";
    const mg = c.classes.mangrove && c.classes.mangrove.biomass;
    const bioNote = mg && mg.source === "measured"
      ? ` Mangrove biomass from NASA's canopy-height map (${BC.fmt(mg.agb_mg_ha)} t/ha above ground, ±30%).`
      : " Biomass from IPCC default values.";
    let change = "";
    if (s.change) {
      const ch = s.change;
      const cr = BC.BLUE.filter(k => Math.max(ch.areas_t0[k] || 0, ch.areas_t1[k] || 0) >= 0.5).map(k => {
        const d = (ch.areas_t1[k] || 0) - (ch.areas_t0[k] || 0);
        return `<tr><td><i style="background:${C[k].color}"></i>${C[k].name}</td><td class="num">${BC.fmt(ch.areas_t0[k])}</td><td class="num">${BC.fmt(ch.areas_t1[k])}</td>
          <td class="num ${d > 0 ? "up" : d < 0 ? "down" : ""}">${d > 0 ? "+" : ""}${BC.fmt(d)}</td></tr>`;
      }).join("");
      const net = ch.stock_t1 - ch.stock_t0;
      change = `<h2>What changed, ${BC.esc(ch.t0)} to ${BC.esc(ch.t1)}</h2>
        <table class="hab-table"><thead><tr><th>Habitat (ha)</th><th class="num">${BC.esc(ch.t0)}</th><th class="num">${BC.esc(ch.t1)}</th><th class="num">Change</th></tr></thead><tbody>${cr}</tbody></table>
        <p class="small-note">Net change in stored carbon: ${net >= 0 ? "+" : "−"}${BC.fmt(Math.abs(net))} t CO₂. Small changes can come from tide, season or image quality rather than real habitat change.</p>`;
    }
    const sl = s.sea_level;
    const slr = sl && sl.ratio !== null && sl.ratio !== undefined ? `<h2>Sea-level rise</h2>
        <p class="desc" style="margin-bottom:6px">Within ${sl.search_km} km behind the wetland there are <b>${BC.fmt(sl.room_ha)} ha</b> of open, low land
        (0 to ${sl.max_elev_m} m above sea level, not built up) it could move into as the sea rises: ${Math.round(sl.ratio * 100)}% of its own area.</p>
        <p class="small-note">Rating: <b>${{low: "low risk", medium: "medium risk", high: "high risk (coastal squeeze)"}[sl.rating]}</b>.
        Sea level is projected to rise ${sl.ar6_2100_m[0]} to ${sl.ar6_2100_m[1]} m by 2100 (IPCC AR6).</p>` : "";
    const reports = s.reports ? `<h2>Report</h2><div class="dl">
        <a class="btn btn-solid btn-sm" href="${s.reports.en}" download>PDF (English)</a>
        <a class="btn btn-line btn-sm" href="${s.reports.es}" download>PDF (español)</a></div>
        <p class="small-note">Map, habitat areas, carbon and credit potential. Rights and status checks are part of a full site screen.</p>` : "";
    document.getElementById("panel").innerHTML = `${badge}
      <h1>${BC.esc(s.title)}</h1>
      <p class="where">${BC.esc(s.region)}, ${BC.esc(s.period)}</p>
      <p class="coords">${BC.coords(s.center)}</p>
      ${s.description ? `<p class="desc">${BC.esc(s.description.charAt(0).toUpperCase() + s.description.slice(1))}</p>` : ""}
      <div class="kpis">
        <div><b>${BC.fmt(BC.blueTotal(s))}</b><small>hectares of blue carbon habitat</small></div>
        <div><b>${BC.big(c.stock_tco2e.mean)}</b><small>tonnes of CO₂ stored (${BC.big(c.stock_tco2e.p05)} to ${BC.big(c.stock_tco2e.p95)})</small></div>
        <div><b>${BC.fmt(c.seq_tco2e_yr.mean)}</b><small>tonnes of CO₂ absorbed per year</small></div>
        <div><b>$${BC.big(c.value_usd_yr.low || 0)}–${BC.big(c.value_usd_yr.high || 0)}</b><small>indicative credit value per year, before deductions</small></div>
      </div>
      <h2>Habitats found</h2>
      <table class="hab-table"><tbody>${rows}</tbody></table>
      <p class="small-note">Areas corrected for the model's known errors. ${soilNote}${bioNote}</p>
      ${s.confidence ? `<p class="small-note">Model confidence: ${Object.entries(s.confidence).map(([k, v]) =>
        `${C[k].name.toLowerCase()} ${Math.round(v.mean_pct)}% average, ${Math.round(v.low_share * 100)}% of its area under 60%`).join("; ")}.</p>` : ""}
      ${s.history ? historyChart(s.history) : ""}
      ${slr}
      ${change}
      ${reports}`;
  }

  function select(id, fly = true) {
    site = doc.sites.find(s => s.id === id) || doc.sites[0];
    view = 0;
    document.getElementById("layers").innerHTML = layersFor(site).map(([n], i) => `<button data-i="${i}">${n}</button>`).join("");
    document.querySelectorAll("#layers button").forEach(b => b.addEventListener("click", () => { view = +b.dataset.i; drawLayers(); }));
    if (outline) map.removeLayer(outline);
    outline = L.rectangle(site.bounds, { color: "#fff", weight: 1.5, fill: false, dashArray: "4 4" }).addTo(map);
    drawLayers();
    panel();
    if (fly) map.fitBounds(site.bounds, { padding: [20, 20] });
    document.querySelectorAll("#list button").forEach(b => {
      b.setAttribute("aria-current", b.dataset.id === site.id);
      if (b.dataset.id === site.id) b.scrollIntoView({ block: "nearest" });
    });
    if (location.hash.slice(1) !== site.id) history.replaceState(null, "", "#" + site.id);
    document.title = `${site.title} | BlueCarbon-AI`;
  }

  function list(q = "") {
    const t = q.trim().toLowerCase();
    const items = doc.sites.filter(s => !t || `${s.title} ${s.region}`.toLowerCase().includes(t))
      .sort((a, b) => ((b.kind === "change") - (a.kind === "change")) || (b.held_out - a.held_out) || a.title.localeCompare(b.title));
    document.getElementById("list").innerHTML = items.map(s => `<li><button data-id="${s.id}" aria-current="${site && s.id === site.id}">
      <b>${BC.esc(s.title)}</b><small>${BC.esc(s.region)}</small>${s.kind === "change" ? "<em>change</em>" : s.held_out ? "<em>unseen</em>" : ""}</button></li>`).join("")
      || '<li style="padding:16px;color:var(--muted)">No sites match. Try a country name.</li>';
    document.querySelectorAll("#list button").forEach(b => b.addEventListener("click", () => select(b.dataset.id)));
  }

  opacity.addEventListener("input", () => over && over.setOpacity(+opacity.value));
  document.getElementById("q").addEventListener("input", e => list(e.target.value));
  window.addEventListener("hashchange", () => site && location.hash.slice(1) !== site.id && select(location.hash.slice(1)));

  BC.load().then(d => {
    doc = d; C = BC.classMap(d);
    list();
    const first = location.hash.slice(1) || (doc.sites.find(s => s.id === "laguna_terminos_mx") || doc.sites[0]).id;
    select(first);
  }).catch(err => {
    document.getElementById("panel").innerHTML = `<p>Site data could not be loaded (${BC.esc(err.message)}). Reload the page to try again.</p>`;
  });
})();
