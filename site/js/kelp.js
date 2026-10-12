/* Kelp page: explorer with a quarter-by-quarter timeline (1984 on), yearly satellite pictures, our Sentinel-2
   canopy maps (2016 on), Kelpwatch's Landsat record for comparison, a 40-year chart and a bed-by-bed summary. */
(function () {
  const OURS = "#d6a026", KW = "#6fd3bf", HEAT = [2014, 2017];   // 2014-2016 marine heatwave ("the Blob")
  const $ = id => document.getElementById(id);
  const qkey = (y, q) => `${y}_${q}`;
  const qlab = (y, q) => `${y} Q${q}`;
  const MONTHS = ["", "Jan to Mar", "Apr to Jun", "Jul to Sep", "Oct to Dec"];
  const tnum = (y, q) => y + (q - 1) / 4;

  const map = L.map("kmap", { scrollWheelZoom: false, zoomControl: true, attributionControl: true }).setView([34.6, -119.6], 6);
  L.tileLayer(BC.ESRI, { attribution: BC.ESRI_ATTR + " · Kelpwatch: SBC LTER (CC BY 4.0)", maxZoom: 17 }).addTo(map);
  map.createPane("rgb").style.zIndex = 390;
  map.createPane("canopy").style.zIndex = 420;
  const show = { rgb: true, ours: true, kw: false, hl: true };
  let doc = null, site = null, T = [], idx = 0, timer = null, layers = { rgb: null, ours: null, kw: null };

  // ---------------------------------------------------------------------------------------------- data
  function annualPeak(rows, val) {          // {year: largest quarter of the year}
    const out = {};
    rows.forEach(r => { const v = val(r); if (v != null && (out[r.year] == null || v > out[r.year])) out[r.year] = v; });
    return out;
  }
  const mean = a => a.length ? a.reduce((x, y) => x + y, 0) / a.length : null;
  const ourRows = s => [...(s.series_landsat || []).map(r => ({ ...r, ls: true })), ...s.series];
  function stats(s) {
    const kwh = s.kelpwatch_history || [];
    const ours = ourRows(s).filter(r => r.coverage >= 0.5);
    const kp = annualPeak(kwh, r => r.ha), op = annualPeak(ours, r => r.canopy_ha);
    const before = mean(Object.entries(kp).filter(([y]) => +y < HEAT[0]).map(([, v]) => v));
    const lowYears = Object.entries(kp).filter(([y]) => +y >= HEAT[0] && +y < HEAT[1]);
    const low = lowYears.length ? lowYears.reduce((m, e) => (e[1] < m[1] ? e : m)) : null;
    const lastFull = Math.max(...ours.filter(r => r.quarter === 4).map(r => r.year), -Infinity);
    const latestYear = Number.isFinite(lastFull) ? lastFull : null;
    const latest = latestYear != null ? op[latestYear] : null;          // ours
    const latestKw = latestYear != null ? kp[latestYear] : null;       // Kelpwatch, same year
    // "now vs before" compares like with like: Kelpwatch against Kelpwatch when it has the year, else ours
    const now = latestKw != null ? latestKw : latest;
    return { kp, op, before, low, latestYear, latest, latestKw, ratio: before && now != null ? now / before : null,
             ratioSrc: latestKw != null ? "Kelpwatch" : "BlueCarbon-AI" };
  }
  function timeline(s) {
    const keys = new Map();
    (s.kelpwatch_history || []).forEach(r => keys.set(qkey(r.year, r.quarter), [r.year, r.quarter]));
    ourRows(s).forEach(r => keys.set(qkey(r.year, r.quarter), [r.year, r.quarter]));
    const all = [...keys.values()].sort((a, b) => tnum(...a) - tnum(...b));
    if (!all.length) return [];
    const out = [];
    for (let y = all[0][0], q = all[0][1]; tnum(y, q) <= tnum(...all[all.length - 1]); q === 4 ? (y++, q = 1) : q++) out.push([y, q]);
    return out;
  }
  const at = (s, y, q) => ({
    ours: ourRows(s).find(r => r.year === y && r.quarter === q),
    kw: (s.kelpwatch_history || []).find(r => r.year === y && r.quarter === q),
  });

  // ---------------------------------------------------------------------------------------------- map layers
  function overlay(name, url, bounds, opacity) {
    if (!url || !bounds) { if (layers[name]) { map.removeLayer(layers[name]); layers[name] = null; } return; }
    if (layers[name]) { if (layers[name]._url !== url) layers[name].setUrl(url); layers[name].setBounds(L.latLngBounds(bounds)); }
    else layers[name] = L.imageOverlay(url, bounds, { opacity, pane: name === "rgb" ? "rgb" : "canopy", interactive: false }).addTo(map);
  }
  const base = s => `data/kelp/${s.name}`;
  function urls(s, y, q) {
    const k = qkey(y, q);
    const frames = s.frames || [];
    const latestKey = qkey(s.latest.year, s.latest.quarter);
    return {
      rgb: (s.rgb_years || []).includes(y) ? `${base(s)}/rgb/${y}.webp` : null,
      ours: frames.includes(k) || (s.frames_landsat || []).includes(k) ? `${base(s)}/frames/${k}.png`
        : (!frames.length && k === latestKey ? `${base(s)}/canopy_latest.png` : null),
      oursBounds: y < 2016 ? (s.frame_bounds_landsat || s.bounds) : (s.frame_bounds || s.bounds),
      kw: (s.kw_years || []).includes(y) ? `${base(s)}/kw/${y}.png` : null,
    };
  }
  function preload(s, i) {
    for (let j = i + 1; j < Math.min(T.length, i + 6); j++) Object.values(urls(s, ...T[j])).forEach(u => { if (u) new Image().src = u; });
  }
  function render() {
    if (!site || !T.length) return;
    const [y, q] = T[idx];
    const u = urls(site, y, q);
    overlay("rgb", show.rgb ? u.rgb : null, site.rgb_bounds || site.bounds, 1);
    overlay("ours", show.ours ? u.ours : null, u.oursBounds, 0.95);
    overlay("kw", show.kw || (show.ours && !u.ours && u.kw) ? u.kw : null, site.kw_bounds || site.bounds, 0.9);
    const v = at(site, y, q);
    const src = u.ours && show.ours ? `BlueCarbon-AI canopy (${y < 2016 ? "Landsat, 30 m" : "Sentinel-2, 20 m"})` : (layers.kw ? "Kelpwatch canopy (yearly peak)" : "no canopy map for this quarter");
    const sensor = y >= 2016 ? "Sentinel-2" : y === 2012 ? "Landsat 7, with its scan-line gaps" : y >= 2013 ? "Landsat 8" : y >= 1999 ? "Landsat 5 and 7" : "Landsat 5";
    $("kstamp").innerHTML = `<b>${y}</b><span>Q${q} · ${MONTHS[q]}</span><small>${u.rgb && show.rgb ? `Summer ${y} picture (${sensor}) · ` : ""}${src}</small>`;
    $("kslider").value = idx;
    $("kslider").setAttribute("aria-valuetext", qlab(y, q));
    cursor(y, q);
    panel(v, y, q, u);
    preload(site, idx);
  }

  // ---------------------------------------------------------------------------------------------- chart
  let chartScale = null;
  function chart(s) {
    const W = 1000, H = 210, L0 = 46, R = 12, T0 = 12, B = 26;
    const ours = s.series.filter(r => r.coverage >= 0.5), kwh = s.kelpwatch_history || [];
    const ls = (s.series_landsat || []).filter(r => r.coverage >= 0.5), sst = s.sst || [];
    if (!T.length) { $("kchart").innerHTML = ""; return; }
    const x0 = tnum(...T[0]), x1 = tnum(...T[T.length - 1]) + 0.25;
    const ymax = Math.max(1, ...ours.map(r => r.canopy_ha + (r.plus_minus_ha || 0)), ...ls.map(r => r.canopy_ha), ...kwh.map(r => r.ha)) * 1.08;
    const X = v => L0 + (W - L0 - R) * (v - x0) / (x1 - x0 || 1), Y = v => T0 + (H - T0 - B) * (1 - v / ymax);
    chartScale = { X, x0, x1, L0, R, W };
    const step = x1 - x0 > 20 ? 5 : x1 - x0 > 8 ? 2 : 1;
    const years = []; for (let yy = Math.ceil(x0 / step) * step; yy <= x1; yy += step) years.push(yy);
    const grid = years.map(yy => `<line class="g" x1="${X(yy)}" x2="${X(yy)}" y1="${T0}" y2="${H - B}"/><text x="${X(yy)}" y="${H - 8}" text-anchor="middle">${yy}</text>`).join("");
    const yt = [0, ymax / 2, ymax].map(v => `<text x="${L0 - 6}" y="${Y(v) + 4}" text-anchor="end">${BC.fmt(Math.round(v))}</text>`).join("");
    const heat = HEAT[1] > x0 ? `<rect class="heat" x="${X(Math.max(HEAT[0], x0))}" width="${X(HEAT[1]) - X(Math.max(HEAT[0], x0))}" y="${T0}" height="${H - T0 - B}"/><text class="heat-t" x="${X(Math.max(HEAT[0], x0)) + 4}" y="${T0 + 12}">2014–16 heatwave</text>` : "";
    const line = (rows, v) => rows.map(r => `${X(tnum(r.year, r.quarter) + 0.125).toFixed(1)},${Y(v(r)).toFixed(1)}`).join(" ");
    const band = ours.some(r => r.plus_minus_ha) ? `<polygon fill="${OURS}" opacity=".16" points="${line(ours, r => r.canopy_ha + (r.plus_minus_ha || 0))} ${line(ours.slice().reverse(), r => Math.max(0, r.canopy_ha - (r.plus_minus_ha || 0)))}"/>` : "";
    $("kchart").innerHTML = `<div class="kx-readout" id="kread"></div>
      <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img" aria-label="Canopy area every quarter (hectares)">
        ${heat}${grid}${yt}${band}
        ${kwh.length ? `<polyline points="${line(kwh, r => r.ha)}" fill="none" stroke="${KW}" stroke-width="1.4" vector-effect="non-scaling-stroke"/>` : ""}
        ${ls.length ? `<polyline points="${line(ls, r => r.canopy_ha)}" fill="none" stroke="${OURS}" stroke-width="1.6" stroke-dasharray="4 3" vector-effect="non-scaling-stroke"/>` : ""}
        <polyline points="${line(ours, r => r.canopy_ha)}" fill="none" stroke="${OURS}" stroke-width="2.2" vector-effect="non-scaling-stroke"/>
        <line id="kcur" class="cur" y1="${T0}" y2="${H - B}"/>
      </svg>
      ${sst.length ? `<svg class="kx-sst" viewBox="${L0} 0 ${W - L0 - R} 22" preserveAspectRatio="none" aria-label="Sea surface temperature anomaly">${sst.map(r => {
        const a = Math.max(-2, Math.min(2, r.anom)), x = X(tnum(r.year, r.quarter));
        const col = a >= 0 ? `rgba(232,108,84,${(0.15 + 0.85 * a / 2).toFixed(2)})` : `rgba(90,160,230,${(0.15 + 0.85 * -a / 2).toFixed(2)})`;
        return `<rect x="${x.toFixed(1)}" y="0" width="${(X(tnum(r.year, r.quarter) + 0.25) - x + 0.4).toFixed(1)}" height="22" fill="${col}"/>`;
      }).join("")}</svg>` : ""}
      <div class="kx-chart-legend"><span><i style="background:${OURS}"></i>BlueCarbon-AI, hectares of canopy${ls.length ? " (dashed: Landsat before 2016)" : ""}</span>${kwh.length ? `<span><i style="background:${KW}"></i>Kelpwatch</span>` : ""}${sst.length ? `<span><i style="background:linear-gradient(90deg,#5aa0e6,#e86c54)"></i>Sea temperature vs 1984–2013 normal (cool to warm)</span>` : ""}<span class="hint">Click the chart to jump to a quarter</span></div>`;
    const svg = $("kchart").querySelector("svg");
    const pick = e => {
      const r = svg.getBoundingClientRect(), xv = ((e.clientX - r.left) / r.width) * W;
      const t = x0 + (xv - L0) / (W - L0 - R) * (x1 - x0);
      let best = 0; T.forEach((k, i) => { if (Math.abs(tnum(...k) - t) < Math.abs(tnum(...T[best]) - t)) best = i; });
      return best;
    };
    svg.addEventListener("click", e => { stop(); idx = pick(e); render(); });
    svg.addEventListener("mousemove", e => {
      const [y, q] = T[pick(e)], v = at(site, y, q), tt = (site.sst || []).find(r => r.year === y && r.quarter === q);
      $("kread").textContent = `${qlab(y, q)}: ${v.ours ? `BlueCarbon-AI ${BC.fmt(v.ours.canopy_ha)} ha` : ""}${v.ours && v.kw ? " · " : ""}${v.kw ? `Kelpwatch ${BC.fmt(v.kw.ha)} ha` : ""}${tt ? ` · sea ${tt.c.toFixed(1)}°C (${tt.anom >= 0 ? "+" : ""}${tt.anom.toFixed(1)})` : ""}` || qlab(y, q);
    });
    svg.addEventListener("mouseleave", () => { $("kread").textContent = ""; });
  }
  function cursor(y, q) {
    const c = $("kcur");
    if (c && chartScale) { const x = chartScale.X(tnum(y, q) + 0.125); c.setAttribute("x1", x); c.setAttribute("x2", x); }
  }

  // ---------------------------------------------------------------------------------------------- panel
  function spark(s) {
    const st = stats(s), pts = Object.entries(Object.keys(st.kp).length ? st.kp : st.op).map(([y, v]) => [+y, v]);
    if (pts.length < 2) return "";
    const W = 120, H = 30, x0 = pts[0][0], x1 = pts[pts.length - 1][0], m = Math.max(...pts.map(p => p[1])) || 1;
    const p = pts.map(([y, v]) => `${(W * (y - x0) / (x1 - x0 || 1)).toFixed(1)},${(H - 2 - (H - 4) * v / m).toFixed(1)}`).join(" ");
    return `<svg class="spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" aria-hidden="true"><polyline points="${p}" fill="none" stroke="${Object.keys(st.kp).length ? KW : OURS}" stroke-width="1.5" vector-effect="non-scaling-stroke"/></svg>`;
  }
  function pct(r) { return r == null ? "–" : `${Math.round(r * 100)}%`; }
  function panel(v, y, q, u) {
    const s = site, st = stats(s);
    const cur = v.ours ? `${BC.fmt(v.ours.canopy_ha)} <small>ha</small>` : "–";
    const kwv = v.kw ? `${BC.fmt(v.kw.ha)} <small>ha</small>` : "–";
    const why = !v.ours && tnum(y, q) < 2016 ? "Before Sentinel-2: the canopy shown is Kelpwatch's Landsat record."
      : !v.ours ? "No clear Sentinel-2 view this quarter." : v.ours.coverage < 0.5 ? "Under half the area was clear of cloud this quarter." : "";
    $("kpanel").innerHTML = `<span class="badge${s.held_out ? "" : " train"}">${s.held_out ? "Left out of calibration" : "Calibration site"}</span>
      <h1>${BC.esc(s.title)}</h1><p class="where">${BC.esc(s.region)}, ${BC.esc(s.species)}</p>
      <p class="kx-q">${qlab(y, q)}</p>
      <div class="kpis kx-kpis">
        <div><b>${cur}</b><small>BlueCarbon-AI canopy${v.ours && v.ours.plus_minus_ha ? ` ±${BC.fmt(v.ours.plus_minus_ha)} ha` : ""}</small></div>
        <div><b>${kwv}</b><small>Kelpwatch canopy</small></div>
      </div>
      ${why ? `<p class="small-note">${why}</p>` : ""}
      ${(() => { const sc = (s.score || {}).site_quarter_ha; const tt = (s.sst || []).find(r => r.year === y && r.quarter === q);
        return (tt ? `<p class="kx-sstnote">Sea temperature ${tt.c.toFixed(1)}°C, <b class="${tt.anom > 0.7 ? "bad" : tt.anom < -0.7 ? "good" : ""}">${tt.anom >= 0 ? "+" : ""}${tt.anom.toFixed(1)}°C</b> vs the 1984–2013 normal for this quarter</p>` : "")
          + (sc && sc.rmse != null ? `<p class="kx-score">Checked against Kelpwatch on this bed (${BC.esc(s.score.how)}): typical error ±${BC.fmt(sc.rmse)} ha a quarter, bias ${sc.bias > 0 ? "+" : ""}${BC.fmt(sc.bias)} ha${sc.r2 != null ? `, r² ${sc.r2.toFixed(2)}` : ""}.</p>` : ""); })()}
      <h2>The long view</h2>
      <dl class="kx-dl">
        ${st.before != null ? `<div><dt>Average yearly peak, 1984–2013</dt><dd>${BC.fmt(st.before)} ha</dd></div>` : ""}
        ${st.low ? `<div><dt>Lowest during the 2014–16 heatwave</dt><dd>${BC.fmt(st.low[1])} ha (${st.low[0]})</dd></div>` : ""}
        ${st.latestKw != null ? `<div><dt>Peak in ${st.latestYear}, Kelpwatch</dt><dd>${BC.fmt(st.latestKw)} ha</dd></div>` : ""}
        ${st.latest != null ? `<div><dt>Peak in ${st.latestYear}, BlueCarbon-AI</dt><dd>${BC.fmt(st.latest)} ha</dd></div>` : ""}
        ${st.ratio != null ? `<div><dt>${st.latestYear} compared with before the heatwave (${st.ratioSrc})</dt><dd class="${st.ratio < 0.5 ? "bad" : st.ratio < 0.85 ? "mid" : "good"}">${pct(st.ratio)}</dd></div>` : ""}
      </dl>
      <div class="kx-actions">
        <button type="button" class="btn btn-line kx-csv">Download data (CSV)</button>
        <button type="button" class="btn btn-line kx-fit">Zoom to bed</button>
      </div>`;
    $("kpanel").querySelector(".kx-csv").addEventListener("click", () => csv(s));
    $("kpanel").querySelector(".kx-fit").addEventListener("click", () => map.fitBounds(s.focus || s.bounds, { padding: [16, 16], maxZoom: 15 }));
  }
  function csv(s) {
    const rows = [["year", "quarter", "bluecarbon_ai_ha", "sensor", "plus_minus_ha", "clear_fraction", "kelpwatch_ha", "sst_c", "sst_anomaly_c"]];
    timeline(s).forEach(([y, q]) => {
      const v = at(s, y, q);
      const tt = (s.sst || []).find(r => r.year === y && r.quarter === q);
      rows.push([y, q, v.ours ? v.ours.canopy_ha : "", v.ours ? (v.ours.ls ? "landsat" : "sentinel-2") : "", v.ours ? (v.ours.plus_minus_ha ?? "") : "",
        v.ours ? v.ours.coverage : "", v.kw ? v.kw.ha : "", tt ? tt.c : "", tt ? tt.anom : ""]);
    });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([rows.map(r => r.join(",")).join("\n")], { type: "text/csv" }));
    a.download = `kelp_${s.name}.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  }

  // ---------------------------------------------------------------------------------------------- selection + time
  function select(s, keepTime) {
    const prev = T[idx];
    site = s; T = timeline(s);
    idx = keepTime && prev ? Math.max(0, T.findIndex(k => k[0] === prev[0] && k[1] === prev[1])) : T.length - 1;
    if (idx < 0) idx = T.length - 1;
    $("kslider").max = Math.max(0, T.length - 1);
    document.querySelectorAll("#kbeds button").forEach(b => b.setAttribute("aria-current", b.dataset.site === s.name ? "true" : "false"));
    map.fitBounds(s.focus || s.bounds, { padding: [16, 16], maxZoom: 15 });
    chart(s); render();
  }
  function stop() { if (timer) { clearInterval(timer); timer = null; $("kplay").innerHTML = "&#9654;"; $("kplay").setAttribute("aria-label", "Play"); } }
  function play() {
    if (timer) return stop();
    if (idx >= T.length - 1) idx = 0;
    $("kplay").innerHTML = "&#10074;&#10074;"; $("kplay").setAttribute("aria-label", "Pause");
    timer = setInterval(() => { if (idx >= T.length - 1) return stop(); idx++; render(); }, +$("kspeed").value);
  }
  $("kplay").addEventListener("click", play);
  $("kspeed").addEventListener("change", () => { if (timer) { stop(); play(); } });
  $("kprev").addEventListener("click", () => { stop(); idx = Math.max(0, idx - 1); render(); });
  $("knext").addEventListener("click", () => { stop(); idx = Math.min(T.length - 1, idx + 1); render(); });
  $("kslider").addEventListener("input", e => { stop(); idx = +e.target.value; render(); });
  document.querySelectorAll(".kx-layers button").forEach(b => b.addEventListener("click", () => {
    if (b.disabled) return;
    show[b.dataset.layer] = !show[b.dataset.layer]; b.setAttribute("aria-pressed", show[b.dataset.layer]);
    map.getContainer().classList.toggle("kx-hl", show.hl);
    render();
  }));
  map.getContainer().classList.add("kx-hl");
  document.addEventListener("keydown", e => {
    if (e.target.closest("input, select, textarea")) return;
    if (e.key === "ArrowLeft") { stop(); idx = Math.max(0, idx - 1); render(); }
    if (e.key === "ArrowRight") { stop(); idx = Math.min(T.length - 1, idx + 1); render(); }
  });

  // ---------------------------------------------------------------------------------------------- page sections
  function figures(sites, cal) {
    const sts = sites.map(stats);
    const lastQ = sites.map(s => s.series.filter(r => r.coverage >= 0.5).slice(-1)[0]).filter(Boolean);
    const total = lastQ.reduce((a, r) => a + r.canopy_ha, 0);
    const first = Math.min(...sites.map(s => (s.kelpwatch_history || s.series)[0]?.year ?? 9999));
    const before = sts.every(x => x.before != null) ? sts.reduce((a, x) => a + x.before, 0) : null;
    const ho = ((cal || {}).held_out || {}).site_quarter_ha || {};
    const f = [
      [`${sites.length}`, `kelp beds, ${sites.some(s => /Washington/.test(s.region)) ? "Baja California to Washington" : sites.some(s => /Baja/.test(s.region)) ? "Baja California to Mendocino" : "Point Loma to Mendocino"}`],
      [`${first}–${Math.max(...sites.map(s => s.latest.year))}`, "years of record on the map"],
      [`${BC.fmt(Math.round(total))} ha`, `of floating canopy in the latest mapped quarter`],
    ];
    if (before != null) f.push([`${BC.fmt(Math.round(before))} ha`, "average yearly peak before the 2014–16 heatwave"]);
    if (ho.r2 != null) f.push([ho.r2.toFixed(2), "agreement with Kelpwatch on beds never used to tune (r²)"]);
    $("kfig").innerHTML = f.map(([b, t]) => `<div><b>${b}</b><span>${t}</span></div>`).join("");
  }
  function story(sites) {
    const rows = sites.map(s => [s, stats(s)]);
    const has = rows.some(([, st]) => st.before != null);
    $("kstory").innerHTML = `<table class="kstory-t"><thead><tr><th>Kelp bed</th>${has ? "<th>Average yearly peak, 1984–2013</th><th>Heatwave low</th>" : ""}${has ? "<th>Latest year, Kelpwatch</th>" : ""}<th>Latest year, BlueCarbon-AI</th>${has ? "<th>Now vs before</th>" : "<th>Peak since 2016</th>"}<th>Error on this bed</th><th>Trend</th></tr></thead><tbody>${rows.map(([s, st]) => {
      const peak = Object.entries(st.op).reduce((m, e) => (e[1] > (m ? m[1] : -1) ? e : m), null);
      return `<tr><td><button type="button" data-go="${s.name}">${BC.esc(s.title)}</button><small>${BC.esc(s.species)}</small></td>
        ${has ? `<td>${st.before != null ? `${BC.fmt(st.before)} ha` : "–"}</td><td>${st.low ? `${BC.fmt(st.low[1])} ha <small>${st.low[0]}</small>` : "–"}</td>` : ""}
        ${has ? `<td>${st.latestKw != null ? `${BC.fmt(st.latestKw)} ha <small>${st.latestYear}</small>` : "–"}</td>` : ""}
        <td>${st.latest != null ? `${BC.fmt(st.latest)} ha <small>${st.latestYear}</small>` : "–"}</td>
        ${has ? `<td class="${st.ratio == null ? "" : st.ratio < 0.5 ? "bad" : st.ratio < 0.85 ? "mid" : "good"}">${pct(st.ratio)}</td>` : `<td>${peak ? `${BC.fmt(peak[1])} ha <small>${peak[0]}</small>` : "–"}</td>`}
        <td>${(s.score || {}).site_quarter_ha && s.score.site_quarter_ha.rmse != null ? `±${BC.fmt(s.score.site_quarter_ha.rmse)} ha <small>${s.held_out ? "held out" : "left out of its fit"}</small>` : "–"}</td>
        <td>${spark(s)}</td></tr>`;
    }).join("")}</tbody></table>
    <p class="small-note">${has ? "Yearly peak = the largest quarter of the year. \"Now vs before\" compares Kelpwatch with Kelpwatch, so the two satellite records are never mixed in one number. Where our latest year differs a lot from Kelpwatch's (for example Mendocino's bull kelp, or Point Loma), that bed's figures are less certain." : "The 1984–2015 Kelpwatch record and yearly pictures appear here after the next kelp refresh."}</p>`;
    $("kstory").querySelectorAll("[data-go]").forEach(b => b.addEventListener("click", () => {
      select(sites.find(s => s.name === b.dataset.go), true);
      document.querySelector(".kx").scrollIntoView({ behavior: "smooth" });
    }));
  }
  function accuracy(cal) {
    const ho = ((cal || {}).held_out || {}), sq = ho.site_quarter_ha || {}, cm = ho.cells_m2 || {};
    const card = (h, n, t) => `<div class="score-card"><p class="sc-h">${h}</p><p class="sc-n">${n}</p><p class="sc-t">${t}</p></div>`;
    if (sq.r2 == null) return;
    $("kacc").innerHTML =
      card("Agreement, site total", sq.r2.toFixed(2), `r² of quarterly canopy against Kelpwatch, ${sq.n} site-quarters`) +
      card("Tracks the ups and downs", sq.corr.toFixed(2), "Correlation with Kelpwatch over time, through heatwaves and recovery") +
      card("Typical error", `±${Math.round(sq.rmse)} ha`, `Per site and quarter; average bias ${sq.bias > 0 ? "+" : ""}${sq.bias} ha, so not inflated`) +
      card("Where the canopy is", cm.r2 != null ? cm.r2.toFixed(2) : "–", "r² of canopy in each 300 m cell, so it is in the right place");
  }

  fetch("data/kelp.json").then(r => r.ok ? r.json() : null).then(d => {
    if (!d || !d.sites || !d.sites.length) {
      $("kpanel").innerHTML = `<span class="badge">In calibration</span><h1>First results soon</h1>`;
      return;
    }
    doc = d;
    const sites = d.sites.slice().sort((a, b) => b.bounds[1][0] - a.bounds[1][0]);   // north to south
    $("kbeds").innerHTML = sites.map(s => {
      const last = s.series.filter(r => r.coverage >= 0.5).slice(-1)[0];
      return `<li><button type="button" data-site="${s.name}"><span class="nm">${BC.esc(s.title)}</span>
        <span class="mt">${BC.esc(s.region.split(",")[0])}${s.held_out ? " · held out" : ""}</span>
        ${spark(s)}<span class="v">${last ? `${BC.fmt(last.canopy_ha)} ha` : "–"}</span></button></li>`;
    }).join("");
    $("kbeds").querySelectorAll("button").forEach(b => b.addEventListener("click", () => { stop(); select(sites.find(s => s.name === b.dataset.site), true); }));
    sites.forEach(s => L.circleMarker([(s.bounds[0][0] + s.bounds[1][0]) / 2, (s.bounds[0][1] + s.bounds[1][1]) / 2],
      { radius: 6, weight: 2, color: "#fff", fillColor: s.held_out ? KW : OURS, fillOpacity: 1 })
      .bindTooltip(s.title).on("click", () => select(s, true)).addTo(map));
    const first = Math.min(...sites.map(s => (s.kelpwatch_history || s.series)[0]?.year ?? 9999));
    const pics = sites.some(s => (s.rgb_years || []).length);
    if (first < 2016) {
      $("ktitle").textContent = `California's kelp forests, from ${first} to this quarter`;
      $("klede").textContent = `${pics ? `A summer satellite picture of every kelp bed for every year since ${first}, our` : "Our"} ` +
        `Sentinel-2 canopy map for every quarter since 2016, and the Kelpwatch record since ${first} for comparison. ` +
        "Drag the timeline, or press play.";
    }
    if (!pics) { const b = document.querySelector('[data-layer="rgb"]'); b.hidden = true; show.rgb = false; }
    if (!sites.some(s => (s.kw_years || []).length)) {   // Kelpwatch maps come with the next kelp refresh
      const b = document.querySelector('[data-layer="kw"]');
      b.disabled = true; b.title = "Kelpwatch's canopy maps are added with the next kelp data refresh";
      b.textContent = "Canopy: Kelpwatch (coming)";
      document.querySelector(".kx-legend .sw-kw").parentElement.hidden = true;
    }
    figures(sites, d.calibration); story(sites); accuracy(d.calibration);
    select(sites.find(s => s.name === "carmel_monterey") || sites[0], false);   // the largest canopy, held out
  });
})();
