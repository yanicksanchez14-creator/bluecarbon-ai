/* Front page: hero reveal, figures, site cards, world map, accuracy bars. */
(function () {
  // ---- hero: drag the handle to reveal the habitat map over the satellite image
  const hero = document.getElementById("hero"), handle = document.getElementById("handle");
  if (hero && handle) {
    const set = pct => {
      pct = Math.max(0, Math.min(100, pct));
      hero.style.setProperty("--reveal", pct + "%");
      handle.setAttribute("aria-valuenow", Math.round(pct));
    };
    let drag = false;
    const at = e => set(((e.clientX - hero.getBoundingClientRect().left) / hero.clientWidth) * 100);
    handle.addEventListener("pointerdown", e => { drag = true; handle.setPointerCapture(e.pointerId); });
    handle.addEventListener("pointermove", e => drag && at(e));
    handle.addEventListener("pointerup", () => { drag = false; });
    handle.addEventListener("keydown", e => {
      const v = parseFloat(handle.getAttribute("aria-valuenow"));
      if (e.key === "ArrowLeft") { set(v - 4); e.preventDefault(); }
      if (e.key === "ArrowRight") { set(v + 4); e.preventDefault(); }
    });
    // one orchestrated moment on load: the map sweeps in from the right edge
    if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      const start = performance.now(), from = 100, to = window.innerWidth < 860 ? 84 : 62;
      const step = t => {
        const k = Math.min(1, (t - start) / 1400), e = 1 - Math.pow(1 - k, 3);
        set(from + (to - from) * e);
        if (k < 1 && !drag) requestAnimationFrame(step);
      };
      set(from);
      setTimeout(() => requestAnimationFrame(step), 350);
    } else set(window.innerWidth < 860 ? 84 : 62);
  }

  BC.load().then(doc => {
    const S = doc.summary, C = BC.classMap(doc);
    document.querySelectorAll("[data-k]").forEach(el => { el.textContent = BC.fmt(S[el.dataset.k]); });

    // hero legend
    const lg = document.getElementById("hero-legend");
    if (lg) lg.innerHTML = ["mangrove", "seagrass", "saltmarsh", "tidal_flat", "water", "other_land"].map(k =>
      `<span><i style="background:${C[k].color}"></i>${C[k].name}</span>`).join("");

    // carbon card (Tampa Bay)
    const tb = doc.sites.find(s => s.id === "tampa_bay_fl");
    const card = document.getElementById("carbon-card");
    if (tb && card) {
      card.querySelector("b").textContent = BC.big(tb.carbon.stock_tco2e.mean);
      document.getElementById("cc-seq").textContent = `${BC.fmt(tb.carbon.seq_tco2e_yr.mean)} t CO₂ absorbed / yr`;
      document.getElementById("cc-soil").textContent = `${BC.fmt(BC.blueTotal(tb))} ha blue carbon`;
    }

    const sites = doc.sites.filter(s => s.kind !== "change");
    const order = [...sites].sort((a, b) => (b.held_out - a.held_out) || (BC.blueTotal(b) - BC.blueTotal(a)));

    // ---- site cards
    const grid = document.getElementById("site-grid");
    let filter = "all", showAll = false;
    const match = s => filter === "all" ? true : filter === "held" ? s.held_out : (s.areas_ha[filter] || 0) >= 50;
    const statLine = s => BC.BLUE.filter(k => (s.areas_ha[k] || 0) >= 1).sort((a, b) => s.areas_ha[b] - s.areas_ha[a]).slice(0, 2)
      .map(k => `<span title="${C[k].name}"><i style="background:${C[k].color}"></i>${BC.fmt(s.areas_ha[k])} ha ${C[k].name.toLowerCase()}</span>`).join("");
    const render = () => {
      const list = order.filter(match);
      const shown = showAll ? list : list.slice(0, 12);
      grid.innerHTML = shown.map(s => `
        <a class="site-card" href="explore.html#${s.id}">
          <div class="thumb" style="background-image:url('${s.images.thumb}')">${s.held_out ? '<span class="tag">Never seen in training</span>' : ""}</div>
          <div class="body"><h3>${BC.esc(s.title)}</h3><div class="where">${BC.esc(s.region)}</div>
          <div class="stats">${statLine(s)}</div></div></a>`).join("");
      const more = document.querySelector(".more");
      more.innerHTML = list.length > 12 && !showAll
        ? `<button class="btn btn-line" id="show-all">Show all ${list.length} sites</button>`
        : `<a class="btn btn-line" href="explore.html">Open the full map</a>`;
      const b = document.getElementById("show-all");
      if (b) b.addEventListener("click", () => { showAll = true; render(); });
    };
    document.querySelectorAll(".chip").forEach(ch => ch.addEventListener("click", () => {
      document.querySelectorAll(".chip").forEach(c => c.setAttribute("aria-pressed", c === ch));
      filter = ch.dataset.f; showAll = false; render();
    }));
    render();

    // ---- world map
    if (window.L && document.getElementById("world")) {
      const m = L.map("world", { worldCopyJump: true, scrollWheelZoom: false, zoomControl: true, attributionControl: true })
        .setView([18, 10], 2);
      L.tileLayer(BC.ESRI, { attribution: BC.ESRI_ATTR, maxZoom: 12 }).addTo(m);
      sites.forEach(s => {
        L.circleMarker(s.center, { radius: 6, weight: 2, color: "#fff", fillColor: s.held_out ? "#6fd3bf" : "#d9c9b4", fillOpacity: 1 })
          .bindTooltip(`${BC.esc(s.title)}, ${BC.esc(s.country)}`, { direction: "top", offset: [0, -6] })
          .on("click", () => { location.href = `explore.html#${s.id}`; })
          .addTo(m);
      });
    }

    // ---- accuracy bars
    const bars = document.getElementById("bars");
    if (bars) {
      const rel = v => v >= 0.7 ? ["Reliable", "hi"] : v >= 0.4 ? ["Fair", ""] : ["Limited", "lo"];
      const keys = ["mangrove", "saltmarsh", "seagrass", "water", "tidal_flat", "other_land", "freshwater"];
      bars.innerHTML = keys.filter(k => doc.model.iou[k] !== undefined).map(k => {
        const v = doc.model.iou[k], [lab, cls] = rel(v);
        return `<div class="bar-row"><span>${C[k].name}</span><div class="track"><div class="fill" style="width:${v * 100}%;background:${C[k].color}"></div></div>
          <span class="v">${v.toFixed(2)}</span><span class="r ${cls}">${lab}</span></div>`;
      }).join("") + `<p class="scale">Mean over all habitats: ${(doc.model.miou ?? 0).toFixed(2)}. Reliable at 0.70 and above, fair from 0.40.</p>`;
    }
  }).catch(err => console.error(err));
})();
