/* Shared helpers for the BlueCarbon-AI website. */
const BC = (() => {
  const ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
  const ESRI_ATTR = "Imagery © Esri, Maxar, Earthstar Geographics";
  const BLUE = ["mangrove", "saltmarsh", "seagrass"];
  let cache = null;

  const load = () => cache || (cache = fetch("data/sites.json").then(r => {
    if (!r.ok) throw new Error(`sites.json: ${r.status}`);
    return r.json();
  }));

  // Plain numbers with separators; 1 decimal under 10.
  const fmt = (x, d) => {
    if (x === null || x === undefined || Number.isNaN(x)) return "–";
    const digits = d ?? (Math.abs(x) < 10 && x !== 0 ? 1 : 0);
    return x.toLocaleString("en-US", { maximumFractionDigits: digits, minimumFractionDigits: 0 });
  };
  // Large numbers in words: 17.2 million.
  const big = x => {
    if (x >= 1e9) return `${fmt(x / 1e9, 1)} billion`;
    if (x >= 1e6) return `${fmt(x / 1e6, 1)} million`;
    return fmt(x);
  };
  const coords = ([lat, lon]) =>
    `${Math.abs(lat).toFixed(2)}° ${lat >= 0 ? "N" : "S"}, ${Math.abs(lon).toFixed(2)}° ${lon >= 0 ? "E" : "W"}`;
  const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const classMap = doc => Object.fromEntries(doc.classes.map(c => [c.key, c]));
  const blueTotal = s => BLUE.reduce((a, k) => a + (s.areas_ha[k] || 0), 0);

  // header: border once scrolled, mobile menu
  document.addEventListener("DOMContentLoaded", () => {
    const h = document.querySelector(".site-header");
    const onScroll = () => h && h.classList.toggle("scrolled", window.scrollY > 8);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    const t = document.querySelector(".nav-toggle"), nav = document.getElementById("nav");
    if (t && nav) {
      t.addEventListener("click", () => {
        const open = nav.classList.toggle("open");
        t.setAttribute("aria-expanded", open);
        t.textContent = open ? "Close" : "Menu";
      });
      nav.querySelectorAll("a").forEach(a => a.addEventListener("click", () => {
        nav.classList.remove("open"); t.setAttribute("aria-expanded", false); t.textContent = "Menu";
      }));
    }
  });

  return { ESRI, ESRI_ATTR, BLUE, load, fmt, big, coords, esc, classMap, blueTotal };
})();
