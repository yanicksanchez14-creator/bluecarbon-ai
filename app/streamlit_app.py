"""BlueCarbon-AI: coastal blue carbon mapping from Sentinel-2.

Runs with zero credentials on the bundled demo sites. With Earth Engine credentials in
Streamlit secrets (GEE_SERVICE_ACCOUNT), the Analyze tab maps any coastline on demand.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import Draw, Fullscreen
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# Streamlit reruns this script in a long-lived process, so an updated deploy can otherwise keep an
# old copy of the bluecarbon package in memory. Drop it so every run imports the current code.
for _m in [m for m in sys.modules if m == "bluecarbon" or m.startswith("bluecarbon.")]:
    del sys.modules[_m]

from bluecarbon.config import load_config  # noqa: E402
from bluecarbon.schema import BLUE_CARBON_KEYS, CLASSES  # noqa: E402

DEMO_DIR = Path(os.environ.get("BLUECARBON_DEMO_DIR", ROOT / "demo_data"))
_CURRENT = ROOT / "models" / "current.txt"
DEFAULT_MODEL = ROOT / "models" / (_CURRENT.read_text().strip() if _CURRENT.exists()
                                   else "pilot_spectral_mission_bay_2018.json")
REPO = "https://github.com/yanicksanchez14-creator/bluecarbon-ai"
MAX_AREA_KM2 = 60
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Imagery © Esri, Maxar, Earthstar Geographics"

CFG = load_config(ROOT / "configs" / "default.yaml")
CLS = {c.key: c for c in CLASSES}

LOGO = """<svg width="34" height="34" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0e5a67"/>
<stop offset="1" stop-color="#082f3d"/></linearGradient></defs>
<rect width="40" height="40" rx="10" fill="url(#g)"/>
<path d="M20 8c5.5 3.2 7.6 8.4 5.2 13.2-1.4 2.7-3.3 3.8-5.2 4.3-1.9-.5-3.8-1.6-5.2-4.3C12.4 16.4 14.5 11.2 20 8z" fill="#7fd6c2"/>
<path d="M20 11v14" stroke="#0b3f4c" stroke-width="1.4" stroke-linecap="round"/>
<path d="M7 28.5c2.2 0 2.2-1.6 4.4-1.6s2.2 1.6 4.4 1.6 2.2-1.6 4.4-1.6 2.2 1.6 4.4 1.6 2.2-1.6 4.4-1.6 2.2 1.6 3.8 1.6"
 fill="none" stroke="#ffffff" stroke-width="2" stroke-linecap="round"/>
<path d="M7 33c2.2 0 2.2-1.6 4.4-1.6s2.2 1.6 4.4 1.6 2.2-1.6 4.4-1.6 2.2 1.6 4.4 1.6 2.2-1.6 4.4-1.6 2.2 1.6 3.8 1.6"
 fill="none" stroke="#ffffff" stroke-opacity=".45" stroke-width="2" stroke-linecap="round"/></svg>"""

st.set_page_config(page_title="BlueCarbon-AI", page_icon=str(ROOT / "app" / "assets" / "favicon.png"), layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------- design system
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
:root{
  --ink:#0b1f2a; --ink-2:#344651; --muted:#6a7a84; --line:#e4eaed; --line-2:#eef2f4;
  --bg:#f5f7f8; --card:#ffffff; --brand:#0e5a67; --brand-ink:#083744; --accent:#14a3a0;
  --accent-soft:#e6f5f3; --warn-bg:#fff6e8; --warn-ink:#8a5a00; --radius:14px;
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
  font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }
.stApp{background:var(--bg); color:var(--ink);}
header[data-testid="stHeader"]{background:transparent; height:0;}
div[data-testid="stToolbar"]{right:1rem;}
.block-container{padding:1.2rem 2.2rem 3rem; max-width:1320px;}
h1,h2,h3,h4{color:var(--ink); letter-spacing:-.015em;}

/* top bar */
.bc-top{display:flex; align-items:center; justify-content:space-between; padding:.35rem 0 1.1rem;
  border-bottom:1px solid var(--line); margin-bottom:1.6rem;}
.bc-brand{display:flex; align-items:center; gap:.7rem;}
.bc-word{font-weight:700; font-size:1.18rem; color:var(--ink); letter-spacing:-.02em;}
.bc-word span{color:var(--accent); font-weight:600;}
.bc-i{display:inline-flex; align-items:center; position:relative; color:#8a9aa3; margin-left:.3rem;
  cursor:help; vertical-align:-1px; outline:none;}
.bc-i:hover, .bc-i:focus{color:var(--brand);}
.bc-i::after{content:attr(data-tip); position:absolute; left:50%; bottom:calc(100% + 8px); transform:translateX(-50%);
  width:max-content; max-width:280px; background:#0b1f2a; color:#f2f6f7; font-size:.78rem; font-weight:400;
  line-height:1.45; padding:.55rem .7rem; border-radius:8px; box-shadow:0 6px 24px rgba(0,0,0,.18);
  opacity:0; pointer-events:none; transition:opacity .12s; z-index:1000; white-space:normal; text-transform:none;
  letter-spacing:normal; text-align:left;}
.bc-i:hover::after, .bc-i:focus::after{opacity:1;}
.bc-glance .bc-i{color:#9fe3d6;}
.bc-how{display:flex; flex-direction:column; gap:.6rem;}
.bc-howstep{display:flex; gap:.8rem; align-items:flex-start; background:var(--card); border:1px solid var(--line);
  border-radius:12px; padding:.75rem .9rem;}
.bc-howstep .num{flex:none; width:24px; height:24px; border-radius:50%; background:var(--accent-soft); color:var(--brand);
  font-weight:700; font-size:.8rem; display:flex; align-items:center; justify-content:center;}
.bc-howstep b{display:block; font-size:.92rem; color:var(--ink);}
.bc-howstep small{display:block; color:var(--muted); font-size:.8rem; line-height:1.45; margin-top:.1rem;}
.bc-glance-h{font-size:.8rem; font-weight:600; color:#9fe3d6; letter-spacing:.02em;}
.bc-relhead{font-size:.74rem; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.05em;}
.bc-sumrow{display:grid; grid-template-columns:190px 1fr; gap:1rem; padding:.7rem 0; border-bottom:1px solid var(--line-2);}
.bc-sumrow:last-child{border-bottom:none;}
.bc-sumrow .k{font-weight:600; font-size:.9rem; color:var(--ink);}
.bc-sumrow .v{font-size:.94rem; color:var(--ink-2); line-height:1.6;}
.bc-tag{font-size:.72rem; font-weight:600; color:var(--brand); background:var(--accent-soft);
  border:1px solid #cfe9e5; padding:2px 8px; border-radius:999px; margin-left:.35rem;}
.bc-links a{color:var(--ink-2); text-decoration:none; font-size:.88rem; font-weight:500; margin-left:1.4rem;}
.bc-links a:hover{color:var(--brand);}

/* hero */
.bc-hero{display:grid; grid-template-columns:1.3fr 1fr; gap:2.4rem; align-items:center; margin-bottom:1.8rem;}
.bc-eyebrow{font-size:.74rem; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:var(--accent);}
.bc-hero h1{font-size:2.2rem; line-height:1.15; font-weight:700; margin:0 0 .8rem; padding:0;}
.bc-hero p{color:var(--ink-2); font-size:1.02rem; line-height:1.6; margin:0; max-width:620px;}
.bc-facts{display:flex; gap:.6rem; flex-wrap:wrap; justify-content:flex-end;}
.bc-fact{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.6rem .85rem; min-width:118px;}
.bc-fact b{display:block; font-size:1.05rem; color:var(--ink);}
.bc-fact small{color:var(--muted); font-size:.74rem;}

/* tabs */
div[data-baseweb="tab-list"]{gap:.25rem; border-bottom:1px solid var(--line);}
button[data-baseweb="tab"]{padding:.55rem .95rem !important; border-radius:10px 10px 0 0;}
button[data-baseweb="tab"] p{font-weight:600 !important; font-size:.92rem !important; color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--brand-ink);}
div[data-baseweb="tab-highlight"]{background:var(--brand) !important; height:2px;}
div[data-baseweb="tab-border"]{display:none;}

/* cards */
.bc-card{background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:1.1rem 1.2rem;}
.bc-kpis{display:grid; grid-template-columns:repeat(4,1fr); gap:.9rem; margin:.4rem 0 1.1rem;}
.bc-kpi .l{font-size:.86rem; color:var(--ink-2); font-weight:600;}
.bc-kpi .v{font-size:1.65rem; font-weight:700; color:var(--ink); margin:.25rem 0 .1rem; letter-spacing:-.02em;}
.bc-kpi .v small{font-size:.85rem; font-weight:500; color:var(--muted); margin-left:.25rem;}
.bc-kpi .s{font-size:.76rem; color:var(--muted);}
.bc-range{position:relative; height:6px; border-radius:6px; background:var(--line-2); margin:.55rem 0 .3rem;}
.bc-range i{position:absolute; top:0; bottom:0; border-radius:6px; background:#9fd9d1;}
.bc-range em{position:absolute; top:-3px; width:3px; height:12px; border-radius:2px; background:var(--brand);}
.bc-section{font-size:.74rem; font-weight:600; letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
  margin:1.6rem 0 .6rem;}
.bc-site h3{margin:0; font-size:1.25rem;}
.bc-site .meta{color:var(--muted); font-size:.84rem; margin:.15rem 0 .6rem;}
.bc-site p{color:var(--ink-2); font-size:.92rem; line-height:1.55; margin:0;}
.bc-legend{display:flex; flex-direction:column; gap:.42rem; margin-top:.2rem;}
.bc-legend .row{display:grid; grid-template-columns:14px 1fr auto auto; gap:.2rem .6rem; align-items:center; font-size:.88rem;}
.bc-legend .bar{grid-column:2 / 5; height:5px; background:var(--line-2); border-radius:4px; overflow:hidden; margin-bottom:.25rem;}
.bc-legend .bar i{display:block; height:100%; border-radius:4px;}
.bc-legend .sw{width:12px; height:12px; border-radius:3px;}
.bc-legend .n{color:var(--ink);} .bc-legend .a{font-variant-numeric:tabular-nums; color:var(--ink); font-weight:600;}
.bc-legend .p{font-variant-numeric:tabular-nums; color:var(--muted); font-size:.8rem; min-width:42px; text-align:right;}
.bc-legend .bc{font-size:.66rem; font-weight:600; color:var(--brand); background:var(--accent-soft);
  padding:1px 6px; border-radius:999px; margin-left:.35rem;}
.bc-badge{display:inline-flex; align-items:center; gap:.35rem; font-size:.74rem; font-weight:600; padding:3px 10px;
  border-radius:999px; background:var(--warn-bg); color:var(--warn-ink); border:1px solid #f3d9a8;}
.bc-note{font-size:.8rem; color:var(--muted); line-height:1.5;}
.bc-map iframe{border-radius:var(--radius); border:1px solid var(--line) !important;}
.bc-hint{font-size:.78rem; color:var(--muted); margin-top:.35rem;}

/* methodology */
.bc-steps{display:grid; grid-template-columns:repeat(5,1fr); gap:.8rem; margin:.4rem 0 1.4rem;}
.bc-step{background:var(--card); border:1px solid var(--line); border-radius:var(--radius); padding:1rem;}
.bc-step .k{font-family:'JetBrains Mono', monospace; font-size:.72rem; color:var(--accent); font-weight:500;}
.bc-step h4{margin:.3rem 0 .35rem; font-size:.98rem;}
.bc-step p{margin:0; font-size:.82rem; line-height:1.5; color:var(--ink-2);}
.bc-doc{max-width:880px;}
.bc-doc h3{font-size:1.15rem; margin:1.6rem 0 .5rem;}
.bc-doc p, .bc-doc li{color:var(--ink-2); font-size:.94rem; line-height:1.65;}
.bc-table{width:100%; border-collapse:collapse; font-size:.88rem; background:var(--card); border:1px solid var(--line);
  border-radius:var(--radius); overflow:hidden;}
.bc-table th{background:#f8fafb; text-align:left; font-weight:600; color:var(--ink-2); padding:.6rem .8rem;
  border-bottom:1px solid var(--line); font-size:.78rem; text-transform:uppercase; letter-spacing:.05em;}
.bc-table td{padding:.6rem .8rem; border-bottom:1px solid var(--line-2); color:var(--ink); vertical-align:top;}
.bc-table tr:last-child td{border-bottom:none;}
.bc-table td.num{text-align:right; font-variant-numeric:tabular-nums;}
.bc-formula{font-family:'JetBrains Mono', monospace; font-size:.86rem; background:var(--card); border:1px solid var(--line);
  border-left:3px solid var(--accent); border-radius:8px; padding:.8rem 1rem; color:var(--ink); margin:.6rem 0;}
.bc-refs li{font-size:.84rem;}
.bc-footer{margin-top:3rem; padding-top:1.2rem; border-top:1px solid var(--line); color:var(--muted); font-size:.8rem;
  display:flex; justify-content:space-between;}
.bc-footer a{color:var(--ink-2);}

/* widgets */
div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, div[data-testid="stSegmentedControl"] label
  {font-size:.78rem !important; color:var(--muted) !important; font-weight:500 !important;}
div[data-baseweb="select"] > div{border-radius:10px; border-color:var(--line); background:var(--card);}
.stButton button[kind="primary"]{background:var(--brand); border:none; border-radius:10px; font-weight:600;}
.stButton button[kind="primary"]:hover{background:var(--brand-ink);}
div[data-testid="stExpander"]{border:1px solid var(--line); border-radius:var(--radius); background:var(--card);}
@media (max-width: 900px){
  .bc-hero{grid-template-columns:1fr;} .bc-facts{justify-content:flex-start;}
  .bc-kpis{grid-template-columns:repeat(2,1fr);} .bc-steps{grid-template-columns:1fr 1fr;}
  .block-container{padding:1rem 1rem 2rem;}
}

.bc-glance{background:linear-gradient(135deg,#0e5a67 0%,#083744 100%); color:#fff; border-radius:var(--radius);
  padding:1.3rem 1.5rem; margin:.3rem 0 1.2rem;}
.bc-glance .bc-eyebrow{color:#9fe3d6;}
.bc-glance p{margin:.4rem 0 0; font-size:1.08rem; line-height:1.65; color:#e8f3f4;}
.bc-glance b{color:#fff;}
.bc-grid3{display:grid; grid-template-columns:repeat(3,1fr); gap:.9rem;}
.bc-hab .name{display:flex; align-items:center; gap:.5rem; font-weight:600; font-size:.95rem;}
.bc-hab .sw, .bc-relrow .sw{width:11px; height:11px; border-radius:3px; display:inline-block;}
.bc-hab .big{font-size:1.9rem; font-weight:700; margin:.55rem 0 0; letter-spacing:-.02em;}
.bc-hab .big small{font-size:.9rem; font-weight:500; color:var(--muted);}
.bc-hab .big.muted{color:#b3bec4; font-size:1.35rem;}
.bc-hab .sub{font-size:.8rem; color:var(--muted); margin-bottom:.6rem;}
.bc-hab .kv{display:flex; justify-content:space-between; font-size:.86rem; padding:.35rem 0; border-top:1px solid var(--line-2);}
.bc-hab .kv span{color:var(--muted);} .bc-hab .kv b{font-variant-numeric:tabular-nums;}
.bc-hab .desc{font-size:.8rem; color:var(--ink-2); line-height:1.5; margin-top:.6rem;}
.bc-kpi .what{font-size:.82rem; color:var(--ink-2); line-height:1.5; margin-top:.6rem; padding-top:.6rem;
  border-top:1px solid var(--line-2);}
.bc-maplegend{display:flex; flex-wrap:wrap; gap:.4rem 1.1rem; margin-top:.55rem; font-size:.84rem; color:var(--ink-2);}
.bc-maplegend span{display:inline-flex; align-items:center;}
.bc-maplegend i{width:11px; height:11px; border-radius:3px; margin-right:.4rem; display:inline-block;}
.bc-rels{display:flex; flex-direction:column;}
.bc-relrow{display:grid; grid-template-columns:14px 1fr auto 72px; align-items:center; gap:.6rem; font-size:.88rem;
  padding:.45rem 0; border-bottom:1px solid var(--line-2);}
.bc-relrow .sc{text-align:right; font-variant-numeric:tabular-nums; color:var(--muted); font-size:.8rem;}
.bc-rel{font-size:.72rem; font-weight:600; padding:2px 9px; border-radius:999px; border:1px solid;}
.bc-rel-hi{background:#e8f6ee; color:#12683b; border-color:#bfe3cc;}
.bc-rel-md{background:#fff6e8; color:#8a5a00; border-color:#f3d9a8;}
.bc-rel-lo{background:#fdecec; color:#9b2c2c; border-color:#f3c4c4;}
.bc-rel-na{background:#f1f4f5; color:#6a7a84; border-color:#e0e6e9;}
.bc-found{margin-bottom:.8rem;}
.bc-up{color:#12683b; font-weight:600;} .bc-down{color:#9b2c2c; font-weight:600;}
.bc-status{display:inline-block; font-size:.72rem; font-weight:600; padding:2px 10px; border-radius:999px;
  margin-bottom:.45rem; border:1px solid;}
.bc-st-held{background:#e6f5f3; color:#0e5a67; border-color:#cfe9e5;}
.bc-st-change{background:#eef2ff; color:#3730a3; border-color:#c7d2fe;}
.bc-st-train{background:#f1f4f5; color:#52616b; border-color:#e0e6e9;}
.bc-maplegend i{box-sizing:border-box;} .bc-found p{margin:0; font-size:1.02rem; line-height:1.7; color:var(--ink);}
.bc-summary ul{margin:.5rem 0 0; padding-left:1.1rem;}
.bc-summary li{font-size:.95rem; line-height:1.65; color:var(--ink); margin:.2rem 0;}
.bc-h2{font-size:1.15rem; font-weight:700; margin:2rem 0 .25rem; color:var(--ink);}
.bc-h2sub{font-size:.88rem; color:var(--muted); margin:0 0 .8rem;}
@media (max-width: 900px){ .bc-grid3{grid-template-columns:1fr;} .bc-sumrow{grid-template-columns:1fr; gap:.2rem;} }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- helpers
def fmt(x: float | None, digits: int = 0) -> str:
    """Plain numbers with thousands separators (no k/M abbreviations)."""
    if x is None:
        return "–"
    a = abs(x)
    if a >= 1000:
        return f"{round(x, -2 if a >= 10000 else -1):,.0f}"
    if a >= 100:
        return f"{x:,.0f}"
    return f"{x:,.{max(digits, 1)}f}"


def tip(text: str) -> str:
    """Small (i) icon that shows an explanation on hover or tap."""
    t = text.replace('"', "&quot;")
    return (f'<span class="bc-i" tabindex="0" data-tip="{t}"><svg viewBox="0 0 16 16" width="13" height="13" '
            'aria-hidden="true"><circle cx="8" cy="8" r="7" fill="none" stroke="currentColor" stroke-width="1.4"/>'
            '<circle cx="8" cy="4.9" r=".95" fill="currentColor"/><path d="M8 7.2v4.6" stroke="currentColor" '
            'stroke-width="1.5" stroke-linecap="round"/></svg></span>')


T_HA = "A hectare is 10,000 m², about 2.5 acres or one and a half soccer fields."
T_CO2 = ("Tonnes of carbon dioxide (CO₂). Scientists convert the carbon stored in soil and plants into the amount of "
         "CO₂ it would form if released, so it can be compared with emissions.")
T_RANGE = ("We are 90% confident the true value lies in this range. It reflects uncertainty in both the mapped area and "
           "the published carbon values.")
T_TIER1 = ("Soil carbon comes from real soil cores measured near this site (Smithsonian Coastal Carbon Library) when "
           "there are enough of them, and otherwise from the global default values of the UN climate science body "
           "(IPCC). Good for a first estimate; real projects measure their own site.")


def soil_line(v: dict) -> str:
    """One line saying where a habitat's soil carbon value comes from."""
    sl = v.get("soil") or {}
    if sl.get("source") == "measured":
        studies = f"{sl['n_studies']} {'study' if sl['n_studies'] == 1 else 'studies'}"
        return (f"Soil carbon from {sl['n_cores']} measured soil cores within {sl['radius_km']} km "
                f"({studies}): about {sl['soil'][1]:.0f} t carbon per hectare")
    return "Soil carbon: global average (IPCC), no measured cores nearby"
T_ADJ = ("Maps are never perfect. We correct each area using how often the model was right or wrong on test areas "
         "(a standard method from Olofsson et al., 2014).")
T_SCORE = ("How closely the model's map overlapped with trusted reference maps, on areas it never saw during training. "
           "1.00 = perfect match, 0 = no match.")
T_S2 = "Sentinel-2 is a pair of European Space Agency satellites that photograph every coastline on Earth every 5 days, free."
T_PILOT = ("Trained only on hand-labelled data from one bay (Mission Bay, 2018). It is well tested here, but other "
           "coastlines look different. The full model trains on 39 coastal sites on six continents.")

def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def rgba_uri(rgba: np.ndarray) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    plt.imsave(buf, rgba, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fit_view(bounds, width_px: int = 820) -> tuple[list[float], int]:
    (s, w), (n, e) = bounds
    span = max(e - w, (n - s) * 1.25, 1e-4)
    zoom = int(math.floor(math.log2(width_px * 360 / (256 * span))))
    return [(s + n) / 2, (w + e) / 2], max(3, min(zoom, 16))


def make_map(bounds, height: int = 540):
    center, zoom = fit_view(bounds)
    m = folium.Map(location=center, zoom_start=zoom, tiles=None, control_scale=True, zoom_control=True)
    folium.TileLayer(ESRI, attr=ESRI_ATTR, name="Satellite", max_zoom=19).add_to(m)
    Fullscreen(position="topright").add_to(m)
    return m


def overlay(m, uri, bounds, opacity=1.0):
    folium.raster_layers.ImageOverlay(uri, bounds=bounds, opacity=opacity, interactive=False, zindex=2).add_to(m)


def outline(m, bounds):
    (s, w), (n, e) = bounds
    folium.Rectangle([[s, w], [n, e]], color="#ffffff", weight=1.2, fill=False, dash_array="4 4", opacity=0.8).add_to(m)


def _demo_signature() -> str:
    """Changes whenever demo pages are added, removed or regenerated (invalidates the caches below)."""
    return "|".join(f"{p.parent.name}:{p.stat().st_mtime_ns}" for p in sorted(DEMO_DIR.glob("*/meta.json")))


def list_sites() -> dict[str, Path]:
    return _list_sites(_demo_signature())


@st.cache_data
def _list_sites(sig: str) -> dict[str, Path]:
    out = {}
    for f in sorted(DEMO_DIR.glob("*/meta.json")):
        out[json.loads(f.read_text())["title"]] = f.parent
    return out


def best_areas(report: dict) -> dict[str, float]:
    adj = report.get("error_adjusted_areas_ha")
    if adj:
        return {k: adj[k]["adjusted_ha"] if k in adj else 0.0 for k in CLS}
    return {k: report["areas_ha"].get(k, 0.0) for k in CLS}


def range_bar(p05: float, mean: float, p95: float) -> str:
    hi = p95 * 1.15 if p95 > 0 else 1
    lo_pct, hi_pct, m_pct = 100 * p05 / hi, 100 * p95 / hi, 100 * mean / hi
    return (f'<div class="bc-range"><i style="left:{lo_pct:.1f}%;width:{max(hi_pct - lo_pct, 1):.1f}%"></i>'
            f'<em style="left:calc({m_pct:.1f}% - 1px)"></em></div>')


def carbon_table(report: dict) -> str:
    rows = []
    for k, v in report["carbon"]["classes"].items():
        s, q = v["stock_tCO2e"], v["sequestration_tCO2e_per_yr"]
        rows.append(f'<tr><td><span class="sw" style="display:inline-block;width:10px;height:10px;border-radius:3px;'
                    f'background:{CLS[k].color};margin-right:8px"></span>{CLS[k].name}</td>'
                    f'<td class="num">{fmt(v["area_ha"], 1)}</td><td class="num">{fmt(s["mean"])}</td>'
                    f'<td class="num">{fmt(s["p05"])} – {fmt(s["p95"])}</td><td class="num">{fmt(q["mean"], 1)}</td></tr>')
    if not rows:
        rows = ['<tr><td colspan="5">No blue carbon habitat detected.</td></tr>']
    return ('<table class="bc-table"><thead><tr><th>Habitat</th><th style="text-align:right">Area (hectares)</th>'
            '<th style="text-align:right">Carbon stored (t CO₂)</th><th style="text-align:right">Likely range</th>'
            '<th style="text-align:right">Absorbed per year (t CO₂)</th></tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def model_desc(model: dict) -> str:
    if model.get("kind") == "spectral-lgbm":
        return ("Gradient-boosted decision trees (LightGBM) reading each pixel's 10 satellite bands, 9 plant and water "
                "indices, and the surrounding 30 m and 90 m neighbourhood")
    ctx = " plus elevation and tidal context" if model.get("uses_context") else ""
    return f"{model['arch']} neural network · {model['encoder']} encoder · satellite bands and plant / water indices{ctx}"


def model_panel(model: dict | None) -> None:
    if not model:
        return
    t = model.get("test") or {}
    badge = (f'<span class="bc-badge">Pilot model · trained on one bay{tip(T_PILOT)}</span>' if model.get("pilot")
             else '<span class="bc-badge" style="background:#e6f5f3;color:#0e5a67;border-color:#cfe9e5">'
                  'Multi-site model</span>')
    st.markdown(f'{badge}<div style="margin:.7rem 0 .2rem;font-weight:600">{model.get("name", "model")}</div>'
                f'<div class="bc-note">{model_desc(model)}</div>',
                unsafe_allow_html=True)
    if t:
        rows = "".join(
            f'<tr><td>{CLS[k].name}</td><td class="num">{v:.2f}</td><td class="num">{(t["f1"].get(k) or 0):.2f}</td>'
            f'<td class="num">{t["support_px"].get(k, 0):,}</td></tr>' for k, v in t["iou"].items()
            if v is not None and k in CLS)
        st.markdown(
            f'<div style="display:flex;gap:1.6rem;margin:.9rem 0">'
            f'<div><div class="bc-note">Mean IoU (overlap)</div><div style="font-size:1.35rem;font-weight:700">{t["mIoU"]:.2f}</div></div>'
            f'<div><div class="bc-note">Mean F1 score</div><div style="font-size:1.35rem;font-weight:700">{t["macro_f1"]:.2f}</div></div>'
            f'<div><div class="bc-note">Cohen\'s κ</div><div style="font-size:1.35rem;font-weight:700">{t["kappa"]:.2f}</div></div></div>'
            f'<table class="bc-table"><thead><tr><th>Habitat</th><th style="text-align:right">IoU</th>'
            f'<th style="text-align:right">F1</th><th style="text-align:right">Test pixels</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>', unsafe_allow_html=True)
    notes = " ".join(x for x in [model.get("evaluation") and f"Evaluated with {model['evaluation']}.",
                                 model.get("training_data") and f"Trained on {model['training_data']}."] if x)
    if notes:
        st.markdown(f'<div class="bc-note" style="margin-top:.7rem">{notes}</div>', unsafe_allow_html=True)


HABITAT_INFO = {
    "mangrove": "Salt-tolerant trees that grow along tropical coasts. Per hectare, the most carbon-rich ecosystem on Earth.",
    "saltmarsh": "Grassy wetlands flooded by the tides. Their muddy soils keep burying carbon for centuries.",
    "seagrass": "Underwater meadows in shallow, clear water. Their roots trap carbon-rich sediment on the seabed.",
}
CAR_TCO2_PER_YR = 4.6  # US EPA: typical passenger vehicle, tonnes CO2 per year
T_CARS = "Based on the US EPA figure of about 4.6 tonnes of CO₂ per typical passenger car per year."


def car_equiv(tco2: float) -> str:
    n = tco2 / CAR_TCO2_PER_YR
    return fmt(round(n)) if n >= 10 else f"{n:,.1f}"


def reliability(iou: float | None) -> tuple[str, str]:
    if iou is None:
        return "Not tested", "bc-rel-na"
    if iou >= 0.7:
        return "Reliable", "bc-rel-hi"
    if iou >= 0.4:
        return "Fair", "bc-rel-md"
    return "Unreliable", "bc-rel-lo"


def site_area_ha(report: dict) -> float:
    return sum(best_areas(report).values())


def glance_html(meta: dict, report: dict) -> str:
    a = best_areas(report)
    tot = site_area_ha(report) or 1
    bc = sum(a[k] for k in BLUE_CARBON_KEYS)
    parts = [f"{fmt(a[k], 1)} hectares of {CLS[k].name.lower()}" for k in BLUE_CARBON_KEYS if a[k] >= 0.05]
    found = (", ".join(parts[:-1]) + " and " + parts[-1]) if len(parts) > 1 else (parts[0] if parts else "")
    cb = report["carbon"]
    s, q = cb["total_stock_tCO2e"], cb["total_sequestration_tCO2e_per_yr"]
    km2 = tot / 100
    if bc < 0.05:
        body = (f"In this {km2:,.1f} square kilometre area, the AI found no mangrove, salt marsh or seagrass, so there "
                "is no blue carbon to report.")
    else:
        body = (f"In this {km2:,.1f} square kilometre area, the AI found <b>{found}</b>. "
                f"Together these habitats hold an estimated <b>{fmt(s['mean'])} tonnes of CO₂</b>, about what "
                f"<b>{car_equiv(s['mean'])} cars</b> emit in a year{tip(T_CARS)}, and absorb roughly "
                f"<b>{fmt(q['mean'], 1)} more tonnes every year</b>.")
    return body


def habitat_cards(report: dict) -> str:
    a = best_areas(report)
    tot = site_area_ha(report) or 1
    per = report["carbon"]["classes"]
    cards = []
    for k in BLUE_CARBON_KEYS:
        c = CLS[k]
        if a[k] >= 0.05 and k in per:
            v = per[k]
            body = (f'<div class="big">{fmt(a[k], 1)}<small> hectares{tip(T_HA)}</small></div>'
                    f'<div class="sub">{100 * a[k] / tot:.1f}% of the mapped area</div>'
                    f'<div class="kv"><span>Carbon stored{tip(T_CO2)}</span><b>{fmt(v["stock_tCO2e"]["mean"])} t CO₂</b></div>'
                    f'<div class="kv"><span>Absorbed each year</span><b>{fmt(v["sequestration_tCO2e_per_yr"]["mean"], 1)} t CO₂</b></div>'
                    f'<div class="sub" style="margin-top:.4rem">{soil_line(v)}</div>')
        else:
            body = '<div class="big muted">None found</div><div class="sub">Not detected in this area</div>'
        cards.append(f'<div class="bc-card bc-hab" style="border-top:3px solid {c.color}">'
                     f'<div class="name"><span class="sw" style="background:{c.color}"></span>{c.name}</div>'
                     f'{body}<div class="desc">{HABITAT_INFO[k]}</div></div>')
    return f'<div class="bc-grid3">{"".join(cards)}</div>'


def carbon_cards(report: dict) -> str:
    cb = report["carbon"]
    s, q = cb["total_stock_tCO2e"], cb["total_sequestration_tCO2e_per_yr"]
    v = cb["indicative_annual_value_usd"]
    items = [
        ("Carbon stored", T_CO2, fmt(s["mean"]), "tonnes of CO₂", "",
         f"Likely between {fmt(s['p05'])} and {fmt(s['p95'])}{tip(T_RANGE)}",
         f"The carbon already locked in the soil and plants. If these habitats were destroyed, much of it would "
         f"escape back into the air, roughly the yearly exhaust of {car_equiv(s['mean'])} cars."),
        ("Absorbed each year", "How much new carbon these habitats pull out of the air and bury in their soil every "
         "year, as long as they stay healthy.", fmt(q["mean"], 1), "tonnes of CO₂ per year",
         "",
         f"Likely between {fmt(q['p05'], 1)} and {fmt(q['p95'], 1)}{tip(T_RANGE)}",
         f"Like taking {car_equiv(q['mean'])} cars off the road, every year, for free."),
        ("Possible carbon credit value", "Companies buy carbon credits to offset their emissions. Voluntary-market "
         "prices for coastal ecosystem credits are roughly $15–40 per tonne of CO₂.", "$" + fmt(v["mid"]), "per year", "",
         f"Between ${fmt(v['low'])} and ${fmt(v['high'])} per year",
         "A rough idea of what protecting these habitats could earn, based on the yearly uptake only. Not a "
         "financial valuation."),
    ]
    html = "".join(
        f'<div class="bc-card bc-kpi"><div class="l">{t}{tip(ti)}</div><div class="v">{val}<small>{u}</small></div>{bar}'
        f'<div class="s">{rng}</div><div class="what">{what}</div></div>' for t, ti, val, u, bar, rng, what in items)
    return f'<div class="bc-grid3">{html}</div>'


def reliability_html(model: dict | None) -> str:
    if not model or not model.get("test"):
        return '<div class="bc-note">No accuracy test is attached to this model.</div>'
    t = model["test"]
    rows = ['<div class="bc-relrow bc-relhead"><span></span><span>Habitat</span><span>Verdict</span>'
            f'<span class="sc">Score{tip(T_SCORE)}</span></div>']
    for c in CLASSES:
        iou = t["iou"].get(c.key)
        label, cls = reliability(iou)
        score = "–" if iou is None else f"{iou:.2f}"
        rows.append(f'<div class="bc-relrow"><span class="sw" style="background:{c.color}"></span>'
                    f'<span class="n">{c.name}</span><span class="bc-rel {cls}">{label}</span>'
                    f'<span class="sc">{score}</span></div>')
    return ('<div class="bc-rels">' + "".join(rows) + '</div><div class="bc-note" style="margin-top:.6rem">'
            "Reliable = score 0.70 or higher · Fair = 0.40 to 0.70 · Unreliable = below 0.40. "
            "“Not tested” means the test areas contained none of that habitat.</div>")


def summary_html(meta: dict, report: dict) -> str:
    a = best_areas(report)
    tot = site_area_ha(report) or 1
    top = sorted(a.items(), key=lambda kv: -kv[1])[:2]
    bc = {k: a[k] for k in BLUE_CARBON_KEYS if a[k] >= 0.05}
    cb = report["carbon"]
    s, q = cb["total_stock_tCO2e"], cb["total_sequestration_tCO2e_per_yr"]
    model = meta.get("model") or {}
    t = model.get("test") or {}
    pts = [("The area", f"Mostly {CLS[top[0][0]].name.lower()} ({100 * top[0][1] / tot:.0f}%) and "
                        f"{CLS[top[1][0]].name.lower()} ({100 * top[1][1] / tot:.0f}%), across {tot / 100:,.1f} km².")]
    if bc:
        lead = max(bc, key=bc.get)
        pts.append(("Blue carbon habitat", f"{fmt(sum(bc.values()), 1)} hectares, mostly {CLS[lead].name.lower()} "
                                           f"({fmt(bc[lead], 1)} hectares). That is {100 * sum(bc.values()) / tot:.1f}% "
                                           "of the area."))
        pts.append(("Carbon", f"About {fmt(s['mean'])} tonnes of CO₂ stored, with roughly {fmt(q['mean'], 1)} tonnes "
                              "added every year."))
    else:
        pts.append(("Blue carbon habitat", "None detected, so no carbon is counted for this area."))
    if t:
        def names(pred):
            n = [CLS[k].name.lower() for k, v in t["iou"].items() if v is not None and k in CLS and pred(v)]
            return ", ".join(n[:-1]) + " and " + n[-1] if len(n) > 1 else (n[0] if n else "")

        good, fair, weak = names(lambda v: v >= 0.7), names(lambda v: 0.4 <= v < 0.7), names(lambda v: v < 0.4)
        parts = []
        if good:
            parts.append(f"{good.capitalize()} are mapped reliably")
        if fair:
            parts.append(f"{fair} {'are' if ' and ' in fair else 'is'} mapped fairly well but misses some patches")
        if weak:
            parts.append(f"{weak} detection is not yet reliable")
        conf = "; ".join(parts) or "Map accuracy has not been tested"
        pts.append(("How much to trust it", conf + "."))
    if model.get("pilot"):
        pts.append(("What's next", "This map comes from the pilot model, trained on hand-labelled data from this bay. "
                                  "The full BlueCarbon-AI model trains on 39 coastal sites on six continents so it "
                                  "works on coastlines it has never seen, including mangrove forests."))
    if a.get("freshwater", 0) >= 0.05:
        pts.insert(2 if len(pts) >= 2 else len(pts), ("Freshwater wetland",
                    f"{fmt(a['freshwater'], 1)} hectares of inland marsh or swamp. It stores carbon too, but it is not "
                    "tidal, so it is shown on the map and not counted as blue carbon."))
    per = report["carbon"]["classes"]
    measured = [CLS[k].name.lower() for k, v in per.items() if (v.get("soil") or {}).get("source") == "measured"]
    if measured:
        src = (f"Soil carbon for {' and '.join(measured)} comes from real soil cores measured near this site "
               "(Smithsonian Coastal Carbon Library); other values are IPCC global averages.")
    else:
        src = "Carbon values are IPCC global averages for each habitat, because no measured soil cores are nearby."
    pts.append(("Fine print", src + " Selling real carbon credits would require measurements at the site itself."))
    rows = "".join(f'<div class="bc-sumrow"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in pts)
    return f'<div class="bc-card bc-summary">{rows}</div>'


def in_image_html(report: dict, only_blue: bool = False) -> str:
    """Legend listing everything in the image with its area and share."""
    a = best_areas(report)
    tot = sum(a.values()) or 1
    rows = []
    for c in CLASSES:
        if only_blue and not c.blue_carbon:
            continue
        pct = 100 * a[c.key] / tot
        tag = '<span class="bc">blue carbon</span>' if c.blue_carbon else ""
        rows.append(f'<div class="row"><span class="sw" style="background:{c.color}"></span>'
                    f'<span class="n">{c.name}{tag}</span><span class="a">{fmt(a[c.key], 1)} ha</span>'
                    f'<span class="p">{pct:.1f}%</span><span></span>'
                    f'<span class="bar"><i style="width:{max(pct, 0.6 if a[c.key] > 0 else 0):.1f}%;'
                    f'background:{c.color}"></i></span></div>')
    return (f'<div class="bc-legend">{"".join(rows)}</div><div class="bc-note" style="margin-top:.5rem">'
            f"ha = hectares{tip(T_HA)} · % = share of the mapped area</div>")


def map_legend(report: dict, only_blue: bool) -> str:
    a = best_areas(report)
    items = [c for c in CLASSES if a[c.key] >= 0.05 and (c.blue_carbon or not only_blue)]
    return '<div class="bc-maplegend">' + "".join(
        f'<span><i style="background:{c.color}"></i>{c.name}</span>' for c in items) + "</div>"


# ----------------------------------------------------------------------------- chrome
st.markdown(
    f'<div class="bc-top"><div class="bc-brand"><div class="bc-word">BlueCarbon<span>-AI</span></div>'
    f'</div><div class="bc-links"><a href="{REPO}" target="_blank">GitHub</a>'
    f'<a href="{REPO}/blob/main/docs/METHODOLOGY.md" target="_blank">How it works</a></div></div>'
    '<div class="bc-hero"><div>'
    "<h1>Finding the coastal ecosystems that fight climate change</h1>"
    "<p>Mangrove forests, salt marshes and seagrass meadows pull carbon dioxide out of the air and lock it away in "
    "their soils, sometimes for thousands of years. Scientists call this <b>blue carbon</b>. These habitats are "
    "disappearing fast, and you can't protect what you haven't mapped.</p>"
    "<p style='margin-top:.7rem'><b>BlueCarbon-AI</b> uses free satellite images and a deep-learning model to find "
    "these habitats automatically, measure how much area they cover, and estimate how much carbon they hold.</p></div>"
    '<div class="bc-how">'
    f'<div class="bc-howstep"><span class="num">1</span><div><b>Satellite image</b>'
    f'<small>A cloud-free photo of the coast from the Sentinel-2 satellites{tip(T_S2)}</small></div></div>'
    '<div class="bc-howstep"><span class="num">2</span><div><b>AI habitat map</b>'
    "<small>An AI model labels every 10 × 10 m patch as water, marsh, mangrove, seagrass or land</small></div></div>"
    f'<div class="bc-howstep"><span class="num">3</span><div><b>Carbon estimate</b>'
    f'<small>Area × published carbon values per habitat, with an honest uncertainty range{tip(T_TIER1)}</small></div></div>'
    "</div></div>",
    unsafe_allow_html=True,
)

tab_explore, tab_analyze, tab_method = st.tabs(["Explore sites", "Analyze an area", "Methodology"])

# ----------------------------------------------------------------------------- explore
VIEWS = {"Habitats": "classes", "Blue carbon only": "bluecarbon", "Satellite": None, "False color": "falsecolor"}
HINTS = {
    "Habitats": "Each color shows what the AI thinks covers that patch of ground or water.",
    "Blue carbon only": "Only mangrove, salt marsh and seagrass: the habitats counted in the carbon numbers.",
    "Satellite": "The cloud-free satellite photo the AI analysed, in natural color.",
    "False color": "An infrared view where healthy plants glow red. Scientists use it to spot vegetation that is hard to see in normal color.",
}


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="bc-h2">{title}</div>' + (f'<div class="bc-h2sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


CHANGE_VIEWS = {"After": "classes_t1", "Before": "classes_t0", "What changed": "change", "Satellite": None,
                "False color": "falsecolor_t1"}


def change_hints(meta: dict) -> dict:
    p = meta.get("periods") or {}
    t0, t1 = p.get("t0", "the earlier date"), p.get("t1", "the later date")
    return {
        "After": f"Habitats in {t1}.",
        "Before": f"Habitats in {t0}.",
        "What changed": f"Green = blue carbon habitat that appeared between {t0} and {t1}. Red = habitat that was lost.",
        "Satellite": f"The cloud-free satellite photo from {t1}, in natural color.",
        "False color": HINTS["False color"],
    }


def render_results(meta: dict, report: dict, map_fn, side_header: str = "", views: dict | None = None,
                   hints: dict | None = None, after_map=None) -> None:
    views, hints = views or VIEWS, hints or HINTS
    default = next(iter(views))
    c1, c2 = st.columns([3, 1])
    view = c1.segmented_control("Map layer", list(views), default=default, key=f"view_{meta['title']}") or default
    opacity = c2.slider("Color overlay strength", 0.0, 1.0, 0.8, 0.05, key=f"op_{meta['title']}")
    left, right = st.columns([0.64, 0.36], gap="large")
    with left:
        map_fn(view, opacity)
        legend = ""
        if views[view] == "change":
            legend = ('<div class="bc-maplegend"><span><i style="background:#22c55e"></i>Blue carbon gained</span>'
                      '<span><i style="background:#ef4444"></i>Blue carbon lost</span></div>')
        st.markdown(legend + f'<div class="bc-hint">{hints[view]}</div>', unsafe_allow_html=True)
    with right:
        if side_header:
            st.markdown(side_header, unsafe_allow_html=True)
        st.markdown(f'<div class="bc-section">In this image{tip("Area of each class the AI found, corrected for its known mistakes. " + T_ADJ)}</div>',
                    unsafe_allow_html=True)
        st.markdown(in_image_html(report, view == "Blue carbon only"), unsafe_allow_html=True)

    if after_map:
        after_map()

    section("Blue carbon habitats", "The three coastal ecosystems that store large amounts of carbon, and how much of each the AI found here.")
    st.markdown(habitat_cards(report), unsafe_allow_html=True)

    section("What that means for the climate", "Estimated from the habitat areas above and published carbon values per hectare." + tip(T_TIER1))
    st.markdown(carbon_cards(report), unsafe_allow_html=True)

    section("How much can you trust this map?", "We tested the AI on areas it never saw during training and compared its map with trusted reference maps.")
    r1, r2 = st.columns([0.55, 0.45], gap="large")
    with r1:
        st.markdown(reliability_html(meta.get("model")), unsafe_allow_html=True)
    with r2:
        with st.expander("Technical details"):
            model_panel(meta.get("model"))
            st.markdown('<div class="bc-section">Carbon by habitat</div>', unsafe_allow_html=True)
            st.markdown(carbon_table(report), unsafe_allow_html=True)
            st.markdown(f'<div class="bc-note" style="margin-top:.5rem">{report["carbon"]["method"]}. Areas are '
                        f"accuracy-corrected{tip(T_ADJ)}.</div>", unsafe_allow_html=True)

    section("Summary", "What we found at this site, in plain words.")
    st.markdown(f'<div class="bc-card bc-found"><p>{glance_html(meta, report)}</p></div>', unsafe_allow_html=True)
    st.markdown(summary_html(meta, report), unsafe_allow_html=True)


def change_section(meta: dict) -> None:
    p = meta.get("periods") or {}
    t0, t1 = p.get("t0", "Before"), p.get("t1", "After")
    a0 = best_areas(meta["t0"])
    a1 = best_areas(meta["t1"])
    rows, sentences = [], []
    for k in BLUE_CARBON_KEYS:
        d = a1[k] - a0[k]
        if a0[k] < 0.05 and a1[k] < 0.05:
            continue
        pct = f"{100 * d / a0[k]:+.0f}%" if a0[k] >= 0.05 else "new"
        cls = "bc-up" if d > 0 else "bc-down" if d < 0 else ""
        rows.append(f'<tr><td><span class="sw" style="display:inline-block;width:10px;height:10px;border-radius:3px;'
                    f'background:{CLS[k].color};margin-right:8px"></span>{CLS[k].name}</td>'
                    f'<td class="num">{fmt(a0[k], 1)}</td><td class="num">{fmt(a1[k], 1)}</td>'
                    f'<td class="num {cls}">{"+" if d > 0 else ""}{fmt(d, 1)} ha ({pct})</td></tr>')
        if abs(d) >= 0.5:
            sentences.append(f"{CLS[k].name.lower()} {'grew' if d > 0 else 'shrank'} by {fmt(abs(d), 1)} hectares")
    s0 = meta["t0"]["carbon"]["total_stock_tCO2e"]["mean"]
    s1 = meta["t1"]["carbon"]["total_stock_tCO2e"]["mean"]
    net = s1 - s0
    lead = (f"Between {t0} and {t1}, " + ", ".join(sentences) + ". " if sentences
            else f"Blue carbon habitat stayed roughly the same between {t0} and {t1}. ")
    lead += (f"That {'adds' if net >= 0 else 'removes'} about <b>{fmt(abs(net))} tonnes of CO₂</b> "
             f"{'to' if net >= 0 else 'from'} the carbon stored at this site.")
    section("What changed", f"Blue carbon habitat in {t0} compared with {t1}.")
    st.markdown(f'<div class="bc-card bc-found"><p>{lead}</p></div>', unsafe_allow_html=True)
    if rows:
        st.markdown('<table class="bc-table"><thead><tr><th>Habitat</th>'
                    f'<th style="text-align:right">{t0} (ha)</th><th style="text-align:right">{t1} (ha)</th>'
                    '<th style="text-align:right">Change</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>",
                    unsafe_allow_html=True)
    st.markdown('<div class="bc-note" style="margin-top:.5rem">Small changes can come from differences in tide, '
                "season or image quality between the two dates rather than real habitat change.</div>",
                unsafe_allow_html=True)


def site_metas() -> dict[str, dict]:
    return _site_metas(_demo_signature())


@st.cache_data
def _site_metas(sig: str) -> dict[str, dict]:
    return {t: json.loads((d / "meta.json").read_text()) for t, d in list_sites().items()}


def site_status(meta: dict) -> tuple[str, str]:
    if meta.get("kind") == "change":
        return "Change over time", "bc-st-change"
    if meta.get("held_out"):
        return "Never seen in training", "bc-st-held"
    if meta.get("held_out") is False:
        return "Training site", "bc-st-train"
    return "", ""


def sites_overview(metas: dict[str, dict]) -> None:
    m = folium.Map(location=[20, -30], zoom_start=2, tiles=None, min_zoom=1, world_copy_jump=True,
                   scrollWheelZoom=False)
    folium.TileLayer(ESRI, attr=ESRI_ATTR).add_to(m)
    for t, meta in metas.items():
        (s, w), (n, e) = meta["bounds"]
        label, _ = site_status(meta)
        color = "#14a3a0" if meta.get("held_out") or meta.get("kind") == "change" else "#f5f7f8"
        folium.CircleMarker([(s + n) / 2, (w + e) / 2], radius=7, color="#083744", weight=2, fill=True,
                            fill_color=color, fill_opacity=1,
                            tooltip=f"{t}" + (f" · {label}" if label else "")).add_to(m)
    st_folium(m, height=300, use_container_width=True, returned_objects=[], key="overview")
    st.markdown('<div class="bc-maplegend"><span><i style="background:#14a3a0;border:2px solid #083744"></i>'
                'Never seen in training</span><span><i style="background:#f5f7f8;border:2px solid #083744"></i>'
                "Training site</span></div>", unsafe_allow_html=True)


with tab_explore:
    metas = site_metas()
    sites = list_sites()
    if not sites:
        st.warning("No demo data found in demo_data/.")
    else:
        order = sorted(metas, key=lambda t: (0 if metas[t].get("kind") == "change" else
                                              1 if metas[t].get("held_out") else 2 if "held_out" not in metas[t] else 3, t))
        if len(metas) > 1:
            section(f"{len(metas)} coastal sites", "Pick a site below. Teal sites were never shown to the AI during "
                                                   "training, so they are the fairest test of how well it works.")
            sites_overview(metas)
        c1, _ = st.columns([1.5, 1.5], gap="large")

        def label(t):
            lab, _ = site_status(metas[t])
            return f"{t}  ·  {lab}" if lab else t

        title = c1.selectbox("Site", order, format_func=label)
        d = sites[title]
        meta = metas[title]
        report = meta["report"] if meta["kind"] == "single" else meta["t1"]
        where = " · ".join(x for x in [meta.get("region"), meta.get("period")] if x)
        lab, cls = site_status(meta)
        badge = f'<span class="bc-status {cls}">{lab}</span>' if lab else ""
        side = (f'<div class="bc-site">{badge}<h3>{meta["title"]}</h3><div class="meta">{where}</div>'
                f'<p>{meta.get("description", "")}</p></div>')

        if meta["kind"] == "change":
            def demo_map(view, opacity):
                m = make_map(meta["bounds"])
                key = CHANGE_VIEWS[view]
                if view == "False color":
                    overlay(m, png_uri(d / "falsecolor_t1.png"), meta["bounds"], 1.0)
                else:
                    base = "rgb_t0.png" if view == "Before" else "rgb_t1.png"
                    overlay(m, png_uri(d / base), meta["bounds"], 1.0)
                    if key:
                        overlay(m, png_uri(d / f"{key}.png"), meta["bounds"], opacity)
                outline(m, meta["bounds"])
                st_folium(m, height=560, use_container_width=True, returned_objects=[],
                          key=f"map_{title}_{view}_{opacity}")

            render_results(meta, report, demo_map, side, CHANGE_VIEWS, change_hints(meta),
                           after_map=lambda: change_section(meta))
        else:
            def demo_map(view, opacity):
                m = make_map(meta["bounds"])
                if view == "False color":
                    overlay(m, png_uri(d / "falsecolor.png"), meta["bounds"], 1.0)
                else:
                    overlay(m, png_uri(d / "rgb.png"), meta["bounds"], 1.0)
                    if VIEWS[view]:
                        overlay(m, png_uri(d / f"{VIEWS[view]}.png"), meta["bounds"], opacity)
                outline(m, meta["bounds"])
                st_folium(m, height=560, use_container_width=True, returned_objects=[],
                          key=f"map_{title}_{view}_{opacity}")

            render_results(meta, report, demo_map, side)


# ----------------------------------------------------------------------------- analyze
def secret(name: str) -> str | None:
    try:
        v = st.secrets.get(name)
    except Exception:
        v = None
    v = v or os.environ.get(name)
    if isinstance(v, dict) or hasattr(v, "to_dict"):
        v = json.dumps(dict(v))
    return v


def _model_signature() -> str:
    """Changes whenever the model file or the prediction code changes, so a cached model built by
    older code is never reused after a redeploy."""
    import bluecarbon.predictors as bp

    parts = [str(DEFAULT_MODEL), secret("MODEL_URL") or ""]
    for f in (Path(DEFAULT_MODEL), Path(bp.__file__), Path(bp.__file__).with_name("features.py")):
        try:
            parts.append(f"{f.stat().st_mtime_ns}:{f.stat().st_size}")
        except OSError:
            pass
    return "|".join(parts)


def get_model():
    return _load_model(_model_signature())


@st.cache_resource(max_entries=1)
def _load_model(signature: str):
    from bluecarbon.predictors import load_predictor

    url, path = secret("MODEL_URL"), DEFAULT_MODEL
    if url:
        import urllib.request

        path = Path(tempfile.gettempdir()) / "bluecarbon_model.pt"
        if not path.exists():
            urllib.request.urlretrieve(url, path)
    return load_predictor(path)


@st.cache_resource
def init_gee(sa_json: str):
    from bluecarbon import gee

    gee.init(secret("GEE_PROJECT") or json.loads(sa_json).get("project_id") or CFG.project, sa_json)
    return True


with tab_analyze:
    sa = secret("GEE_SERVICE_ACCOUNT")
    if not sa:
        st.markdown(
            """
<div class="bc-card" style="max-width:820px;margin-top:.6rem">
  <div class="bc-eyebrow">On-demand analysis</div>
  <h3 style="margin:.35rem 0 .4rem">Map any coastline in about a minute</h3>
  <p style="color:var(--ink-2);line-height:1.6;margin:0 0 .9rem">Draw a box anywhere on Earth, pick a season, and
  BlueCarbon-AI pulls a fresh cloud-free Sentinel-2 composite from Google Earth Engine, runs the model and returns a habitat
  map, carbon report and downloadable GeoTIFF.</p>
  <div class="bc-note">This deployment hasn't been connected to Earth Engine yet, so the live pipeline is switched off.
  Everything in <b>Explore sites</b> works without it. The same analysis runs locally with
  <code>bluecarbon scene --bbox … --start … --end … -m model.pt</code>.</div>
</div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="bc-note" style="margin:.4rem 0 .8rem">Draw a rectangle over a coastline '
                    f"(up to {MAX_AREA_KM2} km²), choose the season, then run.</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1, 1, 1])
        start = c1.date_input("Start", pd.Timestamp("2024-05-01"))
        end = c2.date_input("End", pd.Timestamp("2024-09-30"))
        m = folium.Map(location=[32.78, -117.22], zoom_start=12, tiles=None)
        folium.TileLayer(ESRI, attr=ESRI_ATTR, name="Satellite").add_to(m)
        Draw(draw_options={"polyline": False, "polygon": False, "circle": False, "marker": False,
                           "circlemarker": False, "rectangle": {"shapeOptions": {"color": "#14a3a0"}}},
             edit_options={"edit": False}).add_to(m)
        out = st_folium(m, height=480, use_container_width=True, key="draw")
        feat = (out or {}).get("last_active_drawing")
        c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        run = c3.button("Run analysis", type="primary", disabled=feat is None, width="stretch")
        if run and feat:
            import rasterio
            from rasterio.warp import Resampling

            from bluecarbon import gee
            from bluecarbon.demo import _to_mercator
            from bluecarbon.features import S2_BANDS
            from bluecarbon.report import scene_report
            from bluecarbon.viz import class_rgba, true_color

            coords = np.array(feat["geometry"]["coordinates"][0])
            bbox = [coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()]
            km2 = ((bbox[2] - bbox[0]) * 111.32 * math.cos(math.radians((bbox[1] + bbox[3]) / 2))
                   * (bbox[3] - bbox[1]) * 110.57)
            if km2 > MAX_AREA_KM2:
                st.error(f"That box is {km2:,.0f} km². Please draw one under {MAX_AREA_KM2} km².")
                st.stop()
            init_gee(sa)
            predictor = get_model()
            ck = predictor.meta
            with st.status("Running pipeline…", expanded=True) as status:
                tmp = Path(tempfile.mkdtemp())
                region = gee.bbox_geometry(bbox)
                n = gee.image_count(region, str(start), str(end), CFG)
                st.write(f"{n} Sentinel-2 scenes found. Building a cloud-masked median composite…")
                if n == 0:
                    status.update(label="No imagery for that period", state="error")
                    st.stop()
                bar = st.progress(0.0)
                gee.download(gee.s2_composite(region, str(start), str(end), CFG), bbox, tmp / "image.tif", CFG,
                             "uint16", 0, S2_BANDS, progress=bar.progress)
                anc = None
                if getattr(predictor, "needs_ancillary", False):
                    st.write("Fetching elevation, tidal and clear-water layers…")
                    gee.download_ancillary(bbox, str(start), str(end), tmp / "ancillary.tif", CFG)
                    with rasterio.open(tmp / "ancillary.tif") as a:
                        anc = a.read()
                st.write("Segmenting habitats…")
                with rasterio.open(tmp / "image.tif") as src:
                    bands, prof = src.read(), src.profile
                cls, _ = predictor.predict(bands, CFG.predict.tile, CFG.predict.overlap, anc=anc)
                from bluecarbon.priors import apply_to_classes

                cls = apply_to_classes(cls, (bbox[1] + bbox[3]) / 2)
                prof.update(count=1, dtype="uint8", nodata=255)
                with rasterio.open(tmp / "pred.tif", "w", **prof) as dst:
                    dst.write(cls, 1)
                rep = scene_report(tmp / "pred.tif", CFG.carbon, ck["metrics"].get("test_confusion"))
                status.update(label="Analysis complete", state="complete", expanded=False)
            mb, bnds = _to_mercator(tmp / "image.tif", list(range(1, 11)), Resampling.bilinear, 0)
            mc, _ = _to_mercator(tmp / "pred.tif", [1], Resampling.nearest, 255)
            from bluecarbon.schema import KEY_TO_ID

            layers = {
                "classes": rgba_uri(class_rgba(mc[0], 200)),
                "bluecarbon": rgba_uri(class_rgba(mc[0], 220, [KEY_TO_ID[k] for k in BLUE_CARBON_KEYS])),
                "falsecolor": rgba_uri(true_color(mb, rgb_bands=("B8", "B4", "B3"), gamma=1.0)),
            }
            live_meta = {"title": "Your area", "model": {**ck.get("extra", {}), "arch": ck["arch"], "kind": ck["kind"],
                                                         "encoder": ck["encoder"], "test": ck["metrics"].get("test")}}
            st.session_state["live"] = {"meta": live_meta, "report": rep, "bounds": bnds, "layers": layers,
                                        "rgb": rgba_uri(true_color(mb)), "tif": (tmp / "pred.tif").read_bytes()}

        live = st.session_state.get("live")
        if live:
            def live_map(view, opacity):
                m2 = make_map(live["bounds"])
                if view == "False color":
                    overlay(m2, live["layers"]["falsecolor"], live["bounds"])
                else:
                    overlay(m2, live["rgb"], live["bounds"])
                    if VIEWS[view]:
                        overlay(m2, live["layers"][VIEWS[view]], live["bounds"], opacity)
                st_folium(m2, height=540, use_container_width=True, returned_objects=[], key=f"live_{view}_{opacity}")

            render_results(live["meta"], live["report"], live_map)
            d1, d2, _ = st.columns([1, 1, 2])
            d1.download_button("Download habitat GeoTIFF", live["tif"], "bluecarbon_habitats.tif", width="stretch")
            d2.download_button("Download report (JSON)", json.dumps(live["report"], indent=2), "bluecarbon_report.json",
                               width="stretch")

# ----------------------------------------------------------------------------- methodology
with tab_method:
    steps = [
        ("01", "Acquire", "Sentinel-2 L2A surface reflectance, masked with Cloud Score+ and reduced to a seasonal median."),
        ("02", "Label", "Reference labels fused from ESA WorldCover, Murray tidal flats and the Allen Coral Atlas."),
        ("03", "Learn", "U-Net with a ResNet encoder on 10 bands, 4 indices and 3 context layers, trained with Dice + CE loss."),
        ("04", "Map", "Overlapping tiles blended with a smooth window and flip test-time augmentation."),
        ("05", "Account", "Error-adjusted areas × measured local soil carbon (or IPCC Tier 1), with Monte Carlo 90% intervals."),
    ]
    st.markdown('<div class="bc-steps">' + "".join(
        f'<div class="bc-step"><div class="k">{k}</div><h4>{t}</h4><p>{p}</p></div>' for k, t, p in steps) + "</div>",
        unsafe_allow_html=True)

    cc = CFG.carbon.classes
    carbon_rows = "".join(
        f'<tr><td>{CLS[k].name}</td><td class="num">{v.soil[1]:.0f} <span class="bc-note">({v.soil[0]:.0f}–{v.soil[2]:.0f})</span></td>'
        f'<td class="num">{v.biomass[1]:.0f} <span class="bc-note">({v.biomass[0]:g}–{v.biomass[2]:g})</span></td>'
        f'<td class="num">{v.accumulation[1]:.2f} <span class="bc-note">({v.accumulation[0]:g}–{v.accumulation[2]:g})</span></td></tr>'
        for k, v in cc.items())

    st.markdown(
        f"""
<div class="bc-doc">
<h3>Imagery</h3>
<p>Each scene is a per-pixel median of every Sentinel-2 L2A acquisition in the chosen season
(<code>COPERNICUS/S2_SR_HARMONIZED</code>) after removing cloud and shadow with Google's Cloud Score+
(<code>cs_cdf ≥ 0.6</code>). Ten bands (B2–B8A, B11, B12) are exported at 10&nbsp;m in the local UTM zone,
so every pixel has a true ground area. The model also receives four indices: NDVI (vegetation), NDWI and
MNDWI (water), and NDMI (canopy moisture, which separates mangrove from dry upland).</p>
<p><b>Clear-water image.</b> Seagrass is only visible where the seafloor shows through, and a seasonal median blends
clear days with murky, glinty ones. So each scene also gets a clear-water image: for every pixel, the single
cloud-free observation with the least near-infrared reflectance (least sun glint, haze and white water). Its blue,
green, red and near-infrared bands, plus two band ratios that are largely insensitive to water depth, are inputs.</p>
<p><b>Context layers.</b> Some habitats look identical from space: tidal salt marsh and inland freshwater marsh,
or dense salt marsh and young mangrove. So the model also gets two layers that describe <i>where</i> a pixel is:
elevation (NASADEM) and the probability that the tide reaches it (Murray et al., 2022). Both also help build the
reference labels, so part of what the model learns from them is that labelling rule. That is why the scores on
held-out estuaries, not the training fit, are the numbers that count. Latitude is deliberately <i>not</i> an input:
an earlier model used it as a shortcut and missed mangroves on unseen coasts. Instead, mangroves are limited to
their known latitude range (39°S–32.5°N) by an explicit rule, because frost kills them.</p>

<h3>Reference labels</h3>
<p>Training labels come from independent, peer-reviewed global products, not from thresholds on the
model's own input bands. Pixels within one pixel of a class boundary are excluded, because edges are where these
products are least reliable. Local survey polygons (for example eelgrass surveys) can override any source.</p>
<table class="bc-table"><thead><tr><th>Class</th><th>Source</th><th>Rule</th></tr></thead><tbody>
<tr><td>Open water · Other land</td><td>ESA WorldCover 2021 (10 m)</td><td>Classes 80 · 10–60, 100</td></tr>
<tr><td>Mangrove</td><td>ESA WorldCover 2021</td><td>Class 95</td></tr>
<tr><td>Salt marsh</td><td>ESA WorldCover + GWL_FCS30 wetland map (Zhang et al., 2023)</td><td>Herbaceous, grass or shrub cover that GWL_FCS30 classes as salt marsh. Tidal-zone vegetation it calls non-wetland is left unlabelled</td></tr>
<tr><td>Freshwater wetland</td><td>ESA WorldCover + GWL_FCS30 + Murray et al. tidal wetlands</td><td>Herbaceous wetland outside the tidal zone, or classed as swamp or marsh. Mapped, but not counted as blue carbon</td></tr>
<tr><td>Tidal flat</td><td>Murray et al., global intertidal</td><td>Tidal flat classification</td></tr>
<tr><td>Seagrass</td><td>Allen Coral Atlas benthic map + FWC Florida statewide seagrass + Moreton Bay 2015 (Seamap Australia)</td><td>Atlas seagrass class (tropics) plus survey polygons from 2010 on, burned in over water only. Where seagrass exists but no map covers it (e.g. Shark Bay), water outside the Atlas footprint is left unlabelled rather than taught as open water</td></tr>
</tbody></table>

<h3>Model and evaluation</h3>
<p>A U-Net with a ResNet-34 encoder (<code>segmentation-models-pytorch</code>) and a 22-channel input stem, trained
with cross-entropy plus Dice loss and square-root inverse-frequency class weights, AdamW with a one-cycle
schedule, mixed precision and early stopping on validation mIoU.</p>
<p>Evaluation is built so the model can't score well by memorizing. Chips never overlap, whole 5&nbsp;km blocks are
assigned to a single split, and five complete estuaries (Mission Bay, Plum Island, Moreton Bay, Shoalwater Bay, Tampa Bay) are never seen in
training and are scored separately. The
headline metrics are per-class IoU and F1. Overall accuracy is reported but not emphasized: a scene that is 70% water
can score 90% accuracy while missing every marsh pixel.</p>

<h3>Area and carbon accounting</h3>
<p>Raw pixel counts are biased toward whatever the model over-predicts. Areas are therefore corrected with the
stratified estimator of Olofsson et al. (2014), using the held-out confusion matrix. Carbon is then estimated per
habitat:</p>
<div class="bc-formula">stock (tCO₂e) = area (ha) × [ soil C to 1 m + living biomass C ] (tC/ha) × 44/12<br>
sequestration (tCO₂e/yr) = area (ha) × soil C accumulation (tC/ha/yr) × 44/12</div>
<p>Each coefficient is drawn from a triangular distribution over its published range, jointly with the area
uncertainty (5,000 Monte Carlo draws), and results are reported as a mean with a 90% interval. Tier 1 defaults
(IPCC 2013 Wetlands Supplement), used where no measured soil cores are available:</p>
<table class="bc-table"><thead><tr><th>Habitat</th><th style="text-align:right">Soil C, tC/ha</th>
<th style="text-align:right">Biomass C, tC/ha</th><th style="text-align:right">Accumulation, tC/ha/yr</th></tr></thead>
<tbody>{carbon_rows}</tbody></table>
<p><b>Measured soil carbon.</b> Where possible, the soil value is replaced with real measurements from the
Smithsonian Coastal Carbon Library (open data from hundreds of studies). For every soil core, carbon to 1&nbsp;m is
dry bulk density × organic carbon fraction, averaged over the sampled depth (at least 30&nbsp;cm) and scaled to
1&nbsp;m. Only measurements the study confirms are <i>organic</i> carbon are used; total-carbon values, which count
limestone carbonate in tropical seafloor, are replaced by an estimate from organic matter (Craft et al., 1991). A
site uses the cores of that habitat within 100&nbsp;km (300&nbsp;km if needed) when there are at least 8; its range
reflects the number of independent studies, not cores. Biomass and burial rates stay at IPCC values.</p>
<p>The indicative credit value is based on annual sequestration at $15–40 per tCO₂e. Standing stock is not creditable
on its own.</p>

<h3>Limitations</h3>
<ul>
<li>Carbon values (measured nearby cores or IPCC global averages) are suited to screening and prioritization,
not to issuing credits, which requires measurements at the project site.</li>
<li>Seagrass is learned from the Allen Coral Atlas and official surveys (Florida, Moreton Bay). The model is now
conservative: it rarely calls open water seagrass, but misses about half of seagrass in murky or deep water, and
does not yet detect it in Moreton Bay or Shoalwater Bay. Treat seagrass areas as a lower bound.</li>
<li>Tides change what is exposed in intertidal zones, and a median composite averages across tidal states.</li>
<li>Reference products carry their own errors, which the model partly learns. The confidence intervals treat pixels
as independent samples, so they understate the true uncertainty.</li>
</ul>

<h3>References</h3>
<ul class="bc-refs">
<li>IPCC (2014). <i>2013 Supplement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Wetlands</i>, Chapter 4.</li>
<li>Olofsson, P. et al. (2014). Good practices for estimating area and assessing accuracy of land change. <i>Remote Sensing of Environment</i> 148.</li>
<li>Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200.</li>
<li>Murray, N. J. et al. (2019). The global distribution and trajectory of tidal flats. <i>Nature</i> 565.</li>
<li>Coastal Carbon Network (2023). Coastal Carbon Library, v1.7.0. Smithsonian Environmental Research Center. doi:10.25573/serc.21565671.</li>
<li>Craft, C. B. et al. (1991). Loss on ignition and Kjeldahl digestion for estimating organic carbon and total nitrogen in estuarine marsh soils. <i>Soil Sci. Soc. Am. J.</i> 55.</li>
<li>Zhang, X. et al. (2023). GWL_FCS30: a global 30 m wetland map with a fine classification system. <i>Earth System Science Data</i> 15.</li>
<li>Allen Coral Atlas (2022). Imagery, maps and monitoring of the world's tropical coral reefs.</li>
<li>Pasquarella, V. et al. (2023). Cloud Score+: comprehensive cloud and cloud-shadow detection for Sentinel-2.</li>
</ul>
</div>""",
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="bc-footer"><span>BlueCarbon-AI · Built by Yanick Sanchez</span>'
    f'<span><a href="{REPO}" target="_blank">Source on GitHub</a> · MIT License</span></div>',
    unsafe_allow_html=True,
)
