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

_CURRENT = ROOT / "models" / "current.txt"
DEFAULT_MODEL = ROOT / "models" / (_CURRENT.read_text().strip() if _CURRENT.exists()
                                   else "pilot_spectral_mission_bay_2018.json")
SITE = "https://yanicksanchez14-creator.github.io/bluecarbon-ai/"
MAX_AREA_KM2 = 60
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTR = "Imagery © Esri, Maxar, Earthstar Geographics"

CFG = load_config(ROOT / "configs" / "default.yaml")
CLS = {c.key: c for c in CLASSES}

st.set_page_config(page_title="BlueCarbon-AI", page_icon=str(ROOT / "app" / "assets" / "blank.png"), layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------- design system
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600&family=Instrument+Serif&display=swap');
:root{
  --ink:#10262c; --ink-2:#3a4c52; --muted:#66777c; --line:#dfe6e5; --line-2:#ecf1f0;
  --bg:#f3f6f5; --card:#ffffff; --brand:#1f5f6b; --brand-ink:#123f48; --accent:#2a8a8c;
  --accent-soft:#e5f1f0; --sand:#c4ab7c; --warn-bg:#fff6e8; --warn-ink:#8a5a00; --radius:10px;
  --serif:'Instrument Serif', Georgia, serif;
}
html, body, [class*="css"], .stApp, .stMarkdown, button, input, select, textarea {
  font-family:'Geist', system-ui, -apple-system, 'Segoe UI', sans-serif !important; }
.bc-hero h1, .bc-h2, .bc-doc h3, .bc-site h3, .bc-page h2, .bc-page h3 {font-family:var(--serif) !important;
  font-weight:400; letter-spacing:-.005em;}
:focus-visible{outline:2px solid var(--accent) !important; outline-offset:2px;}
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
.bc-relhead{font-size:.86rem; color:var(--ink-2); font-weight:600;}
.bc-sumrow{display:grid; grid-template-columns:190px 1fr; gap:1rem; padding:.7rem 0; border-bottom:1px solid var(--line-2);}
.bc-sumrow:last-child{border-bottom:none;}
.bc-sumrow .k{font-weight:600; font-size:.9rem; color:var(--ink);}
.bc-sumrow .v{font-size:.94rem; color:var(--ink-2); line-height:1.6;}
.bc-tag{font-size:.72rem; font-weight:600; color:var(--brand); background:var(--accent-soft);
  border:1px solid #cfe9e5; padding:2px 8px; border-radius:999px; margin-left:.35rem;}
.bc-links a{color:var(--ink-2); text-decoration:none; font-size:.88rem; font-weight:500; margin-left:1.4rem;}
.bc-links a:hover{color:var(--brand);}

/* hero */
.bc-hero{display:grid; grid-template-columns:1.05fr 1fr; gap:3rem; align-items:center; margin:.6rem 0 1.4rem;}
.bc-hero h1{font-size:2.6rem; line-height:1.12; margin:0 0 1rem; padding:0; max-width:18ch; color:var(--ink);}
.bc-hero p{color:var(--ink-2); font-size:1.05rem; line-height:1.65; margin:0 0 .8rem; max-width:58ch;}
.bc-plate{margin:0; background:var(--card); border:1px solid var(--line); border-radius:4px; padding:10px 10px 0;}
.bc-plate img{display:block; width:100%; border-radius:2px;}
.bc-plate figcaption{font-size:.82rem; line-height:1.5; color:var(--ink-2); padding:.6rem .15rem .75rem;
  border-top:3px solid var(--sand); margin-top:10px;}
.bc-plate figcaption b{color:var(--ink);}
.bc-proof{display:flex; flex-wrap:wrap; gap:.4rem 2.2rem; padding:1rem 0 1.1rem; margin:0 0 1.2rem;
  border-top:1px solid var(--line); border-bottom:1px solid var(--line);}
.bc-proof div{font-size:.88rem; color:var(--muted);}
.bc-proof b{display:block; font-family:var(--serif); font-size:1.35rem; font-weight:600; color:var(--ink);
  font-variant-numeric:tabular-nums;}
.bc-facts{display:flex; gap:.6rem; flex-wrap:wrap;}
.bc-fact{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:.6rem .85rem; min-width:118px;}
.bc-fact b{display:block; font-size:1.05rem; color:var(--ink);}
.bc-fact small{color:var(--muted); font-size:.74rem;}

/* screening, about */
.bc-page{max-width:900px;}
.bc-page h2{font-size:1.7rem; margin:.8rem 0 .5rem; color:var(--ink);}
.bc-page h3{font-size:1.2rem; margin:1.8rem 0 .5rem; color:var(--ink);}
.bc-page p, .bc-page li{color:var(--ink-2); font-size:.98rem; line-height:1.7; max-width:68ch;}
.bc-contents{display:grid; grid-template-columns:1fr 1fr; gap:0 2rem; margin:.4rem 0 0 !important; padding:0 !important; list-style:none;}
.bc-contents li{margin:0 !important;}
.bc-contents li{padding:.7rem 0; border-top:1px solid var(--line); max-width:none;}
.bc-contents b{display:block; color:var(--ink); font-size:.96rem;}
.bc-contents span{font-size:.88rem; color:var(--muted); line-height:1.55; display:block;}
.bc-who{display:grid; grid-template-columns:repeat(2,1fr); gap:1rem 2rem; margin-top:.4rem;}
.bc-who div{border-left:3px solid var(--accent); padding:.1rem 0 .1rem .9rem;}
.bc-who b{display:block; color:var(--ink);}
.bc-who span{display:block; font-size:.9rem; color:var(--ink-2); line-height:1.5 !important; margin-top:.15rem;}
.bc-status-box{background:var(--accent-soft); border:1px solid #cfe4e2; border-radius:var(--radius); padding:1rem 1.2rem;
  margin:1.6rem 0 .4rem; color:var(--brand-ink); font-size:.94rem; line-height:1.6; max-width:68ch;}
.bc-disclaimer{font-size:.8rem; color:var(--muted); line-height:1.55; max-width:80ch;}

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
.bc-section{font-size:.9rem; font-weight:600; color:var(--ink-2); margin:1.4rem 0 .6rem;}
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
.bc-step .k{font-family:var(--serif); font-size:1.1rem; color:var(--accent); font-weight:600;}
.bc-step h4{margin:.3rem 0 .35rem; font-size:.98rem;}
.bc-step p{margin:0; font-size:.82rem; line-height:1.5; color:var(--ink-2);}
.bc-doc{max-width:880px;}
.bc-doc h3{font-size:1.15rem; margin:1.6rem 0 .5rem;}
.bc-doc p, .bc-doc li{color:var(--ink-2); font-size:.94rem; line-height:1.65;}
.bc-table{width:100%; border-collapse:collapse; font-size:.88rem; background:var(--card); border:1px solid var(--line);
  border-radius:var(--radius); overflow:hidden;}
.bc-table th{background:#f8fafb; text-align:left; font-weight:600; color:var(--ink-2); padding:.6rem .8rem;
  border-bottom:1px solid var(--line); font-size:.84rem;}
.bc-table td{padding:.6rem .8rem; border-bottom:1px solid var(--line-2); color:var(--ink); vertical-align:top;}
.bc-table tr:last-child td{border-bottom:none;}
.bc-table td.num{text-align:right; font-variant-numeric:tabular-nums;}
.bc-formula{font-family:ui-monospace, 'SF Mono', Menlo, Consolas, monospace; font-size:.86rem; background:var(--card); border:1px solid var(--line);
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
  .bc-hero{grid-template-columns:1fr; gap:1.4rem;} .bc-hero h1{font-size:2rem;}
  .bc-contents, .bc-who{grid-template-columns:1fr;}
  .bc-kpis{grid-template-columns:repeat(2,1fr);} .bc-steps{grid-template-columns:1fr 1fr;}
  .block-container{padding:1rem 1rem 2rem;}
}

.bc-glance{background:var(--brand-ink); color:#fff; border-radius:var(--radius);
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
T_PILOT = ("Trained only on hand-labelled data from one bay (Mission Bay, 2018). It is well tested here, but other "
           "coastlines look different. The full model trains on 39 coastal sites on six continents.")



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










def best_areas(report: dict) -> dict[str, float]:
    adj = report.get("error_adjusted_areas_ha")
    if adj:
        return {k: adj[k]["adjusted_ha"] if k in adj else 0.0 for k in CLS}
    return {k: report["areas_ha"].get(k, 0.0) for k in CLS}




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




def box_km2(bbox) -> float:
    """Area of a lon/lat box on the sphere (same formula as the map's live readout)."""
    r = 6378137.0
    w, s_, e, n = bbox
    return r * r * math.radians(e - w) * abs(math.sin(math.radians(n)) - math.sin(math.radians(s_))) / 1e6


# ----------------------------------------------------------------------------- chrome
EMBED = st.query_params.get("view") == "analyze"  # the website embeds only the live analysis tool







SITE_LIVE = "October 10, 2026"
VIEW = st.query_params.get("view")
if not EMBED and VIEW != "tool":
    # Landing page for old links (this Streamlit address was the project's home before the website launched).
    st.markdown(
        f'<div class="bc-top"><div class="bc-brand"><div class="bc-word">BlueCarbon<span>-AI</span></div></div></div>'
        '<div style="max-width:760px;margin:6vh auto 0;color:#0d1f29">'
        '<p style="font-family:var(--mono, monospace);font-size:.8rem;letter-spacing:.06em;color:#3f5f50;margin:0 0 10px">'
        f'NOTICE · UPDATED {SITE_LIVE.upper()}</p>'
        '<h1 style="font-family:var(--serif);font-weight:400;font-size:2.8rem;line-height:1.1;margin:0 0 18px">'
        'BlueCarbon-AI has moved to a new website</h1>'
        '<p style="font-size:1.08rem;line-height:1.6;margin:0 0 14px">If you were given this link, thank you for visiting. '
        f'On {SITE_LIVE} the project moved to its own website, which replaces this page as its home.</p>'
        '<p style="font-size:1.02rem;line-height:1.6;margin:0 0 6px">The website has habitat and carbon maps of 44 coastal '
        'sites on six continents, published accuracy on estuaries the model never saw during training, the full '
        'methodology and sample site screening reports in English and Spanish. This address now hosts only the '
        'live analysis tool.</p></div>',
        unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        a, b = st.columns(2)
        a.link_button("Visit the BlueCarbon-AI website", SITE, type="primary", use_container_width=True)
        if b.button("Open the live analysis tool", use_container_width=True):
            st.query_params["view"] = "tool"
            st.rerun()
        st.markdown(
            '<p style="font-size:.88rem;color:#4b5a61;margin-top:18px">Yanick Sanchez · '
            '<a href="https://github.com/yanicksanchez14-creator/bluecarbon-ai" target="_blank" style="color:#3f5f50">'
            f'Source code on GitHub</a> · Website: <a href="{SITE}" target="_blank" style="color:#3f5f50">{SITE}</a></p>',
            unsafe_allow_html=True)
    st.stop()
if not EMBED:  # the live tool opened directly: a slim header pointing to the website
    st.markdown(
        f'<div class="bc-top"><div class="bc-brand"><div class="bc-word">BlueCarbon<span>-AI</span></div></div>'
        f'<div class="bc-links"><a href="{SITE}" target="_blank">Website</a>'
        f'<a href="{SITE}explore.html" target="_blank">Site maps</a>'
        f'<a href="{SITE}methodology.html" target="_blank">Methodology</a></div></div>'
        '<h1 style="font-family:var(--serif);font-weight:400;font-size:2.6rem;margin:0 0 .3rem">Analyze an area</h1>',
        unsafe_allow_html=True)
tab_analyze = st.container()

# ----------------------------------------------------------------------------- explore
VIEWS = {"Habitats": "classes", "Blue carbon only": "bluecarbon", "Satellite": None, "False color": "falsecolor",
         "Confidence": "confidence"}
HINTS = {
    "Habitats": "Each color shows what the AI thinks covers that patch of ground or water.",
    "Blue carbon only": "Only mangrove, salt marsh and seagrass: the habitats counted in the carbon numbers.",
    "Satellite": "The cloud-free satellite photo the AI analysed, in natural color.",
    "False color": "An infrared view where healthy plants glow red. Scientists use it to spot vegetation that is hard to see in normal color.",
    "Confidence": "Where to doubt the map: orange marks pixels the model was unsure about (under about 90% confident), "
                  "darker orange the least sure. Clear means confident.",
}


def section(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="bc-h2">{title}</div>' + (f'<div class="bc-h2sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)






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


if EMBED:
    import streamlit.components.v1 as components

    st.markdown("<style>.block-container{padding:.4rem 1.6rem 1.4rem !important;}"
                "div[data-testid='stToolbar'],footer{display:none !important;}</style>", unsafe_allow_html=True)
    # Tell the website how tall the tool is, so its frame grows with the content (no scrollbar inside a scrollbar).
    components.html("""<script>
const doc = window.parent.document;
const target = () => doc.querySelector('[data-testid="stMainBlockContainer"]') || doc.querySelector('.block-container') || doc.body;
const send = () => { try { window.top.postMessage({type: "bc-height", h: Math.ceil(target().getBoundingClientRect().height) + 24}, "*"); } catch (e) {} };
try { new ResizeObserver(send).observe(target()); } catch (e) {}
setInterval(send, 1200); send();
</script>""", height=0)

with tab_analyze:
    sa = secret("GEE_SERVICE_ACCOUNT")
    if not sa:
        st.markdown(
            """
<div class="bc-card" style="max-width:820px;margin-top:.6rem">
  <h3 style="margin:0 0 .4rem">Map any coastline in about a minute</h3>
  <p style="color:var(--ink-2);line-height:1.6;margin:0 0 .9rem">Draw a box anywhere on Earth, pick a season, and
  BlueCarbon-AI pulls a fresh cloud-free Sentinel-2 composite from Google Earth Engine, runs the model and returns a habitat
  map, carbon report and downloadable GeoTIFF.</p>
  <div class="bc-note">This deployment hasn't been connected to Earth Engine yet, so the live pipeline is switched off.
  The mapped sites on the website work without it. The same analysis runs locally with
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
                           "circlemarker": False,
                           "rectangle": {"shapeOptions": {"color": "#14a3a0"}, "showArea": True, "metric": True}},
             edit_options={"edit": False}).add_to(m)
        # Live area while drawing. leaflet.draw's own text is in hectares without the limit (and crashes in some
        # versions), so replace it: km² and the limit, said plainly when the box is too big. Attached to the map
        # itself because streamlit-folium only runs the map's own scripts.
        from branca.element import MacroElement
        from jinja2 import Template

        class _AreaReadout(MacroElement):
            _template = Template(
                "{% macro script(this, kwargs) %}"
                "if (window.L && L.GeometryUtil) { L.GeometryUtil.readableArea = function (a) {"
                f"var k = a / 1e6, lim = {MAX_AREA_KM2};"
                "var t = (k < 10 ? k.toFixed(2) : k.toFixed(1)) + ' km²';"
                "return k > lim ? '<b style=\"color:#ffb4a2\">' + t + ' - too big, limit ' + lim + ' km²</b>'"
                " : t + ' of ' + lim + ' km² allowed'; }; }"
                "{% endmacro %}")

        m.add_child(_AreaReadout())
        out = st_folium(m, height=480, use_container_width=True, key="draw")
        feat = (out or {}).get("last_active_drawing")
        bbox = km2 = None
        if feat:
            coords = np.array(feat["geometry"]["coordinates"][0])
            bbox = [coords[:, 0].min(), coords[:, 1].min(), coords[:, 0].max(), coords[:, 1].max()]
            km2 = box_km2(bbox)
        too_big = km2 is not None and km2 > MAX_AREA_KM2
        if km2 is not None:
            msg = (f"Selected area: <b>{km2:,.1f} km²</b> ({km2 * 100:,.0f} ha). "
                   + (f"That's over the {MAX_AREA_KM2} km² limit: draw a smaller box." if too_big
                      else f"Within the {MAX_AREA_KM2} km² limit."))
            st.markdown(f'<div class="bc-note" style="margin:.4rem 0;color:{"#9b2c2c" if too_big else "inherit"}">'
                        f"{msg}</div>", unsafe_allow_html=True)
        c3.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        run = c3.button("Run analysis", type="primary", disabled=feat is None or too_big, width="stretch")
        if run and feat and not too_big:
            import rasterio
            from rasterio.warp import Resampling

            from bluecarbon import gee
            from bluecarbon.demo import _to_mercator
            from bluecarbon.features import S2_BANDS
            from bluecarbon.report import scene_report
            from bluecarbon.viz import class_rgba, true_color

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
                    if any(f.startswith("S1_") for f in getattr(predictor, "features", [])):
                        st.write("Fetching Sentinel-1 radar…")
                        gee.download_radar(bbox, str(start), str(end), tmp / "radar.tif", CFG)
                    from bluecarbon.predictors import read_ancillary

                    with rasterio.open(tmp / "image.tif") as im:
                        img_shape = im.shape
                    anc = read_ancillary(tmp / "image.tif", img_shape)
                st.write("Segmenting habitats…")
                with rasterio.open(tmp / "image.tif") as src:
                    bands, prof = src.read(), src.profile
                cls, conf = predictor.predict(bands, CFG.predict.tile, CFG.predict.overlap, anc=anc)
                from bluecarbon.priors import apply_to_classes

                cls = apply_to_classes(cls, (bbox[1] + bbox[3]) / 2)
                prof.update(count=2, dtype="uint8", nodata=255)
                with rasterio.open(tmp / "pred.tif", "w", **prof) as dst:
                    dst.write(cls, 1)
                    dst.write(np.round(np.nan_to_num(conf) * 100).astype(np.uint8), 2)
                st.write("Looking up measured mangrove biomass (NASA canopy-height map)…")
                try:
                    bio_stats = gee.site_biomass_stats(bbox)
                except Exception:  # fall back to IPCC biomass rather than fail the analysis
                    bio_stats = None
                rep = scene_report(tmp / "pred.tif", CFG.carbon, ck["metrics"].get("test_confusion"), bio_stats)
                from bluecarbon.demo import confidence_summary

                rep["confidence"] = confidence_summary(tmp / "pred.tif")
                try:  # sea-level-rise screen: room for the wetland to move inland
                    from bluecarbon.slr import slr_from_files

                    st.write("Checking room for the wetland to move inland as the sea rises…")
                    if not (tmp / "ancillary.tif").exists():
                        gee.download_ancillary(bbox, str(start), str(end), tmp / "ancillary.tif", CFG)
                    gee.download_built(bbox, tmp / "built.tif", CFG)
                    sea_level = slr_from_files(tmp / "pred.tif", tmp / "ancillary.tif", tmp / "built.tif")
                except Exception:
                    sea_level = None
                status.update(label="Analysis complete", state="complete", expanded=False)
            mb, bnds = _to_mercator(tmp / "image.tif", list(range(1, 11)), Resampling.bilinear, 0)
            mc, _ = _to_mercator(tmp / "pred.tif", [1], Resampling.nearest, 255)
            mconf, _ = _to_mercator(tmp / "pred.tif", [2], Resampling.nearest, 255)
            from bluecarbon.schema import KEY_TO_ID
            from bluecarbon.viz import confidence_rgba

            layers = {
                "classes": rgba_uri(class_rgba(mc[0], 200)),
                "bluecarbon": rgba_uri(class_rgba(mc[0], 220, [KEY_TO_ID[k] for k in BLUE_CARBON_KEYS])),
                "falsecolor": rgba_uri(true_color(mb, rgb_bands=("B8", "B4", "B3"), gamma=1.0)),
                "confidence": rgba_uri(confidence_rgba(mconf[0], mc[0])),
            }
            live_meta = {"title": "Your area", "model": {**ck.get("extra", {}), "arch": ck["arch"], "kind": ck["kind"],
                                                         "encoder": ck["encoder"], "test": ck["metrics"].get("test")}}
            pdfs = {}
            try:  # the same PDF report as the mapped sites, for the area just analysed
                from PIL import Image

                from bluecarbon.screen import build_report

                page = tmp / "page"
                page.mkdir(exist_ok=True)
                Image.fromarray(true_color(mb)).save(page / "rgb.png")
                Image.fromarray(class_rgba(mc[0], 255)).save(page / "classes.png")
                lat, lon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2
                (page / "meta.json").write_text(json.dumps({
                    **live_meta, "kind": "single", "report": rep, "period": f"{start} to {end}", "sea_level": sea_level,
                    "region": f"{abs(lat):.3f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.3f}° {'E' if lon >= 0 else 'W'}"}))
                pdfs = {lang: build_report(page, None, lang) for lang in ("en", "es")}
            except Exception as e:  # never lose the analysis over a report problem
                st.caption(f"PDF report unavailable ({str(e)[:80]})")
            st.session_state["live"] = {"meta": live_meta, "report": rep, "bounds": bnds, "layers": layers,
                                        "rgb": rgba_uri(true_color(mb)), "tif": (tmp / "pred.tif").read_bytes(),
                                        "pdf": pdfs, "sea_level": sea_level}

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
            confs = (live["report"].get("confidence") or {})
            if confs:
                section("How sure is the model here?", "Model confidence on this image, per habitat. Switch the map to "
                        "Confidence to see where it was unsure.")
                st.markdown('<div class="bc-card bc-found"><p>' + " ".join(
                    f"{CLS[k].name}: {v['mean_pct']:.0f}% average confidence, {v['low_share']:.0%} of its area under 60%."
                    for k, v in confs.items()) + "</p></div>", unsafe_allow_html=True)
            sl = live.get("sea_level")
            if sl and sl.get("ratio") is not None:
                rating = {"low": "low risk", "medium": "medium risk", "high": "high risk (coastal squeeze)"}[sl["rating"]]
                section("Sea-level rise", "Can the wetland move inland as the sea rises? Verra VM0033 asks every project.")
                st.markdown(f'<div class="bc-card bc-found"><p>Within {sl["search_km"]:g} km behind the '
                            f'{fmt(sl["wetland_ha"])} ha of mangrove and salt marsh there are <b>{fmt(sl["room_ha"])} ha</b> '
                            f'of open, low land (0 to {sl["max_elev_m"]:g} m above sea level, not built up): '
                            f'{sl["ratio"]:.0%} of the wetland\'s own area. Rating: <b>{rating}</b>. Global sea level is '
                            f'projected to rise {sl["ar6_2100_m"][0]:g} to {sl["ar6_2100_m"][1]:g} m by 2100 (IPCC AR6).'
                            "</p></div>", unsafe_allow_html=True)
            section("Download", "The report, the habitat map for GIS software, and the raw numbers.")
            d1, d2, d3, d4 = st.columns(4)
            pdfs = live.get("pdf") or {}
            if pdfs.get("en"):
                d1.download_button("PDF report (English)", pdfs["en"], "bluecarbon_report_en.pdf", mime="application/pdf",
                                   type="primary", width="stretch")
            if pdfs.get("es"):
                d2.download_button("Informe PDF (español)", pdfs["es"], "bluecarbon_informe_es.pdf", mime="application/pdf",
                                   width="stretch")
            d3.download_button("Habitat map (GeoTIFF)", live["tif"], "bluecarbon_habitats.tif", width="stretch")
            d4.download_button("Numbers (JSON)", json.dumps(live["report"], indent=2), "bluecarbon_report.json",
                               width="stretch")

# ----------------------------------------------------------------------------- site screening

# ----------------------------------------------------------------------------- methodology

# ----------------------------------------------------------------------------- about

