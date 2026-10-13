"""Build the public website (static HTML, served free by GitHub Pages) into _site/.

    python scripts/build_site.py            # full build, with a PDF report for every site
    python scripts/build_site.py --no-pdf   # quick build for checking the layout

Sources: site/ (pages, CSS, JS), demo_data/ (maps and numbers for every site), reports/screens/ (hand-made
site screens), docs/METHODOLOGY.md. Images are converted to compressed JPEG / PNG so pages load fast; the
Streamlit app keeps using the full-size originals.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bluecarbon.schema import BLUE_CARBON_KEYS, CLASSES  # noqa: E402

COUNTRY_ALIASES = {"Western Australia": "Australia"}
PHOTO_PX = 1600  # satellite images (JPEG)
THUMB_PX = 480


def _jpeg(src: Path, dst: Path, px: int = PHOTO_PX, quality: int = 82) -> None:
    from PIL import Image

    im = Image.open(src)
    if im.mode in ("RGBA", "LA", "P"):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", im.size, (11, 27, 36))
        bg.paste(rgba, mask=rgba.split()[3])
        im = bg
    im = im.convert("RGB")
    im.thumbnail((px, px))
    im.save(dst, "JPEG", quality=quality, optimize=True, progressive=True)


def _png(src: Path, dst: Path) -> None:
    """Class overlays: few colours, so a palette PNG is small and stays sharp."""
    from PIL import Image

    im = Image.open(src).convert("RGBA")
    im.quantize(colors=32, method=Image.Quantize.FASTOCTREE).save(dst, optimize=True)


def _content_box(path: Path) -> tuple[int, int, int, int]:
    """Bounding box without the black no-data margin that reprojection leaves around a scene."""
    import numpy as np
    from PIL import Image

    a = np.asarray(Image.open(path).convert("RGB")).max(axis=2) > 12
    rows, cols = np.where(a.mean(axis=1) > 0.98)[0], np.where(a.mean(axis=0) > 0.98)[0]
    if not len(rows) or not len(cols):
        return 0, 0, a.shape[1], a.shape[0]
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def _hero(page: Path, map_dst: Path, sat_dst: Path) -> None:
    from PIL import Image

    box = _content_box(page / "rgb.png")
    base = Image.open(page / "rgb.png").convert("RGBA").crop(box)
    im = Image.open(page / "classes.png").convert("RGBA").resize(Image.open(page / "rgb.png").size).crop(box)
    a = im.split()[3].point(lambda v: 215 if v else 0)  # a little of the photo shows through
    im.putalpha(a)
    Image.alpha_composite(base, im).convert("RGB").save(map_dst, "JPEG", quality=84, optimize=True, progressive=True)
    base.convert("RGB").save(sat_dst, "JPEG", quality=84, optimize=True, progressive=True)


def _best_areas(rep: dict) -> dict[str, float]:
    adj = rep.get("error_adjusted_areas_ha") or {}
    return {c.key: round(adj[c.key]["adjusted_ha"] if c.key in adj else rep["areas_ha"].get(c.key, 0.0), 1)
            for c in CLASSES}


def _carbon(rep: dict) -> dict:
    c = rep["carbon"]
    r = lambda d: {k: round(v) for k, v in d.items()}  # noqa: E731
    per = {}
    for k, v in c["classes"].items():
        soil = v.get("soil") or {}
        bio = v.get("biomass") or {}
        per[k] = {"area_ha": round(v["area_ha"], 1), "stock_tco2e": r(v["stock_tCO2e"]),
                  "biomass": ({"source": "measured", "agb_mg_ha": bio.get("agb_mg_ha")} if bio.get("source") == "measured"
                              else {"source": "ipcc"}),
                  "seq_tco2e_yr": r(v["sequestration_tCO2e_per_yr"]),
                  "soil": ({"source": "measured", "cores": soil.get("n_cores"), "studies": soil.get("n_studies"),
                            "radius_km": soil.get("radius_km")} if soil.get("source") == "measured"
                           else {"source": "ipcc"})}
    return {"stock_tco2e": r(c["total_stock_tCO2e"]), "seq_tco2e_yr": r(c["total_sequestration_tCO2e_per_yr"]),
            "value_usd_yr": r(c.get("indicative_annual_value_usd") or {}), "classes": per}


def _lidar(sid: str) -> dict | None:
    """GEDI lidar biomass and canopy height over the site's mangroves (data/site_biomass.json), if measured."""
    p = ROOT / "data" / "site_biomass.json"
    b = (json.loads(p.read_text()) if p.exists() else {}).get(sid) or {}
    if not b.get("gedi_agbd_mean") or (b.get("gedi_footprints") or 0) < 20:
        return None
    return {"gedi_agb": round(b["gedi_agbd_mean"]), "footprints": b["gedi_footprints"],
            "map_agb": round(b["agb_mean"]) if b.get("agb_mean") else None,
            "canopy_h": round(b["canopy_h_mean"], 1) if b.get("canopy_h_mean") else None,
            "meta_h": round(b["meta_h_mean"], 1) if b.get("meta_h_mean") else None,
            "height_agb": round(b["height_agb_mean"]) if b.get("height_agb_mean") and (b.get("hb_footprints") or 0) >= 30
            else None,
            "height_agb_rmse": b.get("hb_rmse_cv"), "height_agb_r2": b.get("hb_r2_cv")}


def site_record(page: Path, out: Path) -> tuple[dict, dict]:
    meta = json.loads((page / "meta.json").read_text())
    sid = page.name
    d = out / "sites" / sid
    d.mkdir(parents=True, exist_ok=True)
    change = meta.get("kind") == "change"
    sfx = ["_t0", "_t1"] if change else [""]
    imgs = {}
    for s in sfx:
        for name in ("rgb", "falsecolor"):
            if (page / f"{name}{s}.png").exists():
                _jpeg(page / f"{name}{s}.png", d / f"{name}{s}.jpg")
                imgs[f"{name}{s}"] = f"data/sites/{sid}/{name}{s}.jpg"
        for name in ("classes", "bluecarbon", "confidence"):
            if (page / f"{name}{s}.png").exists():
                _png(page / f"{name}{s}.png", d / f"{name}{s}.png")
                imgs[f"{name}{s}"] = f"data/sites/{sid}/{name}{s}.png"
    if change and (page / "change.png").exists():
        _png(page / "change.png", d / "change.png")
        imgs["change"] = f"data/sites/{sid}/change.png"
    rgb = page / ("rgb_t1.png" if change else "rgb.png")
    from PIL import Image

    Image.open(rgb).convert("RGB").crop(_content_box(rgb)).save(d / "thumb_full.png")
    _jpeg(d / "thumb_full.png", d / "thumb.jpg", THUMB_PX, 78)
    (d / "thumb_full.png").unlink()
    imgs["thumb"] = f"data/sites/{sid}/thumb.jpg"

    rep = meta["t1"] if change else meta["report"]
    (s, w), (n, e) = meta["bounds"]
    region = meta.get("region", "")
    country = region.split(",")[-1].strip()
    rec = {
        "id": sid, "title": meta["title"], "region": region, "country": COUNTRY_ALIASES.get(country, country),
        "period": meta.get("period", ""), "kind": meta["kind"], "held_out": bool(meta.get("held_out")),
        "description": meta.get("description", ""), "bounds": meta["bounds"],
        "center": [round((s + n) / 2, 4), round((w + e) / 2, 4)],
        "areas_ha": _best_areas(rep), "carbon": _carbon(rep), "images": imgs, "lidar": _lidar(sid),
    }
    gp = ROOT / "data" / "site_gmw.json"
    gmw = (json.loads(gp.read_text()) if gp.exists() else {}).get(sid)
    if gmw and gmw.get("gmw_ha"):
        rec["gmw"] = gmw
    from bluecarbon.sources import number_sources

    rp = ROOT / "models" / "model_report.json"
    mv = f"model v2, code {json.loads(rp.read_text()).get('code_version', '')}" if rp.exists() else "model v2"
    rec["sources"] = number_sources(rep["carbon"]["classes"], "en", mv, rec["lidar"], bool(meta.get("sea_level")))
    if (page / "history.json").exists():
        h = json.loads((page / "history.json").read_text())
        rec["history"] = {"epochs": [{"year": e["year"], **({"ha": {k: v["ha"] for k, v in e["adjusted_ha"].items()},
                                                            "ci": {k: v["ci95"] for k, v in e["adjusted_ha"].items()}}
                                                           if "adjusted_ha" in e else {})} for e in h["epochs"]],
                          "model": h.get("model"), "scale_m": h.get("scale_m")}
    if rep.get("confidence"):
        rec["confidence"] = rep["confidence"]
    if meta.get("sea_level"):
        rec["sea_level"] = meta["sea_level"]
    if change:
        p = meta.get("periods") or {}
        a0, a1 = _best_areas(meta["t0"]), _best_areas(meta["t1"])
        rec["change"] = {"t0": p.get("t0", "Before"), "t1": p.get("t1", "After"),
                         "areas_t0": a0, "areas_t1": a1,
                         "stock_t0": round(meta["t0"]["carbon"]["total_stock_tCO2e"]["mean"]),
                         "stock_t1": round(meta["t1"]["carbon"]["total_stock_tCO2e"]["mean"])}
    return rec, meta.get("model") or {}


def _cores_used(pages: list[Path]) -> int:
    """Soil cores that actually feed a site page's carbon (within the radius each site used)."""
    import numpy as np

    from bluecarbon.carbon import _cores, _km

    c = _cores()
    if c is None:
        return 0
    used: set[int] = set()
    for p in pages:
        m = json.loads((p / "meta.json").read_text())
        (s, w), (n, e) = m["bounds"]
        for rep in ([m["report"]] if "report" in m else [m[t] for t in ("t0", "t1") if t in m]):
            for k, v in rep["carbon"]["classes"].items():
                if (v.get("soil") or {}).get("source") == "measured":
                    d = _km((s + n) / 2, (w + e) / 2, c["lat"], c["lon"])
                    used.update(np.flatnonzero((c["habitat"] == k) & (d <= v["soil"]["radius_km"])).tolist())
    return len(used)


def build(out: Path, pdf: bool = True) -> dict:
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out, ignore=shutil.ignore_patterns("partials"))
    data = out / "data"
    data.mkdir(exist_ok=True)
    demo = ROOT / "demo_data"
    pages = sorted(p for p in demo.iterdir() if (p / "meta.json").exists())
    sites, model = [], {}
    for p in pages:
        rec, m = site_record(p, data)
        sites.append(rec)
        model = model or m
        print(f"  {p.name}", flush=True)
        if pdf:
            from bluecarbon.screen import build_report

            for lang in ("en", "es"):
                build_report(p, data / "sites" / p.name / f"report_{lang}.pdf", lang)
            rec["reports"] = {lang: f"data/sites/{p.name}/report_{lang}.pdf" for lang in ("en", "es")}

    hero = demo / "laguna_terminos_mx"
    if hero.exists():
        _hero(hero, data / "hero_map.jpg", data / "hero_sat.jpg")

    screens = {}
    for f in sorted((ROOT / "reports" / "screens").glob("*.pdf")):
        shutil.copy(f, data / f.name)
        screens[f.stem] = f"data/{f.name}"

    cores = ROOT / "data" / "soil_carbon_cores.csv"
    n_sites = [s for s in sites if s["kind"] != "change"]
    summary = {
        "n_sites": len(n_sites), "n_change": len(sites) - len(n_sites),
        "n_countries": len({s["country"] for s in n_sites}),
        "n_held_out": sum(s["held_out"] for s in n_sites),
        "n_train_sites": sum(not s["held_out"] for s in n_sites),
        "n_soil_cores": sum(1 for _ in cores.open()) - 1 if cores.exists() else 0,
        "n_soil_cores_used": _cores_used(pages),
        "blue_carbon_ha": round(sum(sum(s["areas_ha"][k] for k in BLUE_CARBON_KEYS) for s in n_sites)),
        "stock_tco2e": sum(s["carbon"]["stock_tco2e"]["mean"] for s in n_sites),
    }
    test = model.get("test") or {}
    doc = {
        "summary": summary,
        "model": {"name": model.get("name", "BlueCarbon-AI"), "arch": model.get("arch"), "encoder": model.get("encoder"),
                  "iou": test.get("iou", {}), "f1": test.get("f1", {}), "miou": test.get("mIoU"),
                  "evaluation": model.get("evaluation", ""), "training_data": model.get("training_data", "")},
        "classes": [{"key": c.key, "name": c.name, "color": c.color, "blue_carbon": c.blue_carbon} for c in CLASSES],
        "screens": screens,
        "sites": sites,
    }
    (data / "sites.json").write_text(json.dumps(doc, separators=(",", ":")))
    _kelp(out)
    _methodology(out, doc)
    _accuracy(out, doc)
    _chrome(out)
    _cache_bust(out)
    (out / ".nojekyll").write_text("")
    return doc


def _canopy_focus(d: Path, meta: dict) -> list | None:
    """[[s, w], [n, e]] around where the canopy actually is (2nd to 98th percentile of canopy pixels across the
    quarterly maps, or the latest map), so the kelp page opens close enough to see the kelp."""
    import math

    import numpy as np
    from PIL import Image

    pngs = sorted((d / "frames").glob("*.png"))[-12:] or [d / "canopy_latest.png"]
    bounds = meta.get("frame_bounds") or meta.get("bounds")
    rows, cols, shape = [], [], None
    for png in pngs:
        if not png.exists():
            continue
        a = np.asarray(Image.open(png).convert("RGBA"))[..., 3]
        shape = a.shape
        r, c = np.nonzero(a > 0)
        rows.append(r)
        cols.append(c)
    if not shape or not rows or not sum(len(r) for r in rows):
        return None
    r, c = np.concatenate(rows), np.concatenate(cols)
    (s, w), (n, e) = bounds
    my = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))  # noqa: E731
    lat = lambda y: math.degrees(2 * math.atan(math.exp(y)) - math.pi / 2)  # noqa: E731
    h, wd = shape
    r0, r1 = np.percentile(r, [2, 98])
    c0, c1 = np.percentile(c, [2, 98])
    yn, ys = my(n), my(s)
    north, south = lat(yn - (yn - ys) * r0 / h), lat(yn - (yn - ys) * (r1 + 1) / h)
    west, east = w + (e - w) * c0 / wd, w + (e - w) * (c1 + 1) / wd
    pad_lat, pad_lon = max((north - south) * 0.15, 0.004), max((east - west) * 0.15, 0.005)
    return [[round(south - pad_lat, 5), round(west - pad_lon, 5)], [round(north + pad_lat, 5), round(east + pad_lon, 5)]]


def _kelp(out: Path) -> None:
    """Kelp pages from demo_kelp/ (scripts/kelp_pipeline.py): index + per-site canopy overlay."""
    src = ROOT / "demo_kelp"
    sites = []
    if src.exists():
        for p in sorted(src.glob("*/kelp.json")):
            d = out / "data" / "kelp" / p.parent.name
            d.mkdir(parents=True, exist_ok=True)
            if (p.parent / "canopy_latest.png").exists():
                shutil.copy(p.parent / "canopy_latest.png", d / "canopy_latest.png")
            for sub in ("frames", "kw", "rgb"):   # every quarter's map, Kelpwatch yearly maps, yearly pictures
                if (p.parent / sub).is_dir():
                    shutil.copytree(p.parent / sub, d / sub, dirs_exist_ok=True)
            meta = json.loads(p.read_text())
            focus = _canopy_focus(p.parent, meta)
            if focus:
                meta["focus"] = focus
            sites.append(meta)
    cal = json.loads((src / "calibration.json").read_text()) if (src / "calibration.json").exists() else {}
    (out / "data" / "kelp.json").write_text(json.dumps({"sites": sites, "calibration": cal}, separators=(",", ":")))


def _chrome(out: Path) -> None:
    """One header and footer for every page (site/partials), with the current page marked."""
    header = (ROOT / "site" / "partials" / "header.html").read_text()
    footer = (ROOT / "site" / "partials" / "footer.html").read_text()
    rp = ROOT / "models" / "model_report.json"
    if rp.exists():
        r = json.loads(rp.read_text())
        footer = footer.replace("<!-- VERSION -->", f'<p class="version">Model v2 \u00b7 code {r.get("code_version", "")} '
                                f'\u00b7 published {r.get("installed", "")} \u00b7 <a href="accuracy.html">accuracy</a></p>')
    for page in out.glob("*.html"):
        name = page.stem
        h = header.replace(f'data-page="{name}"', f'data-page="{name}" aria-current="page"')
        if name != "index":  # pages without a full-bleed hero keep the header border from the start
            h = h.replace('<header class="site-header">', '<header class="site-header scrolled">')
        page.write_text(page.read_text().replace("<!-- HEADER -->", h).replace("<!-- FOOTER -->", footer))


def _cache_bust(out: Path) -> None:
    """Add ?v=<content hash> to CSS/JS links so browsers never keep a stale copy after an update."""
    import hashlib
    import re

    def stamp(m):
        f = out / m.group(2)
        v = hashlib.sha1(f.read_bytes()).hexdigest()[:10] if f.exists() else "0"
        return f'{m.group(1)}="{m.group(2)}?v={v}"'

    for page in out.glob("*.html"):
        html = re.sub(r'(src|href)="((?:css|js)/[^"?]+)"', stamp, page.read_text())
        page.write_text(html)


def _error_table(doc: dict) -> str:
    """'Error on every number': what each published number is, and how wrong it can be."""
    import html as h

    iou = (doc.get("model") or {}).get("iou") or {}
    f = lambda k: f"{iou[k]:.2f}" if iou.get(k) is not None else "–"  # noqa: E731
    hist = next((s["history"]["model"] for s in doc["sites"] if s.get("history")), None)
    hiou = (hist or {}).get("iou") or {}
    kelp_p = ROOT / "demo_kelp" / "calibration.json"
    kc = json.loads(kelp_p.read_text()) if kelp_p.exists() else {}
    ks = ((kc.get("held_out") or {}).get("site_quarter_ha") or {})
    rows = [
        ("Habitat map, per pixel", f"Scored on test areas kept out of training (5 whole estuaries plus held-back 5 km "
         f"blocks): IoU mangrove {f('mangrove')}, "
         f"salt marsh {f('saltmarsh')}, seagrass {f('seagrass')}", "Model test set, published with every model"),
        ("Habitat areas (ha)", "Corrected for the model's known mistakes with its held-out confusion matrix; 95% interval "
         "shown. Site screens add a plain-language range of \u00b1(1 \u2212 F1)", "Olofsson et al. (2014)"),
        ("Soil carbon (t C/ha)", "Measured: 5-95% range of nearby soil cores, widened when few independent studies. "
         "Otherwise the IPCC Tier 1 range", "Smithsonian Coastal Carbon Library; IPCC 2013 Wetlands Supplement"),
        ("Mangrove biomass (t C/ha)", "NASA canopy-height biomass map \u00b130% (no per-pixel error published), "
         "times IPCC carbon fraction and root ratio 95% intervals. Otherwise the IPCC range",
         "Simard et al. (2019); IPCC Tables 4.2 and 4.5"),
        ("Other biomass and burial rates", "IPCC Tier 1 ranges", "IPCC 2013 Wetlands Supplement"),
        ("Carbon stored and absorbed", "5,000 Monte Carlo draws over every input above at once; 90% interval shown",
         "Our own propagation"),
        ("Credit potential", "Low, mid and high from the absorption range, after risk buffer, leakage and uncertainty "
         "deductions. Illustrative: registries set the real deductions", "Business-plan formula; Verra VM0033"),
        ("Sea-level-rise room (ha)", "No statistical error: elevation is NASADEM, which reads the top of vegetation, so "
         "the room is a conservative low estimate", "NASADEM; ESA WorldCover; IPCC AR6 for 2100"),
        ("Model confidence", "The model's own probability per pixel. Not yet calibrated against held-out accuracy, so "
         "read it as a ranking of where to doubt the map", "Our model"),
        ("Habitat since 1985 (ha)", "Separate Landsat model, IoU on the 5 held-out estuaries: "
         + (", ".join(f"{k.replace('saltmarsh', 'salt marsh')} {v:.2f}" for k, v in hiou.items()
                      if k in ("mangrove", "saltmarsh", "seagrass") and v is not None) or "published after the first run")
         + "; 95% interval per year", "Landsat 5/7/8/9; Olofsson et al. (2014)"),
        ("Kelp canopy (ha per quarter)", (f"Against Kelpwatch on kelp beds left out of calibration: r\u00b2 {ks['r2']}, "
                                          f"error \u00b1{ks['rmse']} ha, bias {ks['bias']} ha")
         if ks.get("r2") is not None else "Published after the first calibration run", "Kelpwatch (UC Santa Barbara)"),
    ]
    body = "".join(f"<tr><td>{h.escape(a)}</td><td>{h.escape(b)}</td><td>{h.escape(c)}</td></tr>" for a, b, c in rows)
    return ("<h2 id='error-on-every-number'>Error on every number</h2><p>Every number this site and its reports "
            "publish, with how wrong it can be and where that comes from.</p><table><thead><tr><th>Number</th>"
            f"<th>Error published</th><th>Source</th></tr></thead><tbody>{body}</tbody></table>")


def _methodology(out: Path, doc: dict | None = None) -> None:
    import re

    import markdown

    md = (ROOT / "docs" / "METHODOLOGY.md").read_text()
    md = md.replace("](img/", "](data/docs/")
    (out / "data" / "docs").mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "docs" / "img").glob("*"):
        shutil.copy(f, out / "data" / "docs" / f.name)
    html = markdown.markdown(md, extensions=["tables", "fenced_code", "toc"])
    # The text diagram in the Markdown (for GitHub) becomes a real figure on the website.
    fig = (ROOT / "site" / "partials" / "pipeline.html").read_text()
    html = re.sub(r"<pre><code>[^<]*Sentinel-2 L2A.*?</code></pre>", lambda _: fig, html, count=1, flags=re.S)
    if doc is not None:  # the error table goes right after the pipeline figure
        html = html.replace("</figure>", "</figure>" + _error_table(doc), 1)
    page = out / "methodology.html"
    page.write_text(page.read_text().replace("<!-- METHODOLOGY -->", html))


def _accuracy(out: Path, doc: dict) -> None:
    """Model card: version, training data, held-out scores per estuary, live-vs-new comparison, known issues."""
    import html as h

    rp = ROOT / "models" / "model_report.json"
    if not rp.exists() or not (out / "accuracy.html").exists():
        return
    r = json.loads(rp.read_text())
    titles = {s["id"]: s["title"] for s in doc["sites"]}
    name = lambda s: h.escape(titles.get(s, s.replace("_", " ")))  # noqa: E731
    f2 = lambda v: "–" if v is None else f"{v:.2f}"  # noqa: E731
    H = {"mangrove": "Mangrove", "saltmarsh": "Salt marsh", "seagrass": "Seagrass"}
    DASH = "\u2013"
    MIN_REF_HA = 25
    test = r.get("test") or {}
    held = [s["id"] for s in doc["sites"] if s.get("held_out") and s["id"] in (r.get("areas") or r.get("test_per_site") or {})]
    # whole-estuary scores from the publishing comparison (same numbers as the table below it); test-chip
    # scores only where the comparison has none
    whole = {(x["site"], x["habitat"]): x["new"] for x in (r.get("compare") or {}).get("rows", [])}
    parts = ['<h2 id="model">This model</h2><table><tbody>',
             f"<tr><th>Version</th><td>BlueCarbon-AI v2, code {h.escape(r.get('code_version', '?'))}, "
             f"published {h.escape(r.get('installed', ''))}</td></tr>",
             "<tr><th>Architecture</th><td>" + {
                 "ensemble": "U-Net (ResNet-34 encoder) and a LightGBM pixel model, blended habitat by habitat with "
                             "weights picked on validation areas",
                 "spectral": "LightGBM pixel model with neighbourhood context",
             }.get(r.get("model_kind"), "U-Net (ResNet-34 encoder)") + " on Sentinel-2 bands and indices, "
             "clear-water bands, elevation, tidal-wetland probability and Sentinel-1 radar</td></tr>",
             f"<tr><th>Training data</th><td>{doc['summary'].get('n_train_sites', '?')} coastal sites in "
             f"{doc['summary'].get('n_countries', '?')} countries, 2020-2022 imagery; answer keys from published maps "
             f"and official surveys (see <a href=\"methodology.html\">methodology</a>)</td></tr>",
             f"<tr><th>Training</th><td>{r.get('epochs_run', '?')} epochs, best at epoch {r.get('best_epoch', '?')}</td></tr>",
             f"<tr><th>Test score (all test areas)</th><td>mean IoU {f2(test.get('mIoU'))}, overall accuracy "
             f"{f2(test.get('overall_accuracy'))}</td></tr></tbody></table>"]
    def card(title, score, note):
        return (f'<div class="score-card"><p class="sc-h">{title}</p><p class="sc-n">{score}</p>'
                f'<p class="sc-t">{note}</p></div>')

    def rng(k, sites):
        v = [whole.get((s, k)) for s in sites if whole.get((s, k)) is not None]
        return f"{min(v):.2f}\u2013{max(v):.2f}" if len(v) > 1 else (f2(v[0]) if v else DASH)

    def ref(s, k):
        return (((r.get("areas") or {}).get(s) or {}).get(k) or {}).get("reference", 0)

    # marsh estuaries: salt marsh is the main vegetated blue carbon habitat (not a fringe behind mangroves)
    big_marsh = [s for s in held if ref(s, "saltmarsh") >= MIN_REF_HA and ref(s, "saltmarsh") > ref(s, "mangrove")]
    parts += ['<h2 id="summary">Results on coastlines the model never saw</h2>',
              "<p>Five estuaries were kept out of training entirely and used as a final exam, the same situation as "
              "the live analysis tool on any new coast. Scores are overlap with official reference maps, from 0 to 1.</p>",
              '<div class="score-cards">',
              card("Mangrove", rng("mangrove", held), "Found on every unseen mangrove coast, Australia and Florida"),
              card("Salt marsh", rng("saltmarsh", big_marsh), "On the extensive marshes of Plum Island and Mission Bay"),
              card("Seagrass, clear water", f2(whole.get(("tampa_bay_fl", "seagrass"))),
                   "Tampa Bay; in murky water meadows are partly hidden, so seagrass is reported as a lower bound"),
              card("Improvement", f"+{(cmp_.get('mean_new') or 0) - (cmp_.get('mean_live') or 0):.2f}"
                   if (cmp_ := (r.get("compare") or {})).get("mean_new") else DASH,
                   "Mean blue carbon score over the previous model on the same estuaries"),
              "</div>",
              '<details class="more-detail"><summary>Full results for each estuary</summary>',
              '<h3 id="held-out">Score per estuary and habitat</h3>',
              "<p>These five estuaries were kept out of training entirely, so they show how the model does on a "
              "new coastline, which is what the live analysis tool does every time. Score is IoU: the overlap "
              "between the model's map and the reference map, from 0 (no overlap) to 1 (perfect). \u2013 means the habitat "
              "is not there, or covers under 25 hectares, too little to score fairly.</p>",
              "<table><thead><tr><th>Estuary</th>" + "".join(f"<th>{v}</th>" for v in H.values()) + "</tr></thead><tbody>"]
    per = r.get("test_per_site") or {}
    for s in held:
        iou = per.get(s) or {}
        ar = (r.get("areas") or {}).get(s) or {}
        cells = []
        for k in H:
            ref = (ar.get(k) or {}).get("reference", 1000)
            if ref <= 0:
                cells.append(f"<td>{DASH}</td>")
            elif ref < MIN_REF_HA:  # a few hectares: one misplaced field edge swings the score from 0 to 1
                cells.append(f"<td>{DASH}</td>")
            else:
                cells.append(f"<td>{f2(whole.get((s, k), iou.get(k)))}</td>")
        parts.append(f"<tr><td>{name(s)}</td>{''.join(cells)}</tr>")
    parts.append("</tbody></table>")
    cmp_ = r.get("compare") or {}
    if cmp_.get("rows"):
        parts += ['<h3 id="compare">Previous model against current, same estuaries</h3>',
                  "<p>A new model is published only if, on these same estuaries and the same labels, its mean blue carbon "
                  "score is at least the live model's and no habitat at any estuary drops by more than 0.05. "
                  f"This model: mean {f2(cmp_.get('mean_new'))} against {f2(cmp_.get('mean_live'))}, verdict "
                  f"<b>{h.escape(str(cmp_.get('verdict')))}</b>.</p>",
                  "<table><thead><tr><th>Estuary</th><th>Habitat</th><th>Previous</th><th>Current</th><th>Change</th>"
                  "</tr></thead><tbody>"]
        for x in cmp_["rows"]:
            d = x["new"] - x["live"]
            parts.append(f"<tr><td>{name(x['site'])}</td><td>{H[x['habitat']]}</td><td>{x['live']:.2f}</td>"
                         f"<td>{x['new']:.2f}</td><td>{d:+.2f}</td></tr>")
        parts.append("</tbody></table>")
    if r.get("areas"):
        parts += ['<h3 id="areas">Mapped area against the reference map (hectares)</h3><table><thead><tr><th>Estuary</th>'
                  + "".join(f"<th>{v}</th>" for v in H.values()) + "</tr></thead><tbody>"]
        for s in held:
            ar = r["areas"].get(s) or {}
            cells = "".join(f"<td>{(ar.get(k) or {}).get('mapped', 0):,} / {(ar.get(k) or {}).get('reference', 0):,}</td>"
                            if (ar.get(k) or {}).get("reference") else "<td>\u2013</td>" for k in H)
            parts.append(f"<tr><td>{name(s)}</td>{cells}</tr>")
        parts.append("</tbody></table>")
    parts.append("</details>")
    kc_p = ROOT / "demo_kelp" / "calibration.json"
    if kc_p.exists():
        kc = json.loads(kc_p.read_text())
        sq, cm = (kc.get("held_out") or {}).get("site_quarter_ha") or {}, (kc.get("held_out") or {}).get("cells_m2") or {}
        if sq.get("r2") is not None:
            parts += ['<h2 id="kelp">Kelp canopy</h2>',
                      "<p>A separate Sentinel-2 method, calibrated against Kelpwatch (UC Santa Barbara) on "
                      f"{len(kc.get('sites_calibration', [])) or 'five'} kelp beds and scored on two it was never tuned on (" + ", ".join(
                          s.replace("_", " ").title() for s in kc.get("sites_held_out", [])) + "), every quarter "
                      "since 2016. See the <a href=\"kelp.html\">kelp page</a>.</p>",
                      '<div class="score-cards">',
                      card("Agreement, site total", f"{sq['r2']:.2f}", f"r\u00b2 against Kelpwatch, {sq['n']} site-quarters"),
                      card("Tracks change over time", f"{sq['corr']:.2f}", "Correlation with Kelpwatch"),
                      card("Typical error", f"\u00b1{sq['rmse']:.0f} ha", f"Per site and quarter; bias {sq['bias']:+.1f} ha"),
                      card("Placement", f"{cm.get('r2', 0):.2f}", "r\u00b2 per 300 m cell"),
                      "</div>"]
    parts += ['<h2 id="limits">Limits, and what the next model adds</h2><ul>',
              "<li><b>Seagrass in murky water.</b> No satellite sees the seafloor through turbid water, and many surveyed "
              "meadows at Moreton and Shoalwater Bay were mapped by boat and diver in water the satellite cannot see "
              "into. There the model finds about a quarter of the surveyed area, so seagrass is reported as a lower "
              "bound.</li>",
              "<li><b>Thin salt marsh behind mangroves</b> (Moreton Bay) is often only a few pixels wide and is partly "
              "mapped as mangrove or land.</li>",
              "<li>Areas are corrected for the model's known mistakes, but the correction for blue carbon habitats is "
              "capped at what the model's measured accuracy supports (mapped area \u00d7 precision \u00f7 recall), so "
              "carbon is never based on habitat the map did not find. That is deliberately conservative: where seagrass is "
              "badly under-mapped (Moreton Bay) the published area stays below the true area.</li>",
              "<li>Scores are against published and official reference maps, which have their own errors.</li>",
              "<li><b>Next model</b>: L-band radar from ALOS PALSAR-2, which sees through the canopy to trunks and roots; "
              "a satellite-derived depth layer, with seagrass learned only where the seafloor is visible from space; "
              "95 sites with new official survey maps from Europe, Australia and the US; and two more held-out "
              "estuaries (Brisbane Water and the Ribble).</li></ul>"]
    page = out / "accuracy.html"
    page.write_text(page.read_text().replace("<!-- ACCURACY -->", "\n".join(parts)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    doc = build(a.out, pdf=not a.no_pdf)
    print(f"built {len(doc['sites'])} site pages -> {a.out}")


if __name__ == "__main__":
    main()
