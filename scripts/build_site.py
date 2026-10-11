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
        "areas_ha": _best_areas(rep), "carbon": _carbon(rep), "images": imgs,
    }
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
        "n_soil_cores": sum(1 for _ in cores.open()) - 1 if cores.exists() else 0,
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
    _chrome(out)
    _cache_bust(out)
    (out / ".nojekyll").write_text("")
    return doc


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
            sites.append(json.loads(p.read_text()))
    cal = json.loads((src / "calibration.json").read_text()) if (src / "calibration.json").exists() else {}
    (out / "data" / "kelp.json").write_text(json.dumps({"sites": sites, "calibration": cal}, separators=(",", ":")))


def _chrome(out: Path) -> None:
    """One header and footer for every page (site/partials), with the current page marked."""
    header = (ROOT / "site" / "partials" / "header.html").read_text()
    footer = (ROOT / "site" / "partials" / "footer.html").read_text()
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "_site")
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    doc = build(a.out, pdf=not a.no_pdf)
    print(f"built {len(doc['sites'])} site pages -> {a.out}")


if __name__ == "__main__":
    main()
