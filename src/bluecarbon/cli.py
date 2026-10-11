"""Command-line interface: `bluecarbon --help`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from .config import Config, load_config

app = typer.Typer(add_completion=False, help="BlueCarbon-AI: map blue carbon ecosystems from Sentinel-2.")

CfgOpt = typer.Option("configs/default.yaml", "--config", "-c", help="Pipeline config YAML")
SitesOpt = typer.Option("configs/sites.yaml", "--sites", help="Sites YAML")


def _cfg(path: str) -> Config:
    return load_config(path if Path(path).exists() else None)


def _sites(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _fetch_extra_years(gee, site: dict, d: Path, years: list[int], cfg) -> None:
    """Extra full-year images (+ context layers) of a training site, labelled with the main year's map.
    Habitat changes little in a year or two, so neighbouring years add realistic variation in tide,
    season, water clarity and atmosphere: more training data from the same reference labels."""
    from .features import S2_BANDS

    region = None
    for y in years:
        img_p, anc_p = d / f"image_{y}.tif", d / f"ancillary_{y}.tif"
        if not img_p.exists():
            region = region or gee.bbox_geometry(site["bbox"])
            typer.echo(f"[{site['name']}] {y} imagery ...")
            gee.download(gee.s2_composite(region, f"{y}-01-01", f"{y + 1}-01-01", cfg), site["bbox"], img_p, cfg,
                         "uint16", 0, S2_BANDS)
        if not gee.ancillary_ok(anc_p):
            typer.echo(f"[{site['name']}] {y} ancillary + clear-water layers ...")
            gee.download_ancillary(site["bbox"], f"{y}-01-01", f"{y + 1}-01-01", anc_p, cfg)
        if not (d / f"radar_{y}.tif").exists():
            typer.echo(f"[{site['name']}] {y} radar (Sentinel-1) ...")
            gee.download_radar(site["bbox"], f"{y}-01-01", f"{y + 1}-01-01", d / f"radar_{y}.tif", cfg)


@app.command()
def fetch(config: str = CfgOpt, sites: str = SitesOpt, only: list[str] = typer.Option(None, help="Site names"),
          service_account: Path = typer.Option(None, help="Service account key JSON"),
          labels_only: bool = typer.Option(False, help="Re-build labels, keep already-downloaded imagery")):
    """Download Sentinel-2 composites + fused reference labels for every site (Earth Engine)."""
    from . import gee
    from .features import S2_BANDS
    from .labels import postprocess_label_file

    cfg, s = _cfg(config), _sites(sites)
    gee.init(cfg.project, service_account.read_text() if service_account else None)
    year = s["year"]
    for site in s["sites"]:
        if only and site["name"] not in only:
            continue
        d = cfg.work / "sites" / site["name"]
        yr_start, yr_end = f"{year}-01-01", f"{year + 1}-01-01"
        if not gee.ancillary_ok(d / "ancillary.tif") and (d / "image.tif").exists():
            typer.echo(f"[{site['name']}] ancillary + clear-water layers ...")
            gee.download_ancillary(site["bbox"], yr_start, yr_end, d / "ancillary.tif", cfg)
        if site.get("role") != "test":
            _fetch_extra_years(gee, site, d, s.get("extra_years") or [], cfg)
        if (d / "image.tif").exists() and not (d / "radar.tif").exists():  # Sentinel-1, a model input
            typer.echo(f"[{site['name']}] radar (Sentinel-1) ...")
            gee.download_radar(site["bbox"], yr_start, yr_end, d / "radar.tif", cfg)
        if (d / "image.tif").exists() and not (d / "built.tif").exists():  # sea-level-rise screen (small)
            typer.echo(f"[{site['name']}] built-up layer ...")
            gee.download_built(site["bbox"], d / "built.tif", cfg)
        if labels_only and (d / "meta.json").exists() and \
                gee.LABEL_VERSION in json.loads((d / "meta.json").read_text()).get("label_sources", []):
            typer.echo(f"[{site['name']}] labels already up to date, skipping (delete meta.json to redo)")
            continue
        region = gee.bbox_geometry(site["bbox"])
        if not (labels_only and (d / "image.tif").exists()):
            img = gee.s2_composite(region, f"{year}-01-01", f"{year + 1}-01-01", cfg)
            typer.echo(f"[{site['name']}] imagery ...")
            gee.download(img, site["bbox"], d / "image.tif", cfg, "uint16", 0, S2_BANDS)
        if not gee.ancillary_ok(d / "ancillary.tif"):
            typer.echo(f"[{site['name']}] ancillary + clear-water layers ...")
            gee.download_ancillary(site["bbox"], yr_start, yr_end, d / "ancillary.tif", cfg)
        typer.echo(f"[{site['name']}] labels ...")
        info: dict = {}
        gee.download(gee.reference_labels(region, cfg, info, site.get("seagrass_unmapped", False)), site["bbox"], d / "label.tif", cfg, "uint8", 255,
                     ["label"])
        if cfg.labels.seagrass_surveys:
            from .surveys import apply_surveys

            apply_surveys(d / "label.tif", d / "image.tif", site["bbox"], cfg.labels.seagrass_surveys, info,
                          log=typer.echo)
        from .surveys import ignore_deep_seagrass

        if cfg.labels.seagrass_max_depth_m:
            ignore_deep_seagrass(d / "label.tif", d / "ancillary.tif", cfg.labels.seagrass_max_depth_m, typer.echo)
        hist = postprocess_label_file(d / "label.tif", cfg.labels.boundary_ignore_px, cfg.labels.overrides)
        (d / "meta.json").write_text(json.dumps({**site, "year": year, "label_px": hist, **info}, indent=2))
        typer.echo(f"[{site['name']}] label pixels: {hist}  sources: {info.get('label_sources')}")


@app.command()
def chips(config: str = CfgOpt, sites: str = SitesOpt):
    """Tile site rasters into chips with a spatial-block split (test sites fully held out)."""
    from .tiling import make_chips, write_index

    cfg, s = _cfg(config), _sites(sites)
    c = cfg.chips
    out = cfg.work / "chips"
    recs = []
    for site in s["sites"]:
        d = cfg.work / "sites" / site["name"]
        if not (d / "image.tif").exists():
            typer.echo(f"skip {site['name']} (not fetched)")
            continue
        r = make_chips(d / "image.tif", d / "label.tif", out, site["name"], c.size, c.stride, c.min_labeled_frac,
                       c.block_km, c.split, c.seed, "test" if site.get("role") == "test" else None,
                       ancillary_path=d / "ancillary.tif", radar_path=d / "radar.tif")
        n_extra = 0
        if site.get("role") != "test":
            for y in s.get("extra_years") or []:
                if (d / f"image_{y}.tif").exists() and (d / f"ancillary_{y}.tif").exists():
                    rx = make_chips(d / f"image_{y}.tif", d / "label.tif", out, site["name"], c.size, c.stride,
                                    c.min_labeled_frac, c.block_km, c.split, c.seed,
                                    ancillary_path=d / f"ancillary_{y}.tif", tag=str(y), only_splits={"train"},
                                    radar_path=d / f"radar_{y}.tif")
                    n_extra += len(rx)
                    r += rx
        typer.echo(f"{site['name']}: {len(r)} chips ({n_extra} from extra years)")
        recs += r
    write_index(recs, out / "index.json")
    counts = {k: sum(r.split == k for r in recs) for k in ("train", "val", "test")}
    typer.echo(f"total {len(recs)} chips {counts}")


@app.command()
def train(config: str = CfgOpt, kind: str = typer.Option("both", help="spectral | unet | both"),
          epochs: int = typer.Option(None), device: str = typer.Option(None)):
    """Train habitat models on the chip index. With `both`, the better one on validation becomes model/best."""
    import shutil

    from .tiling import read_index

    cfg = _cfg(config)
    if epochs:
        cfg.train.epochs = epochs
    if device:
        cfg.train.device = device
    recs = read_index(cfg.work / "chips" / "index.json")
    out = cfg.work / "model"
    scores = {}
    if kind in ("spectral", "both"):
        from .spectral import train_spectral

        res = train_spectral(recs, out)
        scores["spectral"] = (_blue_score(res.get("val")), out / "spectral.json")
    if kind in ("unet", "both"):
        from .train import train as run

        res = run(cfg, recs, out)
        scores["unet"] = (_blue_score(res.get("val")), out / "model.pt")
    best = max(scores, key=lambda k: scores[k][0])
    src = scores[best][1]
    for stale in out.glob("best.*"):  # never leave a previous run's winner behind
        stale.unlink()
    shutil.copy(src, out / ("best" + src.suffix))
    (out / "best.txt").write_text(f"{best} {src.name} blue-carbon val IoU {scores[best][0]:.3f}\n")
    typer.echo({k: round(v[0], 3) for k, v in scores.items()})
    typer.echo(f"best model: {best} -> {out / ('best' + src.suffix)}")


def _blue_score(summary: dict | None) -> float:
    """Mean validation IoU over the blue carbon classes present (what the product is for)."""
    from .schema import BLUE_CARBON_KEYS

    if not summary:
        return -1.0
    v = [summary["iou"][k] for k in BLUE_CARBON_KEYS if summary["iou"].get(k) is not None]
    return float(sum(v) / len(v)) if v else float(summary.get("mIoU") or -1)


@app.command()
def evaluate(model: Path = typer.Option(..., "--model", "-m"), image: Path = typer.Option(...),
             label: Path = typer.Option(...), out: Path = typer.Option(None)):
    """Score a model against an independent labelled scene (e.g. the Mission Bay hand labels)."""
    import rasterio

    from .metrics import confusion, summarize
    from .predictors import load_predictor

    pr = load_predictor(model)
    with rasterio.open(image) as s:
        bands = s.read()
    with rasterio.open(label) as s:
        ref = s.read(1)
    pred, _ = pr.predict(bands)
    cm = confusion(ref, pred)
    res = {"summary": summarize(cm), "confusion": cm.tolist(), "model": str(model), "image": str(image)}
    if out:
        Path(out).write_text(json.dumps(res, indent=2))
    typer.echo(json.dumps({k: res["summary"][k] for k in ("mIoU", "macro_f1", "iou")}, indent=2))


@app.command()
def predict(image: Path, model: Path = typer.Option(..., "--model", "-m"), out: Path = typer.Option(None),
            config: str = CfgOpt):
    """Predict a habitat map (GeoTIFF: band 1 class, band 2 confidence %). Works with either model type."""
    import numpy as np
    import rasterio

    from .predictors import load_predictor
    from .schema import IGNORE_INDEX

    cfg = _cfg(config)
    pr = load_predictor(model, cfg.train.device)
    out = out or image.with_name(image.stem + "_pred.tif")
    with rasterio.open(image) as src:
        bands, prof = src.read(), src.profile.copy()
    from .predictors import read_ancillary

    anc = read_ancillary(image, bands.shape[1:]) if pr.needs_ancillary else None
    cls, conf = pr.predict(bands, cfg.predict.tile, cfg.predict.overlap, cfg.predict.tta, anc=anc)
    from .priors import apply_to_classes, raster_center_lat

    cls = apply_to_classes(cls, raster_center_lat(prof["transform"], prof["crs"], *cls.shape))
    prof.update(count=2, dtype="uint8", nodata=IGNORE_INDEX, compress="deflate")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **prof) as dst:
        dst.write(cls, 1)
        dst.write(np.round(conf * 100).astype(np.uint8), 2)
        dst.descriptions = ("class", "confidence_pct")
    typer.echo(f"wrote {out}")


@app.command()
def report(prediction: Path, model: Path = typer.Option(None, "--model", "-m",
                                                        help="Adds error-adjusted areas from test confusion"),
           config: str = CfgOpt, out: Path = typer.Option(None)):
    """Area + carbon stock / sequestration report for a prediction raster."""
    from .report import scene_report, to_markdown, write_report

    cfg = _cfg(config)
    cm = None
    if model:
        from .predictors import model_card

        cm = model_card(model).get("metrics", {}).get("test_confusion")
    rep = scene_report(prediction, cfg.carbon, cm)
    write_report(rep, out or prediction.parent, prediction.stem + "_report")
    typer.echo(to_markdown(rep))


@app.command()
def scene(bbox: list[float] = typer.Option(..., help="lon_min lat_min lon_max lat_max"),
          start: str = typer.Option(...), end: str = typer.Option(...),
          model: Path = typer.Option(..., "--model", "-m"), name: str = "scene", config: str = CfgOpt,
          service_account: Path = typer.Option(None)):
    """Fetch any area + date range, map it and report carbon (Earth Engine)."""
    from . import gee
    from .features import S2_BANDS

    cfg = _cfg(config)
    gee.init(cfg.project, service_account.read_text() if service_account else None)
    d = cfg.work / "scenes" / name
    img = gee.s2_composite(gee.bbox_geometry(bbox), start, end, cfg)
    gee.download(img, bbox, d / "image.tif", cfg, "uint16", 0, S2_BANDS)
    gee.download_ancillary(bbox, start, end, d / "ancillary.tif", cfg)
    gee.download_radar(bbox, start, end, d / "radar.tif", cfg)
    predict(d / "image.tif", model, d / "pred.tif", config)
    report(d / "pred.tif", model, config, d)


def case_studies(s: dict) -> list[dict]:
    """The before/after studies in sites.yaml (`case_studies` list, or the older single `case_study`)."""
    return list(s.get("case_studies") or ([s["case_study"]] if s.get("case_study") else []))


@app.command()
def case_study(model: Path = typer.Option(..., "--model", "-m"), config: str = CfgOpt, sites: str = SitesOpt,
               name: list[str] = typer.Option(None, help="Only these studies"),
               service_account: Path = typer.Option(None)):
    """Before/after change analysis for each study in sites.yaml `case_studies`."""
    from . import gee
    from .features import S2_BANDS
    from .report import change_scene_report, write_report

    cfg, s = _cfg(config), _sites(sites)
    gee.init(cfg.project, service_account.read_text() if service_account else None)
    for cs in case_studies(s):
        if name and cs["name"] not in name:
            continue
        typer.echo(f"[{cs['name']}] change study ...")
        d = cfg.work / "scenes" / cs["name"]
        d.mkdir(parents=True, exist_ok=True)
        for key, (a, b) in cs["periods"].items():
            if not (d / f"{key}_image.tif").exists():
                img = gee.s2_composite(gee.bbox_geometry(cs["bbox"]), a, b, cfg)
                gee.download(img, cs["bbox"], d / f"{key}_image.tif", cfg, "uint16", 0, S2_BANDS)
            if not gee.ancillary_ok(d / f"{key}_ancillary.tif"):  # clear-water bands differ per period
                gee.download_ancillary(cs["bbox"], a, b, d / f"{key}_ancillary.tif", cfg)
            if not (d / f"{key}_radar.tif").exists():
                gee.download_radar(cs["bbox"], a, b, d / f"{key}_radar.tif", cfg)
            predict(d / f"{key}_image.tif", model, d / f"{key}_pred.tif", config)
        rep = change_scene_report(d / "t0_pred.tif", d / "t1_pred.tif", cfg.carbon)
        write_report(rep, d, "change_report")
        (d / "periods.json").write_text(json.dumps(cs.get("labels") or {"t0": "Before", "t1": "After"}))
        typer.echo(json.dumps(rep["change"], indent=2))


@app.command()
def export_demo(scene_dir: Path, name: str = typer.Option(...), title: str = typer.Option(...),
                out: Path = typer.Option(Path("demo_data")), model: Path = typer.Option(None, "--model", "-m"),
                config: str = CfgOpt, description: str = ""):
    """Package a predicted scene (image.tif + pred.tif, or t0_/t1_ pairs) for the Streamlit demo."""
    from .demo import export_scene

    p = export_scene(scene_dir, out / name, title, _cfg(config), model, description)
    typer.echo(f"wrote {p}")


@app.command()
def biomass(config: str = CfgOpt, demo: Path = typer.Option(Path("demo_data"), help="Demo pages to summarise"),
            out: Path = typer.Option(Path("data/site_biomass.json")),
            service_account: Path = typer.Option(None, help="Service account key JSON")):
    """Measured mangrove biomass + rainfall for every demo page (seconds per site, no imagery download).

    Writes data/site_biomass.json; scripts/refresh_carbon.py then uses it for the carbon numbers."""
    from . import gee

    cfg = _cfg(config)
    gee.init(cfg.project, service_account.read_text() if service_account else None)
    res = json.loads(out.read_text()) if out.exists() else {}
    for mp in sorted(demo.glob("*/meta.json")):
        (s, w), (n, e) = json.loads(mp.read_text())["bounds"]
        try:
            res[mp.parent.name] = gee.site_biomass_stats([w, s, e, n])
        except Exception as ex:  # one site failing must not lose the others
            typer.echo(f"{mp.parent.name}: skipped ({str(ex)[:100]})")
            continue
        r = res[mp.parent.name]
        typer.echo(f"{mp.parent.name:24s} mangrove biomass {r['agb_mean'] or 0:7.1f} Mg/ha over {r['agb_count']:>8,} px, "
                   f"rain {r['precip_mm'] or 0:6.0f} mm")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, sort_keys=True))
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
