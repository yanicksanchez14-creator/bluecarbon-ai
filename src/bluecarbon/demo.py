"""Package predicted scenes as lightweight PNG + JSON bundles for the web demo.

Rasters are warped to Web Mercator so they sit exactly on Leaflet basemaps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

from .carbon import class_areas_ha
from .config import Config
from .report import change_scene_report, scene_report
from .schema import BLUE_CARBON_KEYS, CLASSES, IGNORE_INDEX, KEY_TO_ID
from .viz import change_rgba, class_rgba, confidence_rgba, save_png, true_color

MAX_PX = 1600


def _to_mercator(path: Path, band_idx: list[int], resampling, nodata):
    with rasterio.open(path) as src:
        dst_crs = "EPSG:3857"
        tr, w, h = calculate_default_transform(src.crs, dst_crs, src.width, src.height, *src.bounds)
        scale = max(1.0, max(w, h) / MAX_PX)
        w2, h2 = int(w / scale), int(h / scale)
        tr = tr @ rasterio.Affine.scale(w / w2, h / h2)
        out = np.full((len(band_idx), h2, w2), nodata, dtype=src.dtypes[0])
        for i, b in enumerate(band_idx):
            reproject(rasterio.band(src, b), out[i], src_transform=src.transform, src_crs=src.crs,
                      dst_transform=tr, dst_crs=dst_crs, resampling=resampling, src_nodata=src.nodata,
                      dst_nodata=nodata)
        bounds = rasterio.transform.array_bounds(h2, w2, tr)  # (west, south, east, north) in 3857
        west, south, east, north = transform_bounds(dst_crs, "EPSG:4326", *bounds)
    return out, [[south, west], [north, east]]


def _render_period(d: Path, prefix: str, out: Path, tag: str) -> tuple[list, np.ndarray]:
    img = d / f"{prefix}image.tif"
    pred = d / f"{prefix}pred.tif"
    with rasterio.open(img) as src:
        n = src.count
    bands, bounds = _to_mercator(img, list(range(1, n + 1)), Resampling.bilinear, 0)
    save_png(true_color(bands), out / f"rgb{tag}.png")
    save_png(true_color(bands, rgb_bands=("B8", "B4", "B3"), gamma=1.0), out / f"falsecolor{tag}.png")
    cls, _ = _to_mercator(pred, [1], Resampling.nearest, IGNORE_INDEX)
    save_png(class_rgba(cls[0], 200), out / f"classes{tag}.png")
    save_png(class_rgba(cls[0], 220, [KEY_TO_ID[k] for k in BLUE_CARBON_KEYS]), out / f"bluecarbon{tag}.png")
    with rasterio.open(pred) as p:
        has_conf = p.count >= 2
    if has_conf:
        conf, _ = _to_mercator(pred, [2], Resampling.nearest, 255)
        save_png(confidence_rgba(conf[0], cls[0]), out / f"confidence{tag}.png")
    return bounds, cls[0]


def confidence_summary(pred_path) -> dict | None:
    """How sure the model is about each blue carbon habitat it mapped: mean confidence and the share of that
    habitat's area mapped with under 60% confidence. Softmax confidence, not yet calibrated on held-out data."""
    with rasterio.open(pred_path) as p:
        if p.count < 2:
            return None
        cls, conf = p.read(1), p.read(2).astype(float)
    out = {}
    for k in BLUE_CARBON_KEYS:
        m = cls == KEY_TO_ID[k]
        if m.sum() < 25:
            continue
        c = conf[m]
        out[k] = {"mean_pct": round(float(c.mean()), 1), "low_share": round(float((c < 60).mean()), 3)}
    return out


def export_scene(scene_dir: str | Path, out_dir: str | Path, title: str, cfg: Config,
                 model_path: str | Path | None = None, description: str = "") -> Path:
    d, out = Path(scene_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cm, model_info = None, None
    if model_path:
        from .predictors import model_card

        ck = model_card(model_path)
        cm = ck.get("metrics", {}).get("test_confusion")
        model_info = {"arch": ck["arch"], "encoder": ck["encoder"], "kind": ck["kind"],
                      "test": ck.get("metrics", {}).get("test"), **ck.get("extra", {})}
    meta = {"title": title, "description": description, "classes": [c.__dict__ for c in CLASSES],
            "model": model_info}
    if (d / "t0_pred.tif").exists():
        b0, c0 = _render_period(d, "t0_", out, "_t0")
        _, c1 = _render_period(d, "t1_", out, "_t1")
        save_png(change_rgba(c0, c1, [KEY_TO_ID[k] for k in BLUE_CARBON_KEYS]), out / "change.png")
        meta.update(kind="change", bounds=b0, change=change_scene_report(d / "t0_pred.tif", d / "t1_pred.tif", cfg.carbon),
                    t0=scene_report(d / "t0_pred.tif", cfg.carbon, cm), t1=scene_report(d / "t1_pred.tif", cfg.carbon, cm))
        meta["periods"] = json.loads((d / "periods.json").read_text()) if (d / "periods.json").exists() else None
    else:
        b, _ = _render_period(d, "", out, "")
        meta.update(kind="single", bounds=b, report=scene_report(d / "pred.tif", cfg.carbon, cm))
        cs = confidence_summary(d / "pred.tif")
        if cs:
            meta["report"]["confidence"] = cs
        from .slr import slr_from_files

        sl = slr_from_files(d / "pred.tif", d / "ancillary.tif", d / "built.tif")
        if sl:
            meta["sea_level"] = sl
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out


__all__ = ["export_scene", "class_areas_ha"]
