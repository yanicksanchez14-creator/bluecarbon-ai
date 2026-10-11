"""Habitat history from the Landsat archive (1985 to today), for trend lines and avoided-loss claims.

Sentinel-2 only starts in 2015, so history uses Landsat 5, 7, 8 and 9 (30 m). A separate, simpler model
(LightGBM on six harmonised reflectance bands + four indices + a little local texture) is trained on the
same reference labels as the main model, resampled to 30 m, using a three-year composite centred on the
label year. It is scored on the same five held-out estuaries and its accuracy is published separately:
it is less accurate than the 10 m model and is meant for trends, not for areas on its own.

Each epoch is a three-year median composite (e.g. 1990 = 1989-1991) so Landsat 7's striped images and
cloudy years still give a complete picture. Areas are corrected with the history model's own held-out
confusion matrix (Olofsson et al. 2014), like the main maps.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import BLUE_CARBON_KEYS, CLASS_KEYS, IGNORE_INDEX, N_CLASSES

LANDSAT_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
EPOCHS = [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2025]
LABEL_YEAR = 2021
SCALE_M = 30
INDICES = {"NDVI": ("nir", "red"), "NDWI": ("green", "nir"), "MNDWI": ("green", "swir1"), "NDMI": ("nir", "swir1")}
FEATURES = LANDSAT_BANDS + list(INDICES) + ["NDVI_MEAN5", "MNDWI_MEAN5"]
PER_CLASS_PER_SITE = 3000


def landsat_features(bands: np.ndarray) -> np.ndarray:
    """(6, H, W) uint16 reflectance x 1e4 -> (len(FEATURES), H, W) float32."""
    from scipy.ndimage import uniform_filter

    x = np.nan_to_num(bands.astype(np.float32)) / 1e4
    b = dict(zip(LANDSAT_BANDS, x, strict=True))
    f = list(x)
    for a, c in INDICES.values():
        den = b[a] + b[c]
        nd = np.zeros_like(den)
        np.divide(b[a] - b[c], den, out=nd, where=np.abs(den) > 1e-6)
        f.append(np.clip(nd, -1, 1))
    f.append(uniform_filter(f[len(LANDSAT_BANDS)], 5))        # NDVI texture
    f.append(uniform_filter(f[len(LANDSAT_BANDS) + 2], 5))    # MNDWI texture
    return np.stack(f).astype(np.float32)


def valid(bands: np.ndarray) -> np.ndarray:
    return (bands > 0).any(0) & np.isfinite(bands).all(0)


def label_on_grid(label_path, like_path) -> np.ndarray:
    """10 m reference labels resampled to the Landsat grid (most common class per 30 m pixel)."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(like_path) as ref:
        out = np.full((ref.height, ref.width), IGNORE_INDEX, np.uint8)
        with rasterio.open(label_path) as lab:
            reproject(rasterio.band(lab, 1), out, src_transform=lab.transform, src_crs=lab.crs,
                      dst_transform=ref.transform, dst_crs=ref.crs, resampling=Resampling.mode,
                      src_nodata=IGNORE_INDEX, dst_nodata=IGNORE_INDEX)
    return out


def _read(path) -> tuple[np.ndarray, object, object]:
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read(), ds.transform, ds.crs


def training_pixels(sites_dir: Path, names: list[str], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for n in names:
        d = sites_dir / n
        img, lab = d / f"landsat_{LABEL_YEAR}.tif", d / "label.tif"
        if not (img.exists() and lab.exists()):
            continue
        bands, _, _ = _read(img)
        y = label_on_grid(lab, img)
        x = landsat_features(bands)
        ok = (y != IGNORE_INDEX) & valid(bands)
        for c in range(N_CLASSES):
            idx = np.flatnonzero(ok.ravel() & (y.ravel() == c))
            if idx.size:
                pick = idx if idx.size <= PER_CLASS_PER_SITE else rng.choice(idx, PER_CLASS_PER_SITE, replace=False)
                xs.append(x.reshape(x.shape[0], -1)[:, pick].T)
                ys.append(np.full(pick.size, c, np.int64))
    if not xs:
        raise RuntimeError("no Landsat training pixels: run `bluecarbon history-fetch` first")
    return np.concatenate(xs), np.concatenate(ys)


def train(sites_dir: Path, train_names: list[str], test_names: list[str], out_path: Path) -> dict:
    import lightgbm as lgb

    from .metrics import confusion, summarize

    X, y = training_pixels(sites_dir, train_names)
    cnt = np.bincount(y, minlength=N_CLASSES).astype(float)
    w = (1 / np.sqrt(np.maximum(cnt, 1)))[y]
    params = dict(objective="multiclass", num_class=N_CLASSES, learning_rate=0.05, num_leaves=63,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=40, verbose=-1)
    booster = lgb.train(params, lgb.Dataset(X, y, weight=w), num_boost_round=400)
    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    per_site = {}
    for n in test_names:
        d = sites_dir / n
        img = d / f"landsat_{LABEL_YEAR}.tif"
        if not img.exists():
            continue
        bands, _, _ = _read(img)
        yt = label_on_grid(d / "label.tif", img)
        pred = _predict(booster, bands)
        ok = (yt != IGNORE_INDEX) & valid(bands)
        c = confusion(np.where(ok, yt, IGNORE_INDEX), pred, N_CLASSES)
        cm += c
        per_site[n] = summarize(c)
    metrics = {"test": summarize(cm), "test_confusion": cm.tolist(), "test_per_site": per_site,
               "train_pixels": int(len(y)), "label_year": LABEL_YEAR, "scale_m": SCALE_M}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"features": FEATURES, "booster": booster.model_to_string(), "metrics": metrics}))
    return metrics


def load(path: Path):
    import lightgbm as lgb

    d = json.loads(Path(path).read_text())
    return lgb.Booster(model_str=d["booster"]), d["metrics"]


def _predict(booster, bands: np.ndarray) -> np.ndarray:
    x = landsat_features(bands)
    n, h, w = x.shape
    p = booster.predict(x.reshape(n, -1).T)
    cls = p.argmax(1).astype(np.uint8).reshape(h, w)
    cls[~valid(bands)] = IGNORE_INDEX
    return cls


def site_history(booster, metrics: dict, d: Path) -> dict | None:
    """Blue carbon area per epoch for one site (error-adjusted with the history model's confusion)."""
    from .carbon import class_areas_ha, class_pixel_counts, pixel_area_ha
    from .metrics import error_adjusted_area
    from .priors import apply_to_classes, apply_to_confusion, raster_center_lonlat

    rows = []
    for year in EPOCHS:
        p = d / f"landsat_{year}.tif"
        if not p.exists():
            continue
        bands, transform, crs = _read(p)
        ok = valid(bands)
        if ok.mean() < 0.6:  # too much of the site missing (no images that era): skip, don't guess
            rows.append({"year": year, "coverage": round(float(ok.mean()), 2)})
            continue
        cls = _predict(booster, bands)
        lon, lat = raster_center_lonlat(transform, crs, *cls.shape)
        cls = apply_to_classes(cls, lat)
        areas = class_areas_ha(cls, transform, crs)
        px_ha = float(np.mean(pixel_area_ha(transform, crs, cls.shape[0])))
        adj = error_adjusted_area(apply_to_confusion(np.array(metrics["test_confusion"]), lat), class_pixel_counts(cls),
                                  px_ha)
        rows.append({"year": year, "coverage": round(float(ok.mean()), 2),
                     "areas_ha": {k: round(areas[k], 1) for k in CLASS_KEYS},
                     "adjusted_ha": {k: {"ha": round(adj[k]["adjusted_ha"], 1), "ci95": round(adj[k]["ci95_ha"], 1)}
                                     for k in BLUE_CARBON_KEYS if k in adj}})
    if not any("areas_ha" in r for r in rows):
        return None
    t = metrics["test"]
    return {"epochs": rows, "scale_m": SCALE_M, "composite": "3-year median, Landsat 5/7/8/9, harmonised",
            "model": {"iou": t.get("iou"), "f1": t.get("f1"), "miou": t.get("mIoU")}}
