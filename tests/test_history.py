"""Landsat history model: trains on labels resampled to 30 m, scores held-out sites, gives areas per epoch."""

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from bluecarbon.schema import KEY_TO_ID


def _site(d, seed):
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    H = W = 60
    lab = np.full((H * 3, W * 3), KEY_TO_ID["water"], np.uint8)
    lab[:, 60:] = KEY_TO_ID["mangrove"]
    lab[:, 120:] = KEY_TO_ID["other_land"]
    with rasterio.open(d / "label.tif", "w", driver="GTiff", width=W * 3, height=H * 3, count=1, dtype="uint8",
                       nodata=255, transform=from_origin(400000, 3000000, 10, 10), crs=CRS.from_epsg(32617)) as ds:
        ds.write(lab, 1)
    spec = {"water": [300, 500, 400, 200, 100, 80], "mangrove": [200, 500, 300, 3500, 1500, 600],
            "other_land": [900, 1100, 1300, 2500, 3000, 2500]}
    col = np.zeros((6, H, W), np.uint16)
    for j, k in enumerate(["water", "mangrove", "other_land"]):
        col[:, :, j * 20:(j + 1) * 20] = np.array(spec[k])[:, None, None]
    for y in (2021, 1990, 2020):
        img = np.clip(col + rng.normal(0, 40, col.shape), 1, None).astype(np.uint16)
        with rasterio.open(d / f"landsat_{y}.tif", "w", driver="GTiff", width=W, height=H, count=6, dtype="uint16",
                           nodata=0, transform=from_origin(400000, 3000000, 30, 30), crs=CRS.from_epsg(32617)) as ds:
            ds.write(img)


def test_history_train_and_run(tmp_path):
    pytest.importorskip("lightgbm")
    from bluecarbon.history import load, site_history, train

    for i, n in enumerate(["a", "b", "t"]):
        _site(tmp_path / n, i)
    m = train(tmp_path, ["a", "b"], ["t"], tmp_path / "h.json")
    assert m["test"]["iou"]["mangrove"] > 0.9
    booster, metrics = load(tmp_path / "h.json")
    h = site_history(booster, metrics, tmp_path / "t")
    years = [r["year"] for r in h["epochs"] if "areas_ha" in r]
    assert years == [1990, 2020]
    assert abs(h["epochs"][0]["areas_ha"]["mangrove"] - 60 * 20 * 0.09) < 5   # 20 columns of 30 m pixels
