"""Synthetic Sentinel-2-like scenes so the whole pipeline is testable offline in CI."""

from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from bluecarbon.features import S2_BANDS
from bluecarbon.schema import IGNORE_INDEX, N_CLASSES

# Rough per-class reflectance signatures for B2..B12 (x1e4)
SIGNATURES = np.array([
    [600, 700, 500, 450, 400, 380, 300, 280, 150, 100],        # water
    [300, 500, 300, 900, 2500, 3000, 3200, 3300, 1500, 700],   # mangrove
    [500, 700, 600, 1100, 1800, 2100, 2300, 2400, 2000, 1300],  # salt marsh
    [400, 600, 450, 500, 700, 750, 800, 780, 250, 150],        # seagrass
    [1200, 1400, 1600, 1700, 1800, 1850, 1900, 1950, 2400, 2000],  # tidal flat
    [900, 1100, 1300, 1500, 1900, 2100, 2300, 2400, 2800, 2300],  # other land
    [350, 550, 400, 800, 1500, 1800, 2000, 2100, 900, 400],     # freshwater wetland
], np.float32)


def make_scene(path_img, path_lab, size=512, seed=0, crs="EPSG:32611"):
    rng = np.random.default_rng(seed)
    lab = np.zeros((size, size), np.uint8)
    # blocky random layout
    cell = 64
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            lab[y : y + cell, x : x + cell] = rng.integers(0, N_CLASSES)
    img = SIGNATURES[lab].transpose(2, 0, 1) * (1 + rng.normal(0, 0.05, (len(S2_BANDS), size, size)))
    img = np.clip(img, 1, 10000).astype(np.uint16)
    lab_noisy = lab.copy()
    lab_noisy[rng.random((size, size)) < 0.02] = IGNORE_INDEX
    tr = from_origin(480000, 3630000, 10, 10)
    prof = dict(driver="GTiff", width=size, height=size, crs=crs, transform=tr)
    with rasterio.open(path_img, "w", count=len(S2_BANDS), dtype="uint16", nodata=0, **prof) as d:
        d.write(img)
    with rasterio.open(path_lab, "w", count=1, dtype="uint8", nodata=255, **prof) as d:
        d.write(lab_noisy, 1)
    return img, lab


@pytest.fixture
def scene(tmp_path):
    img_p, lab_p = tmp_path / "image.tif", tmp_path / "label.tif"
    img, lab = make_scene(img_p, lab_p)
    return img_p, lab_p, img, lab


def write_ancillary(path_img, lab, legacy: bool = False):
    """Synthetic ancillary.tif next to an image: tide high over marsh/flats, low over freshwater."""
    with rasterio.open(path_img) as src:
        prof = src.profile.copy()
    h, w = lab.shape
    elev = np.where(np.isin(lab, [0, 3]), 0, 3).astype(np.int16)
    tidal = np.where(np.isin(lab, [1, 2, 4]), 90, 5).astype(np.int16)
    lat = np.full((h, w), 2500, np.int16)
    with rasterio.open(path_img) as src:
        clear = src.read([1, 2, 3, 7]).astype(np.int16)  # B2, B3, B4, B8
    depth = np.where(np.isin(lab, [0, 3]), 3, 0).astype(np.int16)
    layers = [np.stack([elev, tidal, lat]), clear] + ([depth[None]] if not legacy else [])
    prof.update(count=sum(x.shape[0] for x in layers), dtype="int16", nodata=None)
    out = path_img.parent / "ancillary.tif"
    with rasterio.open(out, "w", **prof) as d:
        d.write(np.concatenate(layers))
    return out
