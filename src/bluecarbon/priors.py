"""Ecological range priors applied after the model.

Mangroves are frost-intolerant and only occur between roughly 39°S (Corner Inlet, Australia) and
32.5°N (Bermuda / southern Japan). Outside that band any 'mangrove' prediction is dense tidal
vegetation the model has confused with mangrove, so it is reassigned to salt marsh, and the
area-correction step is not allowed to move area into mangrove either.
"""

from __future__ import annotations

import numpy as np

from .schema import KEY_TO_ID, N_CLASSES, pad_confusion

MANGROVE_LAT_RANGE = (-39.0, 32.5)


def mangrove_possible(lat: float | None) -> bool:
    return lat is None or MANGROVE_LAT_RANGE[0] <= lat <= MANGROVE_LAT_RANGE[1]


def apply_to_classes(cls: np.ndarray, lat: float | None) -> np.ndarray:
    if mangrove_possible(lat):
        return cls
    out = cls.copy()
    out[out == KEY_TO_ID["mangrove"]] = KEY_TO_ID["saltmarsh"]
    return out


def apply_to_confusion(cm, lat: float | None) -> np.ndarray:
    """Fold the mangrove row and column into salt marsh so no area is attributed to mangrove."""
    cm = pad_confusion(cm).astype(np.float64).copy()
    if mangrove_possible(lat):
        return cm
    m, s = KEY_TO_ID["mangrove"], KEY_TO_ID["saltmarsh"]
    cm[s, :] += cm[m, :]
    cm[m, :] = 0
    cm[:, s] += cm[:, m]
    cm[:, m] = 0
    assert cm.shape == (N_CLASSES, N_CLASSES)
    return cm


def raster_center_lat(transform, crs, height: int, width: int) -> float:
    from rasterio.warp import transform as warp

    x, y = transform * (width / 2, height / 2)
    _, lat = warp(crs, "EPSG:4326", [x], [y])
    return float(lat[0])


def raster_center_lonlat(transform, crs, height: int, width: int) -> tuple[float, float]:
    from rasterio.warp import transform as warp

    x, y = transform * (width / 2, height / 2)
    lon, lat = warp(crs, "EPSG:4326", [x], [y])
    return float(lon[0]), float(lat[0])
