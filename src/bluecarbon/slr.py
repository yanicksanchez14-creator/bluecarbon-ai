"""Sea-level-rise exposure for a site screen: can the wetland move inland as the sea rises?

Verra VM0033 v2.1 asks every tidal wetland project to rate its sea-level-rise risk, and lets the project area
move inland over time. A wetland backed by low, open land can migrate; one pressed against a town or a steep
shore is squeezed out ("coastal squeeze"). This screen measures that room:

  room to migrate = land within SEARCH_KM of mangrove / salt marsh that is not already wetland, not built up
                    (ESA WorldCover class 50) and between 0 and MAX_ELEV_M above sea level (NASADEM)

and rates it against the wetland area itself. NASADEM measures the top of vegetation, so wooded land reads
higher than the ground: the room is a conservative (low) estimate.
"""

from __future__ import annotations

import numpy as np

from .carbon import pixel_area_ha
from .schema import KEY_TO_ID

MAX_ELEV_M = 2.0      # land up to 2 m above sea level: reachable this century at the upper IPCC projections
SEARCH_KM = 2.0       # how far behind the wetland we look
RATINGS = ((0.5, "low"), (0.1, "medium"))  # room / wetland ratio at or above -> rating; below the last -> high

# IPCC AR6 WG1, Summary for Policymakers B.5.3: global mean sea level rise by 2100 vs 1995-2014,
# likely range 0.28-0.55 m (very low emissions) to 0.63-1.01 m (very high emissions).
AR6_2100_M = (0.28, 1.01)


def slr_exposure(cls: np.ndarray, elevation: np.ndarray, built: np.ndarray | None, transform, crs) -> dict:
    from scipy.ndimage import distance_transform_edt

    wet = np.isin(cls, [KEY_TO_ID["mangrove"], KEY_TO_ID["saltmarsh"]])
    row_ha = pixel_area_ha(transform, crs, cls.shape[0])[:, None]
    wet_ha = float((wet * row_ha).sum())
    out = {"wetland_ha": round(wet_ha, 1), "max_elev_m": MAX_ELEV_M, "search_km": SEARCH_KM,
           "ar6_2100_m": list(AR6_2100_M)}
    if wet_ha < 1:
        return {**out, "room_ha": 0.0, "ratio": None, "rating": "not applicable"}
    px_m = abs(transform.a) if crs is not None and crs.is_projected else abs(transform.a) * 111_320
    near = distance_transform_edt(~wet) * px_m <= SEARCH_KM * 1000
    land = np.isin(cls, [KEY_TO_ID["other_land"], KEY_TO_ID["freshwater"]])
    low = (elevation > 0) & (elevation <= MAX_ELEV_M)
    room = near & ~wet & land & low
    if built is not None:
        room &= built == 0
    room_ha = float((room * row_ha).sum())
    ratio = room_ha / wet_ha
    rating = next((r for t, r in RATINGS if ratio >= t), "high")
    return {**out, "room_ha": round(room_ha, 1), "ratio": round(ratio, 3), "rating": rating}


def slr_from_files(pred_path, ancillary_path, built_path=None) -> dict | None:
    """Exposure from a prediction and its ancillary.tif (+ built.tif); None when the layers are missing."""
    from pathlib import Path

    import rasterio

    if not Path(ancillary_path).exists():
        return None
    with rasterio.open(pred_path) as p:
        cls, transform, crs = p.read(1), p.transform, p.crs
    with rasterio.open(ancillary_path) as a:
        if a.shape != cls.shape:
            return None
        elev = a.read(1).astype(float)
    built = None
    if built_path is not None and Path(built_path).exists():
        with rasterio.open(built_path) as b:
            if b.shape == cls.shape:
                built = b.read(1)
    return slr_exposure(cls, elev, built, transform, crs)
