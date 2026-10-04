"""Blue carbon accounting with Monte Carlo uncertainty.

Soil carbon uses measured soil cores near the site when there are enough (Smithsonian Coastal Carbon
Library, see local_soil), and the IPCC Tier 1 global default otherwise.

Stocks = area x (soil C to 1 m + living biomass C). Sequestration = area x soil accumulation
rate. Each coefficient is sampled from a triangular(low, mode, high) distribution, and area
can optionally carry its own uncertainty (e.g. from the error-adjusted estimator), so the
reported numbers are ranges, not false-precision point values.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import CarbonCfg
from .schema import BLUE_CARBON_KEYS, CLASS_KEYS, IGNORE_INDEX, N_CLASSES


def pixel_area_ha(transform, crs, height: int) -> np.ndarray:
    """Per-row pixel area in hectares (handles projected and geographic CRSs)."""
    a, e = abs(transform.a), abs(transform.e)
    if crs is not None and crs.is_projected:
        return np.full(height, a * e / 1e4)
    lat = transform.f + transform.e * (np.arange(height) + 0.5)
    r = 6371008.8
    dlat, dlon = math.radians(e), math.radians(a)
    return (r**2 * dlon * np.abs(np.sin(np.radians(lat) + dlat / 2) - np.sin(np.radians(lat) - dlat / 2))) / 1e4


def class_areas_ha(class_map: np.ndarray, transform, crs) -> dict[str, float]:
    row_area = pixel_area_ha(transform, crs, class_map.shape[0])
    out = {}
    for i, k in enumerate(CLASS_KEYS):
        out[k] = float(((class_map == i).sum(1) * row_area).sum())
    return out


def class_pixel_counts(class_map: np.ndarray) -> np.ndarray:
    v = class_map[class_map != IGNORE_INDEX]
    return np.bincount(v.ravel(), minlength=N_CLASSES)[:N_CLASSES]


SOIL_CORES = Path(__file__).resolve().parents[2] / "data" / "soil_carbon_cores.csv"
SOIL_RADII_KM = (100, 300)   # look for cores within 100 km, then 300 km
SOIL_MIN_CORES = 8           # fewer measured cores than this -> IPCC default


@lru_cache(maxsize=1)
def _cores():
    if not SOIL_CORES.exists():
        return None
    import csv

    with open(SOIL_CORES, newline="") as f:
        rows = list(csv.DictReader(f))
    return {
        "habitat": np.array([r["habitat"] for r in rows]),
        "lat": np.array([float(r["latitude"]) for r in rows]),
        "lon": np.array([float(r["longitude"]) for r in rows]),
        "soc": np.array([float(r["soc_1m_tC_ha"]) for r in rows]),
        "study": np.array([r["study_id"] for r in rows]),
    }


def _km(lat0, lon0, lat, lon):
    p0, p = np.radians(lat0), np.radians(lat)
    a = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin(np.radians(lon - lon0) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def local_soil(lat: float, lon: float) -> dict[str, dict]:
    """Measured soil carbon to 1 m (t C/ha) near (lat, lon), per blue carbon habitat.

    Returns {habitat: {"soil": (low, mean, high), "n_cores", "n_studies", "radius_km"}} for habitats with
    at least SOIL_MIN_CORES cores within SOIL_RADII_KM. `mean` is the 5-95% winsorized mean of the cores
    (the expected value per hectare); low / high bound the uncertainty of that mean, using the number of
    independent studies (not cores) because cores from one study are not independent.
    """
    c = _cores()
    if c is None:
        return {}
    dist = _km(lat, lon, c["lat"], c["lon"])
    out = {}
    for hab in BLUE_CARBON_KEYS:
        for r in SOIL_RADII_KM:
            m = (c["habitat"] == hab) & (dist <= r)
            if m.sum() < SOIL_MIN_CORES:
                continue
            v = c["soc"][m]
            lo5, hi95 = np.percentile(v, [5, 95])
            w = np.clip(v, lo5, hi95)
            mean, sd = float(w.mean()), float(w.std(ddof=1))
            k = len(set(c["study"][m]))
            half = 1.96 * sd / np.sqrt(k)
            p10, p90 = np.percentile(v, [10, 90])
            out[hab] = {"soil": (float(max(p10, mean - half)), mean, float(min(p90, mean + half))),
                        "n_cores": int(m.sum()), "n_studies": k, "radius_km": r}
            break
    return out


def soil_sources(cfg: CarbonCfg, lat: float | None, lon: float | None) -> dict[str, dict]:
    """Soil coefficients actually used per habitat, with where they came from."""
    local = local_soil(lat, lon) if lat is not None and lon is not None else {}
    out = {}
    for k in BLUE_CARBON_KEYS:
        if k in local:
            out[k] = {**local[k], "source": "measured"}
        elif k in cfg.classes:
            out[k] = {"soil": tuple(cfg.classes[k].soil), "source": "ipcc"}
    return out


def _tri(rng, lmh, n):
    lo, mo, hi = lmh
    if hi <= lo:
        return np.full(n, mo, float)
    return rng.triangular(lo, min(max(mo, lo), hi), hi, n)


def _q(v: np.ndarray) -> dict:
    return {"mean": float(np.mean(v)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def carbon_report(areas_ha: dict[str, float], cfg: CarbonCfg, area_sd_ha: dict[str, float] | None = None,
                  seed: int = 0, lat: float | None = None, lon: float | None = None) -> dict:
    rng = np.random.default_rng(seed)
    soils = soil_sources(cfg, lat, lon)
    n = cfg.monte_carlo
    total_stock = np.zeros(n)
    total_seq = np.zeros(n)
    per = {}
    for k in BLUE_CARBON_KEYS:
        c = cfg.classes.get(k)
        a = areas_ha.get(k, 0.0)
        if c is None or a <= 0:
            continue
        sd = (area_sd_ha or {}).get(k, 0.0)
        area = np.clip(rng.normal(a, sd, n), 0, None) if sd > 0 else np.full(n, a)
        stock_c = area * (_tri(rng, soils[k]["soil"], n) + _tri(rng, c.biomass, n))
        seq_c = area * _tri(rng, c.accumulation, n)
        total_stock += stock_c
        total_seq += seq_c
        per[k] = {
            "area_ha": a,
            "stock_tC": _q(stock_c),
            "stock_tCO2e": _q(stock_c * cfg.co2_per_c),
            "sequestration_tCO2e_per_yr": _q(seq_c * cfg.co2_per_c),
            "soil": soils[k],
        }
    stock_co2 = total_stock * cfg.co2_per_c
    seq_co2 = total_seq * cfg.co2_per_c
    lo, mid, hi = cfg.price_usd_per_tco2e
    return {
        "classes": per,
        "blue_carbon_area_ha": float(sum(areas_ha.get(k, 0.0) for k in BLUE_CARBON_KEYS)),
        "total_stock_tCO2e": _q(stock_co2),
        "total_sequestration_tCO2e_per_yr": _q(seq_co2),
        # Credits are issued for additional sequestration / avoided emissions, not standing stock,
        # so the indicative value is based on annual sequestration only.
        "indicative_annual_value_usd": {"low": float(np.mean(seq_co2) * lo), "mid": float(np.mean(seq_co2) * mid),
                                        "high": float(np.mean(seq_co2) * hi)},
        "method": _method(per, n),
    }


def _method(per: dict, n: int) -> str:
    measured = [k for k, v in per.items() if v["soil"]["source"] == "measured"]
    if not measured:
        return f"IPCC 2013 Wetlands Supplement Tier 1; triangular Monte Carlo, n={n}"
    return (f"Soil carbon from measured soil cores near the site (Smithsonian Coastal Carbon Library) for "
            f"{', '.join(measured)}, IPCC Tier 1 otherwise; biomass and burial rates IPCC Tier 1; "
            f"triangular Monte Carlo, n={n}")


def change_report(areas_t0: dict[str, float], areas_t1: dict[str, float], cfg: CarbonCfg, seed: int = 0,
                  lat: float | None = None, lon: float | None = None) -> dict:
    """Area change per class and the carbon implication of blue carbon gain/loss."""
    rng = np.random.default_rng(seed)
    soils = soil_sources(cfg, lat, lon)
    n = cfg.monte_carlo
    delta = {k: areas_t1.get(k, 0.0) - areas_t0.get(k, 0.0) for k in CLASS_KEYS}
    net = np.zeros(n)
    per = {}
    for k in BLUE_CARBON_KEYS:
        c = cfg.classes.get(k)
        d = delta.get(k, 0.0)
        if c is None or d == 0:
            continue
        v = d * (_tri(rng, soils[k]["soil"], n) + _tri(rng, c.biomass, n)) * cfg.co2_per_c
        net += v
        per[k] = {"delta_ha": d, "stock_change_tCO2e": _q(v)}
    return {"delta_ha": delta, "classes": per, "net_stock_change_tCO2e": _q(net)}
