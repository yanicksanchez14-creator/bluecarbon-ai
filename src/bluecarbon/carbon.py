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


# Measured mangrove biomass (Simard et al. 2019, NASA): aboveground biomass from canopy height, per pixel.
# Converted to carbon with the IPCC 2013 Wetlands Supplement factors (Table 4.2 carbon fraction, Table 4.5
# below- to above-ground ratio by climate zone, each as its 95% interval).
CARBON_FRACTION = (0.429, 0.451, 0.471)
ROOT_SHOOT = {"tropical wet": (0.47, 0.49, 0.51), "tropical dry": (0.28, 0.29, 0.30), "subtropical": (0.91, 0.96, 1.0)}
# The Simard map publishes no per-pixel error. Until our own field plots exist we carry +/-30% on the site
# mean (the 12 m TanDEM-X successor reports 2.4 m height RMSE, about 20-30% of typical mangrove height).
AGB_MAP_REL_ERROR = 0.30
MIN_AGB_PIXELS = 50          # fewer measured pixels than this (~4.5 ha) -> IPCC default


def climate_zone(lat: float, precip_mm: float | None) -> str:
    """IPCC climate zone for mangrove root ratios: subtropical outside the tropics, tropical dry under 1,000 mm/yr."""
    if abs(lat) > 23.44:
        return "subtropical"
    return "tropical dry" if precip_mm is not None and precip_mm < 1000 else "tropical wet"


def measured_biomass(stats: dict | None, lat: float | None, cfg: CarbonCfg) -> dict[str, dict]:
    """Biomass coefficients per habitat (t C/ha, low/mean/high) with their source.

    stats = {"agb_mean": Mg/ha, "agb_count": pixels, "precip_mm": ...} from gee.site_biomass_stats, or None.
    Mangrove uses the measured map when it covers the site; everything else stays IPCC Tier 1."""
    out = {k: {"biomass": tuple(cfg.classes[k].biomass), "source": "ipcc"} for k in BLUE_CARBON_KEYS if k in cfg.classes}
    if not stats or lat is None or (stats.get("agb_count") or 0) < MIN_AGB_PIXELS or not stats.get("agb_mean"):
        return out
    zone = climate_zone(lat, stats.get("precip_mm"))
    agb, e = float(stats["agb_mean"]), AGB_MAP_REL_ERROR
    r, cf = ROOT_SHOOT[zone], CARBON_FRACTION
    out["mangrove"] = {
        "biomass": (agb * (1 - e) * (1 + r[0]) * cf[0], agb * (1 + r[1]) * cf[1], agb * (1 + e) * (1 + r[2]) * cf[2]),
        "source": "measured", "agb_mg_ha": round(agb, 1), "agb_pixels": int(stats["agb_count"]), "climate_zone": zone,
        "dataset": "Simard et al. 2019 (NASA ORNL DAAC 1665), nominal year 2000",
    }
    return out


def _tri(rng, lmh, n):
    lo, mo, hi = lmh
    if hi <= lo:
        return np.full(n, mo, float)
    return rng.triangular(lo, min(max(mo, lo), hi), hi, n)


def _q(v: np.ndarray) -> dict:
    return {"mean": float(np.mean(v)), "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95))}


def carbon_report(areas_ha: dict[str, float], cfg: CarbonCfg, area_sd_ha: dict[str, float] | None = None,
                  seed: int = 0, lat: float | None = None, lon: float | None = None,
                  biomass_stats: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    soils = soil_sources(cfg, lat, lon)
    bio = measured_biomass(biomass_stats, lat, cfg)
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
        stock_c = area * (_tri(rng, soils[k]["soil"], n) + _tri(rng, bio[k]["biomass"], n))
        seq_c = area * _tri(rng, c.accumulation, n)
        total_stock += stock_c
        total_seq += seq_c
        per[k] = {
            "area_ha": a,
            "stock_tC": _q(stock_c),
            "stock_tCO2e": _q(stock_c * cfg.co2_per_c),
            "sequestration_tCO2e_per_yr": _q(seq_c * cfg.co2_per_c),
            "soil": soils[k],
            "biomass": {**bio[k], "biomass": tuple(round(x, 1) for x in bio[k]["biomass"])},
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
    bio = [k for k, v in per.items() if (v.get("biomass") or {}).get("source") == "measured"]
    soil = (f"Soil carbon from measured soil cores near the site (Smithsonian Coastal Carbon Library) for "
            f"{', '.join(measured)}, IPCC Tier 1 otherwise" if measured else "Soil carbon IPCC 2013 Wetlands Supplement Tier 1")
    biomass = ("mangrove biomass from the NASA canopy-height biomass map (Simard et al. 2019) with IPCC carbon and "
               "root factors, other biomass IPCC Tier 1" if bio else "biomass IPCC Tier 1")
    return f"{soil}; {biomass}; burial rates IPCC Tier 1; triangular Monte Carlo, n={n}"


def change_report(areas_t0: dict[str, float], areas_t1: dict[str, float], cfg: CarbonCfg, seed: int = 0,
                  lat: float | None = None, lon: float | None = None, biomass_stats: dict | None = None) -> dict:
    """Area change per class and the carbon implication of blue carbon gain/loss."""
    rng = np.random.default_rng(seed)
    soils = soil_sources(cfg, lat, lon)
    bio = measured_biomass(biomass_stats, lat, cfg)
    n = cfg.monte_carlo
    delta = {k: areas_t1.get(k, 0.0) - areas_t0.get(k, 0.0) for k in CLASS_KEYS}
    net = np.zeros(n)
    per = {}
    for k in BLUE_CARBON_KEYS:
        c = cfg.classes.get(k)
        d = delta.get(k, 0.0)
        if c is None or d == 0:
            continue
        v = d * (_tri(rng, soils[k]["soil"], n) + _tri(rng, bio[k]["biomass"], n)) * cfg.co2_per_c
        net += v
        per[k] = {"delta_ha": d, "stock_change_tCO2e": _q(v)}
    return {"delta_ha": delta, "classes": per, "net_stock_change_tCO2e": _q(net)}
