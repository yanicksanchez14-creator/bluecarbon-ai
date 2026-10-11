"""Kelp canopy monitoring for California, calibrated and scored against Kelpwatch.

Method (published, peer-reviewed building blocks):

1. Every cloud-free Sentinel-2 scene of a quarter -> Floating Algae Index (FAI, Hu 2009) on bands 4, 8A and
   11 at 20 m. Floating kelp canopy reflects near-infrared strongly; water doesn't.
2. Per pixel and quarter, the 90th percentile FAI over the clear scenes (canopy lies flat at low tide and
   slack current, so the higher values are the canopy actually present).
3. Background removed: FAI minus the local ocean median (21 x 21 pixels) = sFAI (Wang and Hu).
4. Kelp fraction per pixel = sFAI scaled between a detection threshold and a pure-canopy end member,
   only within 2 km of land (where canopy forms) and never on land.
5. The threshold and end member are fitted to Kelpwatch's measured canopy (UC Santa Barbara SBC LTER,
   Landsat, calibrated with divers) on the calibration sites, then scored on sites left out, at 300 m cells
   and per site and quarter. Those scores are the published error.

Canopy area = sum(fraction x pixel area). Quarterly area comes with a +/- range from the held-out RMSE.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCALE_M = 20
NEARSHORE_KM = 2.0
BG_WINDOW = 21
CELL_M = 300                # comparison cells against Kelpwatch (robust to 20 m vs 30 m misregistration)
DEFAULT_THRESHOLD = 0.004   # sFAI (reflectance) above which a pixel holds some canopy
DEFAULT_ENDMEMBER = 0.08    # sFAI of a pixel fully covered by canopy (fitted against Kelpwatch)


def quarters(start_year: int, end_year: int) -> list[tuple[int, int]]:
    return [(y, q) for y in range(start_year, end_year + 1) for q in (1, 2, 3, 4)]


def quarter_dates(year: int, q: int) -> tuple[str, str]:
    m0 = 3 * (q - 1) + 1
    end = f"{year + 1}-01-01" if q == 4 else f"{year}-{m0 + 3:02d}-01"
    return f"{year}-{m0:02d}-01", end


# ---------------------------------------------------------------------------------------- per-pixel maths
def kelp_fraction(fai_p90: np.ndarray, land: np.ndarray, px_m: float, threshold: float = DEFAULT_THRESHOLD,
                  endmember: float = DEFAULT_ENDMEMBER) -> np.ndarray:
    """fai_p90: quarterly 90th-percentile FAI (reflectance; NaN = no clear scene); land: bool."""
    from scipy.ndimage import distance_transform_edt, median_filter

    fai = fai_p90.astype(np.float32)
    ocean = ~land & np.isfinite(fai)
    bg_src = np.where(ocean, fai, np.nan)
    # local ocean background: median of nearby ocean pixels (NaN-safe via fill with the global ocean median)
    fill = np.nanmedian(bg_src) if ocean.any() else 0.0
    bg = median_filter(np.where(ocean, fai, fill), size=BG_WINDOW, mode="nearest")
    sfai = fai - bg
    near = distance_transform_edt(~land) * px_m <= NEARSHORE_KM * 1000 if land.any() else np.ones_like(land)
    frac = np.clip((sfai - threshold) / max(endmember - threshold, 1e-6), 0, 1)
    frac[~(ocean & near)] = 0
    frac[~np.isfinite(fai)] = np.nan
    return frac.astype(np.float32)


def canopy_ha(frac: np.ndarray, px_m: float) -> float:
    return float(np.nansum(frac) * px_m * px_m / 1e4)


def coverage(fai_p90: np.ndarray, land: np.ndarray) -> float:
    ocean = ~land
    return float(np.isfinite(fai_p90[ocean]).mean()) if ocean.any() else 0.0


# ---------------------------------------------------------------------------------------- Kelpwatch
KELPWATCH_PACKAGE = ("knb-lter-sbc", 74)
PASTA = "https://pasta.lternet.edu/package"


def download_kelpwatch(dst: Path) -> Path:
    """The newest SBC LTER quarterly kelp canopy NetCDF (Kelpwatch's data) via the EDI PASTA API.
    Free and public; nothing is billed. If the API layout changes, download it by hand from
    portal.edirepository.org (package knb-lter-sbc.74) and pass the file path instead."""
    import urllib.request

    dst = Path(dst)
    if dst.exists():
        return dst
    scope, ident = KELPWATCH_PACKAGE
    rev = urllib.request.urlopen(f"{PASTA}/eml/{scope}/{ident}?filter=newest", timeout=60).read().decode().strip()
    urls = urllib.request.urlopen(f"{PASTA}/data/eml/{scope}/{ident}/{rev}", timeout=60).read().decode().split()
    for u in urls:
        name = urllib.request.urlopen(f"{PASTA}/name/eml/{scope}/{ident}/{rev}/{u.rsplit('/', 1)[-1]}",
                                      timeout=60).read().decode().strip()
        if name.lower().endswith(".nc"):
            dst.parent.mkdir(parents=True, exist_ok=True)
            print(f"downloading Kelpwatch data: {name} (revision {rev})", flush=True)
            urllib.request.urlretrieve(u, dst)
            return dst
    raise RuntimeError(f"no NetCDF in EDI package {scope}.{ident} revision {rev}; download it by hand")


def _pick(names, *cands):
    low = {n.lower(): n for n in names}
    for c in cands:
        for n_low, n in low.items():
            if n_low == c or n_low.startswith(c):
                return n
    return None


def read_kelpwatch(nc_path: Path, bbox: list[float]) -> dict:
    """Kelpwatch canopy area per station (30 m Landsat pixel) and quarter inside bbox.

    Returns {"lat": (S,), "lon": (S,), "year": (T,), "quarter": (T,), "area_m2": (T, S)}. Variable names are
    looked up loosely (area / latitude / longitude / year / quarter) because they have changed between
    package revisions; the error lists what the file holds if they can't be found."""
    import xarray as xr

    ds = xr.open_dataset(nc_path)
    names = list(ds.variables)
    lat_n, lon_n = _pick(names, "latitude", "lat"), _pick(names, "longitude", "lon")
    area_n = _pick(names, "area", "canopy_area")
    year_n, q_n = _pick(names, "year"), _pick(names, "quarter")
    if not all([lat_n, lon_n, area_n, year_n, q_n]):
        raise RuntimeError(f"unexpected Kelpwatch layout; variables: {names}")
    lat, lon = ds[lat_n].values, ds[lon_n].values
    w, s, e, n = bbox
    keep = (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)
    area = ds[area_n]
    st_dim = ds[lat_n].dims[0]
    area = area.transpose(..., st_dim).values[..., keep]
    return {"lat": lat[keep], "lon": lon[keep], "year": ds[year_n].values.astype(int),
            "quarter": ds[q_n].values.astype(int), "area_m2": np.nan_to_num(area.astype(np.float64))}


def kelpwatch_cells(kw: dict, t_index: int, transform, crs, shape) -> np.ndarray:
    """Kelpwatch canopy (m2) summed into CELL_M cells of our raster grid for one quarter."""
    from rasterio.warp import transform as warp

    xs, ys = warp("EPSG:4326", crs, list(kw["lon"]), list(kw["lat"]))
    inv = ~transform
    cols, rows = inv * (np.array(xs), np.array(ys))
    k = max(1, int(round(CELL_M / abs(transform.a))))
    H, W = (shape[0] + k - 1) // k, (shape[1] + k - 1) // k
    out = np.zeros((H, W))
    r, c = (rows // k).astype(int), (cols // k).astype(int)
    ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    np.add.at(out, (r[ok], c[ok]), kw["area_m2"][t_index][ok])
    return out


def our_cells(frac: np.ndarray, px_m: float) -> np.ndarray:
    k = max(1, int(round(CELL_M / px_m)))
    H, W = frac.shape
    pad = np.pad(np.nan_to_num(frac), ((0, (-H) % k), (0, (-W) % k)))
    return pad.reshape(pad.shape[0] // k, k, pad.shape[1] // k, k).sum((1, 3)) * px_m * px_m


def _stats(ours: np.ndarray, ref: np.ndarray) -> dict:
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    if ours.size < 3 or np.var(ref) == 0:
        return {"n": int(ours.size)}
    r2 = 1 - np.sum((ref - ours) ** 2) / np.sum((ref - ref.mean()) ** 2)
    return {"n": int(ours.size), "r2": round(float(r2), 3), "rmse": round(float(np.sqrt(np.mean((ours - ref) ** 2))), 1),
            "bias": round(float(np.mean(ours - ref)), 1), "corr": round(float(np.corrcoef(ours, ref)[0, 1]), 3)}


def pairs(samples: list[dict], threshold: float, endmember: float) -> tuple[np.ndarray, np.ndarray, list]:
    """samples: [{"fai", "land", "px_m", "kw_cells", "site", "year", "quarter"}] -> cell pairs + site totals."""
    o, r, totals = [], [], []
    for s in samples:
        f = kelp_fraction(s["fai"], s["land"], s["px_m"], threshold, endmember)
        oc = our_cells(f, s["px_m"])
        kc = s["kw_cells"][: oc.shape[0], : oc.shape[1]]
        oc = oc[: kc.shape[0], : kc.shape[1]]
        m = (oc > 0) | (kc > 0)
        o.append(oc[m])
        r.append(kc[m])
        totals.append((s["site"], s["year"], s["quarter"], float(oc.sum() / 1e4), float(kc.sum() / 1e4)))
    return (np.concatenate(o) if o else np.zeros(0)), (np.concatenate(r) if r else np.zeros(0)), totals


def calibrate(samples: list[dict], test_sites: set[str]) -> dict:
    """Grid-search threshold and end member on calibration sites (cell RMSE), then score held-out sites."""
    cal = [s for s in samples if s["site"] not in test_sites]
    best = None
    for t in (0.002, 0.003, 0.004, 0.006, 0.008, 0.012):
        for e in (0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.22):
            if e <= t * 2:
                continue
            o, r, _ = pairs(cal, t, e)
            rmse = float(np.sqrt(np.mean((o - r) ** 2))) if o.size else np.inf
            if best is None or rmse < best[0]:
                best = (rmse, t, e)
    _, t, e = best
    res = {"threshold": t, "endmember": e, "cell_m": CELL_M}
    for name, group in (("calibration", cal), ("held_out", [s for s in samples if s["site"] in test_sites])):
        o, r, totals = pairs(group, t, e)
        site_o = [x[3] for x in totals]
        site_r = [x[4] for x in totals]
        res[name] = {"cells_m2": _stats(o, r), "site_quarter_ha": _stats(site_o, site_r)}
    return res


def site_series(samples: list[dict], cal: dict) -> list[dict]:
    """Quarterly canopy (ha) for one site's samples, with the Kelpwatch value when it exists."""
    out = []
    ho = (cal.get("held_out") or {}).get("site_quarter_ha") or {}
    for s in sorted(samples, key=lambda s: (s["year"], s["quarter"])):
        f = kelp_fraction(s["fai"], s["land"], s["px_m"], cal["threshold"], cal["endmember"])
        ha = canopy_ha(f, s["px_m"])
        row = {"year": s["year"], "quarter": s["quarter"], "canopy_ha": round(ha, 2),
               "coverage": round(coverage(s["fai"], s["land"]), 2)}
        if ho.get("rmse") is not None:
            row["plus_minus_ha"] = ho["rmse"]
        if s.get("kw_cells") is not None:
            row["kelpwatch_ha"] = round(float(s["kw_cells"].sum() / 1e4), 2)
        out.append(row)
    return out


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
