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
def _background(fai: np.ndarray, ocean: np.ndarray) -> np.ndarray:
    """Local ocean background: median of nearby ocean pixels over a ~BG_WINDOW-pixel window.

    Computed on a 3x coarser grid and expanded back, which is ~10x faster than a full-resolution 21x21 median
    filter (that made calibration take many hours) and gives the same smooth background."""
    from scipy.ndimage import median_filter, zoom

    fill = np.nanmedian(fai[ocean]) if ocean.any() else 0.0
    src = np.where(ocean, fai, fill).astype(np.float32)
    k = 3
    h, w = src.shape
    H, W = -(-h // k) * k, -(-w // k) * k
    pad = np.pad(src, ((0, H - h), (0, W - w)), mode="edge")
    coarse = np.median(pad.reshape(H // k, k, W // k, k).transpose(0, 2, 1, 3).reshape(H // k, W // k, -1), axis=2)
    coarse = median_filter(coarse, size=max(3, BG_WINDOW // k), mode="nearest")
    return zoom(coarse, k, order=1)[:h, :w]


def kelp_signal(fai_p90: np.ndarray, land: np.ndarray, px_m: float) -> tuple:
    """Background-removed FAI, masks and distance to shore. Independent of the fitted settings, so calibration
    computes it once per quarter."""
    from scipy.ndimage import distance_transform_edt

    fai = fai_p90.astype(np.float32)
    finite = np.isfinite(fai)
    ocean = ~land & finite
    sfai = fai - _background(fai, ocean)
    dist = (distance_transform_edt(~land) * px_m if land.any() else np.full(land.shape, 1e5)).astype(np.float32)
    keep = ocean & (dist <= NEARSHORE_KM * 1000)
    # cached for every quarter of every site during calibration: store compactly (half precision is ample here)
    return sfai.astype(np.float16), keep, finite, np.minimum(dist, 60000).astype(np.float16)


def fraction_from_signal(sig: tuple, threshold: float, endmember: float, shore_m: float = 0.0) -> np.ndarray:
    """shore_m: ignore the surf zone (breaking waves and foam look like floating kelp to the FAI)."""
    sfai, keep, finite, dist = sig
    frac = np.clip((sfai.astype(np.float32) - threshold) / max(endmember - threshold, 1e-6), 0, 1)
    frac[~(keep & (dist >= shore_m))] = 0
    frac[~finite] = np.nan
    return frac.astype(np.float32)


def kelp_fraction(fai_p90: np.ndarray, land: np.ndarray, px_m: float, threshold: float = DEFAULT_THRESHOLD,
                  endmember: float = DEFAULT_ENDMEMBER, shore_m: float = 0.0) -> np.ndarray:
    """fai_p90: quarterly 90th-percentile FAI (reflectance; NaN = no clear scene); land: bool."""
    return fraction_from_signal(kelp_signal(fai_p90, land, px_m), threshold, endmember, shore_m)


def _sample_fraction(s: dict, threshold: float, endmember: float, shore_m: float = 0.0) -> np.ndarray:
    if "_sig" not in s:  # cache the expensive part on the sample
        s["_sig"] = kelp_signal(s["fai"], s["land"], s["px_m"])
    return fraction_from_signal(s["_sig"], threshold, endmember, shore_m)


def canopy_ha(frac: np.ndarray, px_m: float) -> float:
    return float(np.nansum(frac) * px_m * px_m / 1e4)


def coverage(fai_p90: np.ndarray, land: np.ndarray) -> float:
    ocean = ~land
    return float(np.isfinite(fai_p90[ocean]).mean()) if ocean.any() else 0.0


# ---------------------------------------------------------------------------------------- Kelpwatch
KELPWATCH_PACKAGE = ("knb-lter-sbc", 74)
PASTA = "https://pasta.lternet.edu/package"


def _http(url: str, timeout: int = 120, tries: int = 4) -> bytes:
    """GET with a browser-like User-Agent: EDI answers 403 Forbidden to Python's default 'Python-urllib'."""
    import time
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/124.0 Safari/537.36 BlueCarbon-AI",
        "Accept": "*/*"})
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            if k == tries - 1:
                raise
            print(f"  EDI request failed ({e}); retrying", flush=True)
            time.sleep(5 * (k + 1))
    raise RuntimeError("unreachable")


MANUAL = ("Download it by hand: open https://portal.edirepository.org/nis/mapbrowse?scope=knb-lter-sbc&identifier=74 "
          "-> Data Entities -> the .nc file, upload it to Google Drive as My Drive/bluecarbon-ai-cache/sites/_kelp/"
          "kelpwatch.nc (in Colab that is {dst}), and run the cell again.")


def download_kelpwatch(dst: Path) -> Path:
    """The newest SBC LTER quarterly kelp canopy NetCDF (Kelpwatch's data) via the EDI PASTA API.
    Free and public; nothing is billed. If the API refuses, download it by hand (see MANUAL)."""
    dst = Path(dst)
    if dst.exists():
        return dst
    scope, ident = KELPWATCH_PACKAGE
    rev = "?"
    try:
        rev = _http(f"{PASTA}/eml/{scope}/{ident}?filter=newest", 60).decode().strip()
        urls = _http(f"{PASTA}/data/eml/{scope}/{ident}/{rev}", 60).decode().split()
        for u in urls:
            name = _http(f"{PASTA}/name/eml/{scope}/{ident}/{rev}/{u.rsplit('/', 1)[-1]}", 60).decode().strip()
            if name.lower().endswith(".nc"):
                dst.parent.mkdir(parents=True, exist_ok=True)
                print(f"downloading Kelpwatch data: {name} (revision {rev})", flush=True)
                tmp = dst.with_suffix(".part")
                tmp.write_bytes(_http(u, 1800))
                tmp.rename(dst)
                return dst
    except Exception as e:
        raise RuntimeError(f"could not download Kelpwatch data from EDI ({e}). " + MANUAL.format(dst=dst)) from e
    raise RuntimeError(f"no NetCDF in EDI package {scope}.{ident} revision {rev}. " + MANUAL.format(dst=dst))


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
    area = area.isel({st_dim: np.flatnonzero(keep)}).transpose(..., st_dim).values  # load only this site, not 2.3 GB
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


MIN_COVERAGE = 0.5   # quarters where under half the ocean had a clear scene are too patchy to score


def _covered(s: dict) -> bool:
    if "_cov" not in s:
        s["_cov"] = coverage(s["fai"], s["land"])
    return s["_cov"] >= MIN_COVERAGE


def pairs(samples: list[dict], threshold: float, endmember: float, shore_m: float = 0.0
          ) -> tuple[np.ndarray, np.ndarray, list]:
    """samples: [{"fai", "land", "px_m", "kw_cells", "site", "year", "quarter"}] -> cell pairs + site totals."""
    o, r, totals = [], [], []
    for s in samples:
        if not _covered(s):
            continue
        f = _sample_fraction(s, threshold, endmember, shore_m)
        oc = our_cells(f, s["px_m"])
        kc = s["kw_cells"][: oc.shape[0], : oc.shape[1]]
        oc = oc[: kc.shape[0], : kc.shape[1]]
        m = (oc > 0) | (kc > 0)
        o.append(oc[m])
        r.append(kc[m])
        totals.append((s["site"], s["year"], s["quarter"], float(oc.sum() / 1e4), float(kc.sum() / 1e4)))
    return (np.concatenate(o) if o else np.zeros(0)), (np.concatenate(r) if r else np.zeros(0)), totals


THRESHOLDS = (0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.016, 0.024)
ENDMEMBERS = (0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.18, 0.22, 0.28)
SHORE_M = (0, 40, 80, 120)


def _score(samples, t, e, b) -> float:
    o, r, tot = pairs(samples, t, e, b)
    if not o.size:
        return np.inf
    so, sr = np.array([x[3] for x in tot]), np.array([x[4] for x in tot])
    return float(np.sqrt(np.mean((o - r) ** 2)) / (np.std(r) or 1) + np.sqrt(np.mean((so - sr) ** 2)) / (np.std(sr) or 1))


def _search(samples, grid=None, log=print, label="") -> tuple[float, float, float]:
    """Best (threshold, end member, surf-zone buffer). The fit balances per-cell placement (300 m cells) and each
    site's total canopy per quarter, the number that is published; fitting cells alone left totals ~30 ha high."""
    best = None
    ts, es, bs = grid or (THRESHOLDS, ENDMEMBERS, SHORE_M)
    for t in ts:
        if grid is None:
            log(f"  calibrating{label}: threshold {t} ...")
        for e in es:
            if e <= t * 2:
                continue
            for b in bs:
                sc = _score(samples, t, e, b)
                if best is None or sc < best[0]:
                    best = (sc, t, e, b)
    return best[1:] if best and np.isfinite(best[0]) else (DEFAULT_THRESHOLD, DEFAULT_ENDMEMBER, 0.0)


def _near(t, e, b):
    """A small grid around a fit, for the leave-one-bed-out refits (27 settings instead of ~270)."""
    return ((round(t / 1.5, 4), t, round(t * 1.5, 4)), (round(e * 0.8, 3), e, round(e * 1.25, 3)),
            tuple(sorted({max(0, b - 40), b, b + 40})))


def _metrics(samples, t, e, b) -> dict:
    o, r, totals = pairs(samples, t, e, b)
    return {"cells_m2": _stats(o, r), "site_quarter_ha": _stats([x[3] for x in totals], [x[4] for x in totals])}


def params_for(cal: dict, site: str) -> tuple[float, float, float]:
    """(threshold, end member, surf-zone buffer) used for a site: its zone's fit when there is one, else global."""
    z = (cal.get("site_zone") or {}).get(site)
    p = (cal.get("zones") or {}).get(z) or cal
    return p["threshold"], p["endmember"], p.get("shore_m", 0)


def calibrate(samples: list[dict], test_sites: set[str], zones: dict[str, str] | None = None, log=print,
              fit_filter=None) -> dict:
    """Fit the canopy settings on the calibration sites, then score every bed on data it was not tuned on.

    - Global fit over all calibration sites (as before), and a separate fit per zone (e.g. giant kelp vs bull
      kelp, whose floating canopy looks different) when the zone has at least two calibration sites.
    - Held-out sites (role: test) are never used to fit anything; they are scored with their zone's settings.
    - Every calibration site is also scored leave-one-bed-out: its zone (or global) settings are refitted without
      it, so each bed gets its own honest error, not only the two held-out beds.
    fit_filter(sample) -> bool: optional subset used for the searches (e.g. every other year of a 32-year Landsat
    record, to keep the fit fast); every score uses all samples.
    """
    zones = zones or {}
    keep = fit_filter or (lambda s: True)
    cal = [s for s in samples if s["site"] not in test_sites]
    cal_fit = [s for s in cal if keep(s)]
    t, e, b = _search(cal_fit, log=log)
    res = {"threshold": t, "endmember": e, "shore_m": b, "cell_m": CELL_M, "min_coverage": MIN_COVERAGE,
           "zones": {}, "site_zone": zones}
    for z in sorted(set(zones.values())):
        zs = [s for s in cal_fit if zones.get(s["site"]) == z]
        if len({s["site"] for s in zs}) >= 2:
            zt, ze, zb = _search(zs, log=log, label=f" {z}")
            res["zones"][z] = {"threshold": zt, "endmember": ze, "shore_m": zb,
                               "sites": sorted({s["site"] for s in zs})}
    held = [s for s in samples if s["site"] in test_sites]
    per_site = {}
    for site in sorted({s["site"] for s in samples}):
        own = [s for s in samples if s["site"] == site]
        if site in test_sites:
            per_site[site] = {**_metrics(own, *params_for(res, site)), "how": "held out"}
            continue
        z = zones.get(site)
        pool = [s for s in cal_fit if s["site"] != site and (z not in res["zones"] or zones.get(s["site"]) == z)]
        if len({s["site"] for s in pool}) < 1:
            continue
        lt, le, lb = _search(pool, grid=_near(*params_for(res, site)))
        per_site[site] = {**_metrics(own, lt, le, lb), "how": "left out of its own fit"}
    res["per_site"] = per_site
    res["calibration"] = _metrics(cal, t, e, b)
    # held-out beds, each with its zone's settings, pooled
    o, r, so, sr = [], [], [], []
    for site in sorted({s["site"] for s in held}):
        oo, rr, tot = pairs([s for s in held if s["site"] == site], *params_for(res, site))
        o.append(oo); r.append(rr); so += [x[3] for x in tot]; sr += [x[4] for x in tot]  # noqa: E702
    res["held_out"] = ({"cells_m2": _stats(np.concatenate(o), np.concatenate(r)), "site_quarter_ha": _stats(so, sr)}
                       if o else {})
    return res


def site_series(samples: list[dict], cal: dict) -> list[dict]:
    """Quarterly canopy (ha) for one site's samples, with the Kelpwatch value when it exists. The +/- range is the
    site's own out-of-fit error when known, else the pooled held-out error."""
    out = []
    ho = (cal.get("held_out") or {}).get("site_quarter_ha") or {}
    for s in sorted(samples, key=lambda s: (s["year"], s["quarter"])):
        own = (((cal.get("per_site") or {}).get(s["site"]) or {}).get("site_quarter_ha") or {})
        pm = own.get("rmse", ho.get("rmse"))
        f = _sample_fraction(s, *params_for(cal, s["site"]))
        ha = canopy_ha(f, s["px_m"])
        row = {"year": s["year"], "quarter": s["quarter"], "canopy_ha": round(ha, 2),
               "coverage": round(coverage(s["fai"], s["land"]), 2)}
        if pm is not None:
            row["plus_minus_ha"] = pm
        if s.get("kw_cells") is not None:
            row["kelpwatch_ha"] = round(float(s["kw_cells"].sum() / 1e4), 2)
        out.append(row)
    return out


def save_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1))
