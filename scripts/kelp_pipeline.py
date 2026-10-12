"""Kelp canopy product: fetch, calibrate against Kelpwatch, write the website's kelp pages. Run in Colab.

    python scripts/kelp_pipeline.py                 # resumable; writes results_kelp.zip
    python scripts/kelp_pipeline.py --kelpwatch /content/drive/MyDrive/kelpwatch.nc   # if the auto-download fails

Earth Engine only (Sentinel-2 + WorldCover, 20 m, one composite per site per quarter) plus the public
Kelpwatch NetCDF from the EDI data portal (free, not a Google service). Quarterly composites are cached under
runs/default/sites/_kelp/, so with the Drive cache linked an interrupted run picks up where it stopped.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bluecarbon import gee  # noqa: E402
from bluecarbon.config import load_config  # noqa: E402
from bluecarbon.kelp import (  # noqa: E402
    SCALE_M,
    calibrate,
    download_kelpwatch,
    kelp_fraction,
    kelpwatch_cells,
    params_for,
    quarters,
    read_kelpwatch,
    save_json,
    site_series,
)

WORK = ROOT / "runs" / "default" / "sites" / "_kelp"
OUT = ROOT / "demo_kelp"


def last_full_quarter(today: dt.date) -> tuple[int, int]:
    q = (today.month - 1) // 3  # quarters completed this year
    return (today.year, q) if q else (today.year - 1, 4)


def read_quarter(p: Path):
    import rasterio

    with rasterio.open(p) as ds:
        fai, n, land = ds.read()
        tr, crs = ds.transform, ds.crs
    fai = np.where(fai == -32768, np.nan, fai / 1e4).astype(np.float32)
    return fai, n, land.astype(bool), tr, crs


def fraction_png(frac: np.ndarray, tr, crs, dst: Path, color=(214, 160, 38)) -> list:
    """Kelp fraction as a warm overlay in Web Mercator for the website; returns [[s, w], [n, e]]."""
    import tempfile

    import rasterio
    from rasterio.warp import Resampling

    from bluecarbon.demo import _to_mercator
    from bluecarbon.viz import save_png

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "f.tif"
        with rasterio.open(p, "w", driver="GTiff", width=frac.shape[1], height=frac.shape[0], count=1,
                           dtype="uint8", nodata=255, transform=tr, crs=crs) as ds:
            ds.write(np.where(np.isfinite(frac), np.round(frac * 100), 255).astype(np.uint8), 1)
        m, bounds = _to_mercator(p, [1], Resampling.nearest, 255)
    f = m[0].astype(float)
    rgba = np.zeros((*f.shape, 4), np.uint8)
    on = (f > 0) & (f <= 100)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = color
    rgba[..., 3] = np.where(on, 90 + 1.6 * np.clip(f, 0, 100), 0).astype(np.uint8)
    save_png(rgba, dst)
    return bounds


def load_samples(spec: dict, qs: list, folder, px_m: float, kw_path: Path, log=print):
    """Quarterly composites of every site as calibration samples, each with Kelpwatch's canopy on 300 m cells when
    Kelpwatch has that quarter. folder(site_name) -> directory of q{year}_{quarter}.tif files."""
    samples, per_site = [], {}
    for s in spec["sites"]:
        kw_cache = WORK / s["name"] / "kelpwatch_site.npz"  # on Drive: a disconnect never repeats this slow read
        if kw_cache.exists():
            kw = dict(np.load(kw_cache))
        else:
            kw = read_kelpwatch(kw_path, s["bbox"])
            kw_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(kw_cache, **{k: np.asarray(v) for k, v in kw.items()})
        tidx = {(int(y), int(q)): i for i, (y, q) in enumerate(zip(kw["year"], kw["quarter"], strict=True))}
        has_kw = len(kw["lat"]) > 0
        per_site[s["name"]] = []
        for y, q in qs:
            p = folder(s["name"]) / f"q{y}_{q}.tif"
            if not p.exists():
                continue
            fai, n, land, tr, crs = read_quarter(p)
            smp = {"site": s["name"], "year": y, "quarter": q, "fai": fai, "land": land, "px_m": px_m,
                   "tr": tr, "crs": crs, "n_scenes": int(np.median(n[~land])) if (~land).any() else 0,
                   "kw_cells": (kelpwatch_cells(kw, tidx[(y, q)], tr, crs, fai.shape)
                                if has_kw and (y, q) in tidx else None)}
            per_site[s["name"]].append(smp)
            if smp["kw_cells"] is not None:
                samples.append(smp)
        log(f"{s['name']}: {len(per_site[s['name']])} quarters, {len(kw['lat'])} Kelpwatch pixels")
    return samples, per_site


def cached_calibration(name: str, samples, test, zones, fit_filter=None):
    cal_cache = WORK / f"calibration_cache{'' if name == 'sentinel2' else '_' + name}.json"   # on Drive
    key = {"method": 3, "zones": zones, "sites": sorted(zones), "test": sorted(test), "n_samples": len(samples)}
    cached = json.loads(cal_cache.read_text()) if cal_cache.exists() else {}
    if cached.get("key") == key:
        print(f"using saved {name} calibration from Drive", flush=True)
        return cached["cal"]
    cal = calibrate(samples, test, zones, fit_filter=fit_filter)
    cal_cache.write_text(json.dumps({"key": key, "cal": cal}))
    return cal


def fetch(spec, qs, cfg, folder, download, label):
    for s in spec["sites"]:
        for y, q in qs:
            p = folder(s["name"]) / f"q{y}_{q}.tif"
            if not p.exists():
                print(f"[{s['name']}] {label} {y} Q{q} ...", flush=True)
                try:
                    download(s["bbox"], y, q, p, cfg)
                except Exception as e:  # one bad quarter must not stop the run
                    print(f"  skipped ({str(e)[:100]})", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kelpwatch", type=Path, default=WORK / "kelpwatch.nc")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-landsat", action="store_true", help="skip our own 1984-2015 Landsat record")
    a = ap.parse_args()
    cfg = load_config(ROOT / "configs" / "default.yaml")
    spec = yaml.safe_load((ROOT / "configs" / "kelp_sites.yaml").read_text())
    ly, lq = last_full_quarter(dt.date.today())
    qs = [x for x in quarters(spec["start_year"], ly) if x <= (ly, lq)]
    qs_ls = quarters(spec.get("landsat_start_year", 1984), spec["start_year"] - 1)
    s2_dir = lambda n: WORK / n  # noqa: E731
    ls_dir = lambda n: WORK / n / "landsat"  # noqa: E731
    if not a.no_fetch:
        gee.init(cfg.project)
        fetch(spec, qs, cfg, s2_dir, gee.download_kelp_quarter, "Sentinel-2")
    kw_path = a.kelpwatch if a.kelpwatch.exists() else download_kelpwatch(a.kelpwatch)
    test = {s["name"] for s in spec["sites"] if s.get("role") == "test"}
    zones = {s["name"]: s.get("zone", s.get("species", "kelp")) for s in spec["sites"]}

    samples, per_site = load_samples(spec, qs, s2_dir, SCALE_M, kw_path)
    cal = cached_calibration("sentinel2", samples, test, zones)
    print(json.dumps({k: v for k, v in cal.items() if k != "per_site"}, indent=1), flush=True)
    shutil.rmtree(OUT, ignore_errors=True)
    save_json({**cal, "sites_calibration": sorted({s["name"] for s in spec["sites"]} - test),
               "sites_held_out": sorted(test), "kelpwatch_file": kw_path.name,
               "method": "Sentinel-2 Floating Algae Index (B4, B8A, B11), quarterly 90th percentile, background "
                         "removed, surf zone masked, scaled to canopy fraction; threshold, end member and surf-zone "
                         "buffer fitted to Kelpwatch per zone (giant kelp, bull kelp); every bed scored on data it "
                         "was not tuned on; quarters under 50% clear ocean not scored"},
              OUT / "calibration.json")
    for s in spec["sites"]:
        smp = per_site[s["name"]]
        if not smp:
            continue
        series = site_series(smp, cal)
        latest = next((x for x in reversed(smp) if np.isfinite(x["fai"]).mean() > 0.6), smp[-1])
        frac = kelp_fraction(latest["fai"], latest["land"], SCALE_M, *params_for(cal, s["name"]))
        bounds = fraction_png(frac, latest["tr"], latest["crs"], OUT / s["name"] / "canopy_latest.png")
        save_json({"name": s["name"], "title": s["title"], "region": s["region"], "species": s["species"],
                   "zone": zones[s["name"]], "held_out": s["name"] in test, "bbox": s["bbox"], "bounds": bounds,
                   "series": series, "score": (cal.get("per_site") or {}).get(s["name"]),
                   "latest": {"year": latest["year"], "quarter": latest["quarter"]}},
                  OUT / s["name"] / "kelp.json")
    del samples, per_site

    if not a.no_fetch:   # sea temperature next to the canopy (NOAA OISST, Earth Engine)
        for s in spec["sites"]:
            c = WORK / s["name"] / "sst.json"
            try:
                if not c.exists():
                    rows = []
                    for y0 in range(spec.get("landsat_start_year", 1984), ly + 1, 10):   # decade per request
                        rows += gee.sea_surface_temperature(s["bbox"], y0, min(y0 + 9, ly))
                    c.write_text(json.dumps(rows))
                rows = json.loads(c.read_text())
                clim = {q: [r["c"] for r in rows if r["quarter"] == q and r["year"] < 2014] for q in (1, 2, 3, 4)}
                clim = {q: sum(v) / len(v) for q, v in clim.items() if v}
                for r in rows:
                    r["anom"] = round(r["c"] - clim.get(r["quarter"], r["c"]), 2)
                m = json.loads((OUT / s["name"] / "kelp.json").read_text())
                m["sst"] = rows
                save_json(m, OUT / s["name"] / "kelp.json")
            except Exception as e:
                print(f"[{s['name']}] sea temperature skipped ({str(e)[:100]})", flush=True)

    cal_ls = None
    if not a.no_landsat:   # our own canopy record before Sentinel-2: Landsat 5/7/8, 1984-2015, same method
        if not a.no_fetch:
            fetch(spec, qs_ls, cfg, ls_dir, gee.download_kelp_quarter_landsat, "Landsat")
        smp_ls, per_ls = load_samples(spec, qs_ls, ls_dir, 30, kw_path)
        fit = lambda s: s["year"] % 2 == 0 and s["quarter"] in (2, 3)  # noqa: E731
        for rows in per_ls.values():   # 1,400 quarters: keep memory low (half precision, cache only the fit subset)
            for s in rows:
                s["fai"] = s["fai"].astype(np.float16)
                s["_nocache"] = not fit(s)
        if smp_ls:
            cal_ls = cached_calibration("landsat", smp_ls, test, zones, fit_filter=fit)
            c = json.loads((OUT / "calibration.json").read_text())
            c["landsat"] = cal_ls
            save_json(c, OUT / "calibration.json")
            for s in spec["sites"]:
                if per_ls.get(s["name"]) and (OUT / s["name"] / "kelp.json").exists():
                    m = json.loads((OUT / s["name"] / "kelp.json").read_text())
                    m["series_landsat"] = site_series(per_ls[s["name"]], cal_ls)
                    m["score_landsat"] = (cal_ls.get("per_site") or {}).get(s["name"])
                    save_json(m, OUT / s["name"] / "kelp.json")
        del smp_ls, per_ls

    if not a.no_fetch:  # yearly pictures, every quarter's canopy map and Kelpwatch since 1984 for the website
        from kelp_gallery import build_gallery

        build_gallery(cfg, spec, OUT, WORK, kw_path, cal, cal_landsat=cal_ls)
    shutil.make_archive(str(ROOT / "results_kelp"), "zip", str(ROOT), "demo_kelp")
    print("\nDone: results_kelp.zip. Download it with: from google.colab import files; "
          "files.download('results_kelp.zip')")


if __name__ == "__main__":
    main()
