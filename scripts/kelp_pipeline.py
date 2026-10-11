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


def fraction_png(frac: np.ndarray, tr, crs, dst: Path) -> list:
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
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 214, 160, 38
    rgba[..., 3] = np.where(on, 90 + 1.6 * np.clip(f, 0, 100), 0).astype(np.uint8)
    save_png(rgba, dst)
    return bounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kelpwatch", type=Path, default=WORK / "kelpwatch.nc")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    cfg = load_config(ROOT / "configs" / "default.yaml")
    spec = yaml.safe_load((ROOT / "configs" / "kelp_sites.yaml").read_text())
    ly, lq = last_full_quarter(dt.date.today())
    qs = [x for x in quarters(spec["start_year"], ly) if x <= (ly, lq)]

    if not a.no_fetch:
        gee.init(cfg.project)
        for s in spec["sites"]:
            d = WORK / s["name"]
            for y, q in qs:
                p = d / f"q{y}_{q}.tif"
                if not p.exists():
                    print(f"[{s['name']}] {y} Q{q} ...", flush=True)
                    try:
                        gee.download_kelp_quarter(s["bbox"], y, q, p, cfg)
                    except Exception as e:  # one bad quarter must not stop the run
                        print(f"  skipped ({str(e)[:100]})", flush=True)

    kw_path = a.kelpwatch if a.kelpwatch.exists() else download_kelpwatch(a.kelpwatch)
    samples, per_site = [], {}
    for s in spec["sites"]:
        kw = read_kelpwatch(kw_path, s["bbox"])
        tidx = {(int(y), int(q)): i for i, (y, q) in enumerate(zip(kw["year"], kw["quarter"], strict=True))}
        per_site[s["name"]] = []
        for y, q in qs:
            p = WORK / s["name"] / f"q{y}_{q}.tif"
            if not p.exists():
                continue
            fai, n, land, tr, crs = read_quarter(p)
            smp = {"site": s["name"], "year": y, "quarter": q, "fai": fai, "land": land, "px_m": SCALE_M,
                   "tr": tr, "crs": crs, "n_scenes": int(np.median(n[~land])) if (~land).any() else 0,
                   "kw_cells": (kelpwatch_cells(kw, tidx[(y, q)], tr, crs, fai.shape) if (y, q) in tidx else None)}
            per_site[s["name"]].append(smp)
            if smp["kw_cells"] is not None:
                samples.append(smp)
        print(f"{s['name']}: {len(per_site[s['name']])} quarters, {len(kw['lat'])} Kelpwatch pixels", flush=True)

    test = {s["name"] for s in spec["sites"] if s.get("role") == "test"}
    cal = calibrate(samples, test)
    print(json.dumps(cal, indent=1), flush=True)
    shutil.rmtree(OUT, ignore_errors=True)
    save_json({**cal, "sites_calibration": sorted({s["name"] for s in spec["sites"]} - test),
               "sites_held_out": sorted(test), "kelpwatch_file": kw_path.name,
               "method": "Sentinel-2 Floating Algae Index (B4, B8A, B11), quarterly 90th percentile, background "
                         "removed, scaled to canopy fraction; threshold and end member fitted to Kelpwatch"},
              OUT / "calibration.json")
    for s in spec["sites"]:
        smp = per_site[s["name"]]
        if not smp:
            continue
        series = site_series(smp, cal)
        latest = next((x for x in reversed(smp) if np.isfinite(x["fai"]).mean() > 0.6), smp[-1])
        frac = kelp_fraction(latest["fai"], latest["land"], SCALE_M, cal["threshold"], cal["endmember"])
        bounds = fraction_png(frac, latest["tr"], latest["crs"], OUT / s["name"] / "canopy_latest.png")
        save_json({"name": s["name"], "title": s["title"], "region": s["region"], "species": s["species"],
                   "held_out": s["name"] in test, "bbox": s["bbox"], "bounds": bounds, "series": series,
                   "latest": {"year": latest["year"], "quarter": latest["quarter"]}},
                  OUT / s["name"] / "kelp.json")
    shutil.make_archive(str(ROOT / "results_kelp"), "zip", str(ROOT), "demo_kelp")
    print("\nDone: results_kelp.zip. Download it with: from google.colab import files; "
          "files.download('results_kelp.zip')")


if __name__ == "__main__":
    main()
