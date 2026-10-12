"""Website gallery for the kelp page: every quarter's canopy map, a summer satellite picture for every year since
1984, and Kelpwatch's own canopy record since 1984 for comparison. Run in Colab after kelp_pipeline.py (it is also
called at the end of it):

    python scripts/kelp_gallery.py          # reads the cached quarters on Drive; writes demo_kelp/ + results_kelp.zip

Earth Engine only (Sentinel-2 and Landsat true colour). Quarterly canopy maps come from the composites already
cached on Drive and the published calibration (demo_kelp/calibration.json); nothing is recalibrated.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bluecarbon import gee  # noqa: E402
from bluecarbon.config import load_config  # noqa: E402
from bluecarbon.kelp import SCALE_M, kelp_fraction, read_kelpwatch  # noqa: E402

FIRST_RGB_YEAR = 1984
KW_COLOR = (111, 211, 191)   # Kelpwatch layer: teal, so it never reads as our own map


def _kw(work: Path, kw_path: Path, site: dict) -> dict | None:
    c = work / site["name"] / "kelpwatch_site.npz"
    if c.exists():
        return dict(np.load(c))
    if kw_path and Path(kw_path).exists():
        return read_kelpwatch(kw_path, site["bbox"])
    return None


def _paint_kelpwatch(kw: dict, area: np.ndarray, tr, crs, shape) -> np.ndarray:
    """Kelpwatch station canopy (m2 per 30 m Landsat pixel) as a canopy fraction on our 20 m grid."""
    from rasterio.warp import transform as warp

    frac = np.zeros(shape, np.float32)
    xs, ys = warp("EPSG:4326", crs, list(map(float, kw["lon"])), list(map(float, kw["lat"])))
    cols, rows = ~tr * (np.array(xs), np.array(ys))
    f = np.clip(area / 900.0, 0, 1)
    for dr in (-1, 0):          # a 30 m pixel covers about 1.5 x 1.5 of our 20 m pixels: paint 2 x 2
        for dc in (-1, 0):
            r, c = (rows + dr + 0.5).astype(int), (cols + dc + 0.5).astype(int)
            ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1]) & (f > 0)
            np.maximum.at(frac, (r[ok], c[ok]), f[ok])
    return frac


def _rgb_jpeg(tif: Path, dst: Path) -> list:
    from PIL import Image
    from rasterio.warp import Resampling

    from bluecarbon.demo import _to_mercator

    m, bounds = _to_mercator(tif, [1, 2, 3], Resampling.bilinear, 0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.moveaxis(m, 0, -1).astype(np.uint8)).save(dst, quality=78, optimize=True, progressive=True)
    return bounds


def build_gallery(cfg, spec: dict, out: Path, work: Path, kw_path: Path | None, cal: dict, log=print) -> None:
    import datetime as dt

    from kelp_pipeline import fraction_png, read_quarter

    last_year = dt.date.today().year
    for s in spec["sites"]:
        d, o = work / s["name"], out / s["name"]
        meta_p = o / "kelp.json"
        if not meta_p.exists():
            log(f"[{s['name']}] no kelp.json yet (run kelp_pipeline.py first); skipped")
            continue
        meta = json.loads(meta_p.read_text())
        quarters = sorted(d.glob("q*_*.tif"), key=lambda p: tuple(int(x) for x in p.stem[1:].split("_")))
        grid = None
        frames = []
        for p in quarters:                                    # 1. our canopy map, every quarter
            fai, _n, land, tr, crs = read_quarter(p)
            grid = grid or (tr, crs, fai.shape)
            frac = kelp_fraction(fai, land, SCALE_M, cal["threshold"], cal["endmember"], cal.get("shore_m", 0))
            key = p.stem[1:]
            meta["frame_bounds"] = fraction_png(frac, tr, crs, o / "frames" / f"{key}.png")
            frames.append(key)
        meta["frames"] = frames
        kw = _kw(work, kw_path, s)                            # 2. Kelpwatch since 1984 (its own Landsat record)
        if kw is not None and grid is not None:
            yrs, qq, area = np.asarray(kw["year"]).astype(int), np.asarray(kw["quarter"]).astype(int), kw["area_m2"]
            meta["kelpwatch_history"] = [{"year": int(y), "quarter": int(q), "ha": round(float(np.nansum(a) / 1e4), 2)}
                                         for y, q, a in zip(yrs, qq, area, strict=True)]
            kw_years = []
            for y in sorted(set(yrs.tolist())):
                peak = np.nanmax(np.nan_to_num(area[yrs == y]), axis=0)   # the year's largest canopy, per station
                frac = _paint_kelpwatch(kw, peak, *grid)
                meta["kw_bounds"] = fraction_png(frac, grid[0], grid[1], o / "kw" / f"{y}.png", KW_COLOR)
                kw_years.append(int(y))
            meta["kw_years"] = kw_years
        rgb_years = []                                        # 3. a summer satellite picture, every year
        for y in range(FIRST_RGB_YEAR, last_year + 1):
            tif = d / "rgb" / f"{y}.tif"
            try:
                if not tif.exists():
                    gee.download_kelp_rgb(s["bbox"], y, tif, cfg)
                meta["rgb_bounds"] = _rgb_jpeg(tif, o / "rgb" / f"{y}.jpg")
                rgb_years.append(y)
            except Exception as e:  # a year with no clear summer scene: skip it
                log(f"[{s['name']}] {y} picture skipped ({str(e)[:80]})")
        meta["rgb_years"] = rgb_years
        meta_p.write_text(json.dumps(meta, indent=1))
        log(f"[{s['name']}] {len(frames)} quarterly maps, {len(meta.get('kw_years', []))} Kelpwatch years, "
            f"{len(rgb_years)} yearly pictures")


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    spec = yaml.safe_load((ROOT / "configs" / "kelp_sites.yaml").read_text())
    work, out = ROOT / "runs" / "default" / "sites" / "_kelp", ROOT / "demo_kelp"
    cal = json.loads((out / "calibration.json").read_text())
    gee.init(cfg.project)
    build_gallery(cfg, spec, out, work, work / "kelpwatch.nc", cal)
    shutil.make_archive(str(ROOT / "results_kelp"), "zip", str(ROOT), "demo_kelp")
    print("\nDone: results_kelp.zip")


if __name__ == "__main__":
    main()
