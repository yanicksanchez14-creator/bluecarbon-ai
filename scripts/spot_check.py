"""Independent spot-check sample for every held-out estuary (product plan: every run).

    python scripts/spot_check.py          # after export_all_sites.py wrote runs/default/sites/<site>/pred.tif

200 points per held-out site: 100 at random over the whole box, and 100 split across the blue carbon classes the
model mapped (random points alone are mostly water). Each row has the model's class, the reference label and a
link to high-resolution satellite imagery, so a person can check the map by eye. Writes results/spotcheck.csv.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.warp import transform as warp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bluecarbon.schema import BLUE_CARBON_KEYS, CLASS_KEYS, IGNORE_INDEX, KEY_TO_ID  # noqa: E402

WORK = Path("runs/default")


def sample_site(d: Path, rng, n_random: int = 100, n_blue: int = 100) -> list[dict]:
    with rasterio.open(d / "pred.tif") as ds:
        pred, tr, crs = ds.read(1), ds.transform, ds.crs
    lab = None
    if (d / "label.tif").exists():
        with rasterio.open(d / "label.tif") as ds:
            lab = ds.read(1)
    h, w = pred.shape
    picks = list(zip(rng.integers(0, h, n_random), rng.integers(0, w, n_random), strict=True))
    blue = [KEY_TO_ID[k] for k in BLUE_CARBON_KEYS if (pred == KEY_TO_ID[k]).any()]
    for c in blue:
        rr, cc = np.nonzero(pred == c)
        k = rng.choice(len(rr), min(len(rr), n_blue // len(blue)), replace=False)
        picks += list(zip(rr[k], cc[k], strict=True))
    xs, ys = zip(*[tr * (c + 0.5, r + 0.5) for r, c in picks], strict=True)
    lons, lats = warp(crs, "EPSG:4326", list(xs), list(ys))
    rows = []
    for (r, c), lon, lat in zip(picks, lons, lats, strict=True):
        ref = int(lab[r, c]) if lab is not None and lab.shape == pred.shape else IGNORE_INDEX
        rows.append({"site": d.name, "lat": round(lat, 6), "lon": round(lon, 6),
                     "model": CLASS_KEYS[int(pred[r, c])] if pred[r, c] < len(CLASS_KEYS) else "none",
                     "reference": CLASS_KEYS[ref] if ref < len(CLASS_KEYS) else "unlabelled",
                     "imagery": f"https://www.google.com/maps/@{lat:.6f},{lon:.6f},120m/data=!3m1!1e3",
                     "by_eye": ""})
    return rows


def main() -> None:
    sites = yaml.safe_load((ROOT / "configs" / "sites.yaml").read_text())["sites"]
    rng = np.random.default_rng(0)
    rows = []
    for s in sites:
        d = WORK / "sites" / s["name"]
        if s.get("role") == "test" and (d / "pred.tif").exists():
            rows += sample_site(d, rng)
    out = Path("results")
    out.mkdir(exist_ok=True)
    with open(out / "spotcheck.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["site", "lat", "lon", "model", "reference", "imagery", "by_eye"])
        wr.writeheader()
        wr.writerows(rows)
    print(f"spot-check: {len(rows)} points over {len({r['site'] for r in rows})} held-out sites -> {out / 'spotcheck.csv'}")


if __name__ == "__main__":
    main()
