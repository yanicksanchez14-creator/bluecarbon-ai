"""Preflight for a long training run: catch in 2 minutes what would otherwise fail after 20 hours.

    python scripts/preflight.py

Checks the GPU, Google Drive and free disk, then that every Earth Engine dataset the run reads is reachable from
this project (each with one tiny request), and downloads one small real L-band + clear-water sample. Ends with
PREFLIGHT OK or a list of problems. Earth Engine only; nothing else is called.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import ee  # noqa: E402
import numpy as np  # noqa: E402

from bluecarbon import gee  # noqa: E402
from bluecarbon.config import load_config  # noqa: E402

# (name, asset id, kind, needed): needed=False means the run carries on without it (a check is skipped)
TEST_BOX = [-81.52, 25.86, -81.47, 25.91]   # Ten Thousand Islands, Florida: mangrove forest, creeks and bay


def assets(cfg) -> list[tuple[str, str, str, bool]]:
    lc, ic = cfg.labels, cfg.imagery
    return [
        ("Sentinel-2", ic.collection, "col", True),
        ("Cloud Score+", ic.cloud_score_collection, "col", True),
        ("Sentinel-1 radar", "COPERNICUS/S1_GRD", "col", True),
        ("ALOS PALSAR-2 L-band", gee.PALSAR_YEARLY, "col", True),
        ("ESA WorldCover", lc.worldcover, "col", True),
        ("NASADEM elevation", lc.dem, "img", True),
        ("Murray tidal wetlands", lc.tidal_wetland, "img", True),
        ("Murray tidal flats", lc.intertidal, "col", True),
        ("Allen Coral Atlas", lc.reef_habitat, "img", False),
        ("GWL_FCS30 wetlands", lc.wetland_map, "col", True),
        ("Global Mangrove Watch", lc.mangrove_map, "fc", True),
        ("GEBCO depth", lc.bathymetry, "col", False),
        ("NASA mangrove biomass", gee.MANGROVE_AGB, "col", False),
        ("GEDI L4A biomass", gee.GEDI_L4A, "col", False),
        ("ETH 10 m canopy height", gee.CANOPY_HEIGHT, "img", False),
        ("Meta 1 m canopy height", gee.META_CANOPY_HEIGHT, "col", False),
        ("Landsat 8 (history)", "LANDSAT/LC08/C02/T1_L2", "col", False),
    ]


def probe(asset: str, kind: str) -> str:
    region = gee.bbox_geometry(TEST_BOX)
    if kind == "col":
        n = ee.ImageCollection(asset).limit(5).size().getInfo()
        if not n:
            raise RuntimeError("collection is empty")
        return f"{n}+ images"
    if kind == "fc":
        return f"{ee.FeatureCollection(asset).filterBounds(region).limit(50).size().getInfo()} features here"
    return f"{len(ee.Image(asset).bandNames().getInfo())} bands"


def main() -> None:
    problems, warnings = [], []
    try:
        import torch

        print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
        if not torch.cuda.is_available():
            problems.append("no GPU: Runtime -> Change runtime type -> GPU, then run everything again")
    except Exception as e:
        warnings.append(f"torch check failed ({e})")
    sites = ROOT / "runs" / "default" / "sites"
    print("Drive cache linked:", sites.is_symlink())
    if not sites.is_symlink():
        problems.append("Drive cache not linked: run the Google Drive cell (step 3) first")
    free = shutil.disk_usage("/content" if Path("/content").exists() else ".").free / 1e9
    print(f"Free disk: {free:.0f} GB")
    if free < 30:
        warnings.append(f"only {free:.0f} GB free disk")

    cfg = load_config(ROOT / "configs" / "default.yaml")
    gee.init(cfg.project)
    for name, asset, kind, needed in assets(cfg):
        if not asset:
            continue
        try:
            print(f"  ok    {name:26s} {probe(asset, kind)}", flush=True)
        except Exception as e:
            msg = f"{name} ({asset}): {str(e)[:100]}"
            print(f"  {'FAIL' if needed else 'warn'}  {msg}", flush=True)
            (problems if needed else warnings).append(msg)

    with tempfile.TemporaryDirectory() as tmp:  # a real download through the same code the run uses
        try:
            import rasterio

            p = gee.download_lband(TEST_BOX, 2021, Path(tmp) / "lband.tif", cfg)
            with rasterio.open(p) as ds:
                a = ds.read()
            ok = (a[0] != -3000).mean()          # -3000 = no radar coverage (open water reads -30 dB or lower)
            hv = a[1][a[1] != -3000] / 100
            print(f"  ok    L-band sample: {a.shape[1]}x{a.shape[2]} px, {ok:.0%} with data, "
                  f"HV median {np.median(hv) if hv.size else float('nan'):.1f} dB (mangrove is about -15)", flush=True)
            if ok < 0.5:
                problems.append("L-band sample has almost no data over a mangrove forest")
            p = gee.download_ancillary(TEST_BOX, "2021-01-01", "2022-01-01", Path(tmp) / "anc.tif", cfg)
            print("  ok    clear-water + elevation sample", flush=True)
        except Exception as e:
            problems.append(f"sample download failed: {str(e)[:150]}")

    flag = ROOT / "runs" / "default" / "PREFLIGHT_FAILED.txt"   # retrain.py refuses to start while this exists
    flag.parent.mkdir(parents=True, exist_ok=True)
    if problems:
        flag.write_text("\n".join(problems) + "\n")
    else:
        flag.unlink(missing_ok=True)
    print()
    for w in warnings:
        print("warning:", w)
    if problems:
        print("PREFLIGHT FAILED:")
        for p in problems:
            print("  -", p)
        print("-> paste this output to Claude before training")
    else:
        print("PREFLIGHT OK: safe to start training")


if __name__ == "__main__":
    main()
