"""Which layers each held-out estuary has on Drive, and whether they line up (for the live-vs-new comparison).

    python scripts/check_layers.py
"""

from pathlib import Path

import rasterio
import yaml

ROOT = Path(__file__).resolve().parents[1]
sites = yaml.safe_load((ROOT / "configs" / "sites.yaml").read_text())["sites"]
for s in (x for x in sites if x.get("role") == "test"):
    d = ROOT / "runs" / "default" / "sites" / s["name"]
    row = []
    for f in ("image.tif", "ancillary.tif", "radar.tif", "lband.tif", "label.tif"):
        p = d / f
        if p.exists():
            with rasterio.open(p) as ds:
                row.append(f"{f.split('.')[0]} {ds.count}b {ds.height}x{ds.width}")
        else:
            row.append(f"{f.split('.')[0]} MISSING")
    print(f"{s['name']:20s} " + " | ".join(row))
