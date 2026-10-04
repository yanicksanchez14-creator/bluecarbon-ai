"""Recompute the carbon numbers of every demo page from its stored areas (no imagery or model needed).

    python scripts/refresh_carbon.py

Use after changing carbon coefficients or the measured soil carbon table (data/soil_carbon_cores.csv).
"""

from __future__ import annotations

import json
from pathlib import Path

from bluecarbon.carbon import carbon_report, change_report
from bluecarbon.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def _basis(rep: dict) -> tuple[dict, dict | None]:
    adj = rep.get("error_adjusted_areas_ha")
    if adj:
        return {k: v["adjusted_ha"] for k, v in adj.items()}, {k: v["ci95_ha"] / 1.96 for k, v in adj.items()}
    return rep["areas_ha"], None


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml").carbon
    for meta_p in sorted((ROOT / "demo_data").glob("*/meta.json")):
        meta = json.loads(meta_p.read_text())
        (s, w), (n, e) = meta["bounds"]
        lat, lon = (s + n) / 2, (w + e) / 2
        reps = [meta["report"]] if "report" in meta else [meta[t] for t in ("t0", "t1") if t in meta]
        for rep in reps:
            areas, sd = _basis(rep)
            rep["carbon"] = carbon_report(areas, cfg, sd, lat=lat, lon=lon)
        if isinstance(meta.get("change"), dict) and "areas_t0_ha" in meta["change"]:
            ch = meta["change"]
            ch["change"] = change_report(ch["areas_t0_ha"], ch["areas_t1_ha"], cfg, lat=lat, lon=lon)
        meta_p.write_text(json.dumps(meta, indent=2))
        srcs = {k: v["soil"]["source"] for r in reps for k, v in r["carbon"]["classes"].items()}
        print(f"{meta_p.parent.name:22s} {srcs}")


if __name__ == "__main__":
    main()
