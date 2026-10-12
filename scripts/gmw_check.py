"""Global Mangrove Watch check of mangrove change, for every site page with mangrove (run in Colab, a few minutes):

    python scripts/gmw_check.py

Writes data/site_gmw.json: Global Mangrove Watch v3 mangrove area per year (1996-2020) inside each site's box and,
where our Landsat habitat history exists, how well the two records agree (same years compared, and whether both
show the same direction of change). Earth Engine only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bluecarbon import gee  # noqa: E402
from bluecarbon.config import load_config  # noqa: E402


def agreement(gmw: dict, hist: dict) -> dict | None:
    """Compare GMW with our history (adjusted mangrove ha per epoch year) on the years both have (+/- 1 year)."""
    pairs = []
    for y, ha in gmw.items():
        near = [e for e in hist if abs(e["year"] - int(y)) <= 1]
        if near:
            pairs.append((ha, min(near, key=lambda e: abs(e["year"] - int(y)))["ha"]))
    if len(pairs) < 2:
        return None
    g0, o0 = pairs[0]
    g1, o1 = pairs[-1]
    rel = [abs(o - g) / g for g, o in pairs if g > 0]
    return {"years_compared": len(pairs), "median_diff_pct": round(100 * sorted(rel)[len(rel) // 2], 1) if rel else None,
            "gmw_change_pct": round(100 * (g1 - g0) / g0, 1) if g0 else None,
            "ours_change_pct": round(100 * (o1 - o0) / o0, 1) if o0 else None,
            "same_direction": (g1 - g0) * (o1 - o0) >= 0}


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    gee.init(cfg.project)
    out_p = ROOT / "data" / "site_gmw.json"
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    for mp in sorted((ROOT / "demo_data").glob("*/meta.json")):
        meta = json.loads(mp.read_text())
        rep = meta.get("report") or meta.get("t1") or {}
        if (rep.get("areas_ha") or {}).get("mangrove", 0) < 5:
            continue
        (s, w), (n, e) = meta["bounds"]
        sid = mp.parent.name
        gmw = res.get(sid, {}).get("gmw_ha") or gee.gmw_areas([w, s, e, n])
        rec = {"gmw_ha": gmw}
        hp = mp.parent / "history.json"
        if hp.exists():
            h = json.loads(hp.read_text())
            hist = [{"year": e["year"], "ha": e["adjusted_ha"]["mangrove"]["ha"]} for e in h["epochs"]
                    if "adjusted_ha" in e and "mangrove" in e["adjusted_ha"]]
            rec["agreement"] = agreement(gmw, hist)
        res[sid] = rec
        print(f"{sid:24s} GMW {gmw}  {rec.get('agreement') or ''}", flush=True)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(res, indent=1, sort_keys=True))
    print(f"wrote {out_p}")


if __name__ == "__main__":
    main()
