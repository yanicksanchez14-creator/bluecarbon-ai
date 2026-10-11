"""Check that every survey answer key loads, before a long training run. Run in Colab (takes a few minutes):

    python scripts/check_surveys.py

For each site and each survey that overlaps it: how many polygons came back, how many pass the keep/drop
filter, and the values found in the filter fields (so a wrong class name shows up here, not after 30 hours).
Nothing is downloaded from Google; these are the agencies' own public map services.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bluecarbon.surveys import _intersects, fetch_survey, filter_features, seagrass_features  # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    surveys = cfg["labels"]["seagrass_surveys"]
    sites = yaml.safe_load((ROOT / "configs" / "sites.yaml").read_text())["sites"]
    used, bad = Counter(), []
    for site in sites:
        for s in (s for s in surveys if _intersects(site["bbox"], s.get("extent", [-180, -90, 180, 90]))):
            try:
                feats = fetch_survey(s, site["bbox"])
                if s.get("keep") or s.get("drop"):
                    kept = filter_features(feats, s.get("keep"), s.get("drop"))
                    fields = list((s.get("keep") or {}) | (s.get("drop") or {}))
                    vals = Counter(str((f.get("properties") or {}).get(k)) for f in feats for k in fields)
                    extra = "  values: " + ", ".join(f"{v} ({n})" for v, n in vals.most_common(8))
                else:
                    kept, _ = seagrass_features(feats) if s.get("class", "seagrass") == "seagrass" else (feats, "")
                    extra = ""
                print(f"{site['name']:24s} {s['name']:44s} {len(feats):6d} polygons, {len(kept):6d} kept{extra}",
                      flush=True)
                if kept:
                    used[s["name"]] += 1
            except Exception as e:
                bad.append(s["name"])
                print(f"{site['name']:24s} {s['name']:44s} FAILED: {str(e)[:120]}", flush=True)
    print("\nSurveys giving polygons:", ", ".join(f"{k} ({v} sites)" for k, v in used.items()))
    never = sorted({s["name"] for s in surveys} - set(used))
    print("Surveys giving nothing at any site:", ", ".join(never) or "none")
    if bad:
        print("FAILED requests:", ", ".join(sorted(set(bad))), "-> paste this output to Claude before training")


if __name__ == "__main__":
    main()
