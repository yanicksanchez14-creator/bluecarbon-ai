"""Full retraining run: download anything missing, (re)build labels, train, export every demo page.

Resumable: imagery, ancillary layers and labels that already exist in runs/default are reused, so it
can be re-run in the same Colab session after an interruption. Writes results.zip.

    !git pull -q && python scripts/retrain.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("runs/default")


def run(*args: str) -> None:
    print("\n$", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def best_model() -> Path:
    """The model named in best.txt (never a stale best.* left over from an earlier run)."""
    name = (WORK / "model" / "best.txt").read_text().split()[1]
    return WORK / "model" / name


BLUE = ("mangrove", "saltmarsh", "seagrass")


def held_out_check() -> str:
    """Per-site scores on the estuaries never used in training, plus mapped vs reference area.

    The pooled test score is dominated by test blocks inside training sites, so it can look good
    while a held-out coastline fails. This table is the check to read before publishing.
    """
    import yaml

    sites = [s["name"] for s in yaml.safe_load(Path("configs/sites.yaml").read_text())["sites"] if s.get("role") == "test"]
    lines = ["HELD-OUT ESTUARIES (never used in training)", ""]
    for f, label in (("spectral_metrics.json", "spectral"), ("metrics.json", "unet")):
        p = WORK / "model" / f
        if not p.exists():
            continue
        per = json.loads(p.read_text()).get("test_per_site", {})
        lines.append(f"{label:9s} IoU      " + "  ".join(f"{c:>9s}" for c in BLUE))
        for s in sites:
            iou = per.get(s, {}).get("iou", {})
            lines.append(f"  {s:22s}" + "  ".join(f"{'-' if iou.get(c) is None else f'{iou[c]:.2f}':>9s}" for c in BLUE))
        lines.append("")
    lines.append("mapped / reference area (ha)")
    warn = []
    for s in sites:
        meta_p, site_p = Path("demo_data") / s / "meta.json", WORK / "sites" / s / "meta.json"
        if not (meta_p.exists() and site_p.exists()):
            continue
        mapped = json.loads(meta_p.read_text())["report"]["areas_ha"]
        ref = {k: v / 100 for k, v in (json.loads(site_p.read_text()).get("label_px") or {}).items()}
        row = []
        for c in BLUE:
            m, r = mapped.get(c, 0), ref.get(c, 0)
            row.append(f"{m:7.0f}/{r:<7.0f}")
            if r > 100 and not 0.33 < m / r < 3:
                warn.append(f"  WARNING {s}: {c} mapped {m:.0f} ha vs reference {r:.0f} ha")
        lines.append(f"  {s:22s}" + "  ".join(row))
    lines += [""] + (warn or ["  all held-out blue carbon areas within 3x of reference"])
    return "\n".join(lines)


def code_version() -> str:
    try:
        return subprocess.run(["git", "log", "-1", "--format=%h %s"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    print("Code version:", code_version(), flush=True)
    skip_train = "--skip-train" in sys.argv
    if not skip_train:
        run("bluecarbon", "fetch", "--labels-only")
        shutil.rmtree(WORK / "chips", ignore_errors=True)
        run("bluecarbon", "chips")
        for old in (WORK / "model").glob("best.*"):
            old.unlink()
        run("bluecarbon", "train", "--kind", "both")
    best = best_model()
    print((WORK / "model" / "best.txt").read_text(), "->", best)

    run("bluecarbon", "case-study", "-m", str(best))
    shutil.rmtree("demo_data", ignore_errors=True)
    import yaml

    for cs in (yaml.safe_load(Path("configs/sites.yaml").read_text()).get("case_studies") or []):
        scene = WORK / "scenes" / cs["name"]
        if not (scene / "t1_pred.tif").exists():
            print(f"skip {cs['name']} (change study did not finish)")
            continue
        run("bluecarbon", "export-demo", str(scene), "--name", cs["name"], "--title", cs["title"], "-m", str(best),
            "--description", cs.get("description", ""))
        meta_p = Path("demo_data") / cs["name"] / "meta.json"
        meta = json.loads(meta_p.read_text())
        lab = cs.get("labels") or {}
        meta.update(region=cs.get("region", ""), period=f"{lab.get('t0', 'Before')} → {lab.get('t1', 'After')}",
                    held_out=bool(cs.get("held_out")))
        meta_p.write_text(json.dumps(meta, indent=2))
    run(sys.executable, "scripts/export_all_sites.py")

    out = Path("results")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    shutil.copy(best, out / best.name)
    shutil.copy(WORK / "model" / "best.txt", out / "best.txt")
    for f in ("spectral_metrics.json", "metrics.json"):
        if (WORK / "model" / f).exists():
            shutil.copy(WORK / "model" / f, out / f)
    labels = {p.parent.name: json.loads(p.read_text()).get("label_px") for p in (WORK / "sites").glob("*/meta.json")}
    (out / "label_summary.json").write_text(json.dumps(labels, indent=2))
    shutil.copytree("demo_data", out / "demo_data")
    (out / "code_version.txt").write_text(code_version() + "\n")
    check = held_out_check()
    (out / "held_out_check.txt").write_text(check + "\n")
    print("\n" + check)
    shutil.make_archive("results", "zip", ".", str(out))
    print("\nDone: results.zip. Download it with: from google.colab import files; "
          "files.download('results.zip')")


if __name__ == "__main__":
    main()
