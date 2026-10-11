"""Install the results of a Colab training run into the repo in one step.

    python scripts/ingest_results.py --model ~/Downloads/model.pt --demo ~/Downloads/demo_all.zip

What it does:
  1. Copies the trained model into models/ and makes it the app's default (models/current.txt).
  2. Replaces demo_data/ with the exported site pages and refreshes their model info.
  3. Removes the single-bay pilot (model, data, scripts).
  4. Writes the held-out test results into README.md and renders the figures in docs/img/.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np

from bluecarbon.predictors import is_spectral, model_card
from bluecarbon.schema import CLASS_KEYS, CLASSES

ROOT = Path(__file__).resolve().parents[1]
MODEL_INFO = {
    "name": "BlueCarbon-AI v2",
    "pilot": False,
    "evaluation": "non-overlapping 5 km spatial blocks plus five estuaries never used in training "
                  "(Mission Bay, Plum Island, Moreton Bay, Shoalwater Bay, Tampa Bay)",
    "training_data": "Sentinel-2 2021 composites and clear-water images of 39 coastal sites on six continents, "
                     "labelled from ESA WorldCover, the GWL_FCS30 wetland map, Murray et al. tidal flats and "
                     "the Allen Coral Atlas, plus official seagrass surveys (Florida FWC, Moreton Bay)",
    "uses_context": True,
}
PILOT_PATHS = ["models/pilot_spectral_mission_bay_2018.json", "data/pilot", "scripts/pilot_spectral.py",
               "scripts/pilot_mission_bay.py", "docs/EXPERIMENTS.md", "docs/img/pilot_confusion.png"]


def install_model(src: Path) -> tuple[Path, dict]:
    spectral = is_spectral(src)
    dst = ROOT / "models" / ("bluecarbon_v2.json" if spectral else "bluecarbon_v2.pt")
    if spectral:
        d = json.loads(src.read_text())
        d["extra"] = {**d.get("extra", {}), **MODEL_INFO}
        dst.write_text(json.dumps(d))
    else:
        import torch

        ck = torch.load(src, map_location="cpu", weights_only=False)
        ck["state_dict"] = {k: (v.half() if v.is_floating_point() else v) for k, v in ck["state_dict"].items()}
        ck["extra"] = {**ck.get("extra", {}), **MODEL_INFO}
        torch.save(ck, dst)
    mb = dst.stat().st_size / 1e6
    if mb > 95:
        raise SystemExit(f"{dst.name} is {mb:.0f} MB, too big for GitHub. Attach it to a release and set MODEL_URL.")
    (ROOT / "models" / "current.txt").write_text(dst.name + "\n")
    print(f"model -> {dst.relative_to(ROOT)} ({mb:.1f} MB)")
    from bluecarbon.seagrass import specialist_path_for

    for f, to in ((specialist_path_for(src), specialist_path_for(dst)),
                  (src.with_name("history_landsat.json"), ROOT / "models" / "history_landsat.json"),
                  (src.with_name("site_biomass.json"), ROOT / "data" / "site_biomass.json")):
        if f.exists():  # companions from the same run: seagrass specialist, history model, biomass summaries
            shutil.copy(f, to)
            print(f"{f.name} -> {to.relative_to(ROOT)}")
        elif to.exists() and to.suffix == ".json" and "seagrass" in to.name:
            to.unlink()  # a specialist tuned for the previous model must not ride along with a new one
    return dst, model_card(dst)


def install_demo(zip_path: Path, card: dict) -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(zip_path).extractall(tmp)
        src = next(Path(tmp).rglob("demo_data"), None)
        if src is None:
            raise SystemExit(f"{zip_path} has no demo_data/ folder")
        dst = ROOT / "demo_data"
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst)
        img = next(Path(tmp).rglob("docs/img"), None)
        if img is not None:
            for f in img.glob("*.png"):
                shutil.copy(f, ROOT / "docs" / "img" / f.name)
    model = {"arch": card["arch"], "encoder": card["encoder"], "kind": card["kind"],
             "test": card["metrics"].get("test"), **MODEL_INFO}
    names = []
    for meta_p in sorted((ROOT / "demo_data").glob("*/meta.json")):
        meta = json.loads(meta_p.read_text())
        meta["model"] = model
        meta_p.write_text(json.dumps(meta, indent=2))
        names.append(meta["title"])
    print(f"demo sites: {len(names)}")
    return names


def remove_pilot() -> None:
    for rel in PILOT_PATHS:
        p = ROOT / rel
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def figures(card: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from bluecarbon.viz import plot_confusion

    cm = card["metrics"].get("test_confusion")
    if cm:
        cm = np.asarray(cm)
        keep = [i for i in range(len(CLASS_KEYS)) if cm[i].sum() > 0]
        plot_confusion(cm[np.ix_(keep, keep)], [CLASSES[i].name for i in keep], ROOT / "docs/img/confusion_test.png",
                       "Held-out test: reference vs predicted (row-normalized)")
    # hero: up to three held-out / change sites side by side
    picks = []
    for meta_p in sorted((ROOT / "demo_data").glob("*/meta.json")):
        meta = json.loads(meta_p.read_text())
        if meta.get("held_out") or meta.get("kind") == "change":
            picks.append((meta_p.parent, meta))
    picks = picks[:3]
    if not picks:
        return
    fig, ax = plt.subplots(2, len(picks), figsize=(5 * len(picks), 9.5), dpi=120, squeeze=False)
    for j, (d, meta) in enumerate(picks):
        tag = "_t1" if meta.get("kind") == "change" else ""
        rgb = mpimg.imread(d / f"rgb{tag}.png")
        ax[0, j].imshow(rgb)
        ax[1, j].imshow(rgb)
        ax[1, j].imshow(mpimg.imread(d / f"classes{tag}.png"), alpha=0.85)
        ax[0, j].set_title(meta["title"], fontsize=11)
        for a in ax[:, j]:
            a.axis("off")
    fig.legend(handles=[Patch(color=c.color, label=c.name) for c in CLASSES], loc="lower center", ncol=6,
               frameon=False, fontsize=10)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / "docs/img/hero.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def readme(card: dict, sites: list[str]) -> None:
    t = card["metrics"].get("test")
    if not t:
        print("no test metrics in model file; README results not updated")
        return
    rows = []
    for c in CLASSES:
        iou = t["iou"].get(c.key)
        if iou is None:
            continue
        rows.append(f"| {c.name}{' ◆' if c.blue_carbon else ''} | {iou:.2f} | {t['f1'][c.key]:.2f} | "
                    f"{t['support_px'][c.key]:,} |")
    model_name = "LightGBM spectral model" if card["kind"] == "spectral-lgbm" else f"U-Net ({card['encoder']})"
    block = "\n".join([
        "<!-- results:start -->",
        "## Results",
        "",
        f"Scores are from areas the model **never saw during training**: held-out 5 km blocks and three entire "
        f"estuaries. The selected model is the **{model_name}**, which beat the alternative on the blue carbon "
        "habitats (◆).",
        "",
        "| Habitat | IoU | F1 | Test pixels |",
        "|---|---:|---:|---:|",
        *rows,
        f"| **Mean** | **{t['mIoU']:.2f}** | **{t['macro_f1']:.2f}** | |",
        "",
        "IoU (intersection over union) measures how well the predicted map overlaps the reference map, where "
        "1.0 is a perfect match. Overall pixel accuracy "
        f"({t['overall_accuracy']:.0%}) is not the headline metric, because it is dominated by open water.",
        "",
        "![Held-out sites: satellite image (top) and BlueCarbon-AI habitat map (bottom)](docs/img/hero.png)",
        "",
        f"The live app includes {len(sites)} mapped sites, including a 2018 → 2024 change analysis of Mission Bay.",
        "<!-- results:end -->",
    ])
    p = ROOT / "README.md"
    s = p.read_text()
    if "<!-- results:start -->" in s:
        a, b = s.index("<!-- results:start -->"), s.index("<!-- results:end -->") + len("<!-- results:end -->")
        s = s[:a] + block + s[b:]
    else:
        s = s.replace("## How it works", block + "\n\n## How it works", 1)
    if (ROOT / "docs/img/hero.png").exists():
        s = s.replace("![Mission Bay: satellite image, false-color infrared, and the BlueCarbon-AI habitat map]"
                      "(docs/img/mission_bay_hero.png)\n\n", "")
    p.write_text(s)
    print("README results updated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, help="model file (not needed with --results)")
    ap.add_argument("--demo", type=Path, help="demo_all.zip / demo_data.zip (not needed with --results)")
    ap.add_argument("--results", type=Path, help="results.zip from scripts/retrain.py")
    ap.add_argument("--keep-pilot", action="store_true")
    a = ap.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        if a.results:
            zipfile.ZipFile(a.results).extractall(tmp)
            best_txt = next(Path(tmp).rglob("best.txt"))
            a.model = best_txt.parent / best_txt.read_text().split()[1]
            a.demo = a.results
        if not (a.model and a.demo):
            raise SystemExit("give --results, or both --model and --demo")
        _, card = install_model(a.model)
    sites = install_demo(a.demo, card)
    if not a.keep_pilot:
        remove_pilot()
    figures(card)
    readme(card, sites)
    if (ROOT / "docs/img/hero.png").exists() and (ROOT / "docs/img/mission_bay_hero.png").exists():
        (ROOT / "docs/img/mission_bay_hero.png").unlink()


if __name__ == "__main__":
    main()
