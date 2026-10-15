"""Blend test on the models already trained (run 10, saved in Drive): do simple averages of several models beat the
live model on the 7 held-out estuaries?

No weights are tuned on the held-out estuaries: every blend here is a plain average of class probabilities, fixed
in advance. Writes blend_test.txt (same layout and PUBLISH / REVIEW rule as `bluecarbon compare`) and copies it to
Drive (runs/default/sites/_analysis/). Runs in about 30-60 minutes on a Colab GPU.

    python scripts/blend_test.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

WORK = Path("runs/default")
MODEL = WORK / "model"
BACKUP = WORK / "sites" / "_trained_models"  # Google Drive
OUT = WORK / "sites" / "_analysis"

MEMBERS = {"unet": "model.pt", "lightgbm": "spectral.json", "unet_convnext": "unet_convnext.pt",
           "segformer": "segformer.pt", "prithvi": "prithvi.pt"}
BLENDS = {  # fixed before looking at blend results; each is an equal-weight average of probabilities
    "all5": ["unet", "lightgbm", "unet_convnext", "segformer", "prithvi"],
    "deep4": ["unet", "unet_convnext", "segformer", "prithvi"],
    "prithvi+segformer": ["prithvi", "segformer"],
    "prithvi+segformer+unet": ["prithvi", "segformer", "unet"],
    "prithvi+lightgbm": ["prithvi", "lightgbm"],
}


def probabilities(name, pr, bands, anc, cfg):
    from bluecarbon.predict import predict_array

    if name == "lightgbm":
        return pr.model.predict_proba(bands, anc=pr._check_anc(anc)).astype(np.float16)
    _, _, p = predict_array(pr.model, pr.norm, bands, cfg.predict.tile, cfg.predict.overlap, cfg.predict.tta,
                            anc=pr._check_anc(anc), names=pr.features, class_bias=pr.class_bias, return_prob=True)
    return p.astype(np.float16)


def main() -> None:
    import rasterio

    from bluecarbon.config import load_config
    from bluecarbon.features import valid_mask
    from bluecarbon.metrics import confusion, summarize
    from bluecarbon.predictors import SpectralPredictor, TorchPredictor, load_predictor, read_ancillary
    from bluecarbon.priors import apply_to_classes, raster_center_lat
    from bluecarbon.schema import BLUE_CARBON_KEYS, IGNORE_INDEX

    if not all((MODEL / f).exists() for f in MEMBERS.values()):
        if not BACKUP.exists():
            raise SystemExit("No trained models: runs/default/model and the Drive copy are both missing.")
        print("copying the trained models from Drive ...", flush=True)
        shutil.rmtree(MODEL, ignore_errors=True)
        shutil.copytree(BACKUP, MODEL)
    cfg = load_config("configs/default.yaml")
    dev = cfg.train.device
    preds = {n: (SpectralPredictor(MODEL / f) if f.endswith(".json") else TorchPredictor(MODEL / f, dev))
             for n, f in MEMBERS.items()}
    live = load_predictor(Path("models") / Path("models/current.txt").read_text().strip(), dev)
    sites = [s["name"] for s in yaml.safe_load(Path("configs/sites.yaml").read_text())["sites"] if s.get("role") == "test"]
    names = ["live", *MEMBERS, *BLENDS]
    rows = {}
    for site in sites:
        d = WORK / "sites" / site
        if not (d / "image.tif").exists():
            print(f"skip {site}: no image", flush=True)
            continue
        print(f"[{site}] ...", flush=True)
        with rasterio.open(d / "image.tif") as im:
            bands, tr, crs = im.read(), im.transform, im.crs
        with rasterio.open(d / "label.tif") as lb:
            ref = lb.read(1)
        lat = raster_center_lat(tr, crs, *ref.shape)
        anc = read_ancillary(d / "image.tif", bands.shape[1:])
        ok = valid_mask(bands)

        def score(cls, ref=ref, lat=lat):
            return summarize(confusion(ref, apply_to_classes(cls, lat)))["iou"]

        r = {"live": score(live.predict(bands, cfg.predict.tile, cfg.predict.overlap, cfg.predict.tta,
                                        anc=anc if live.needs_ancillary else None)[0])}
        probs = {}
        for n, pr in preds.items():
            probs[n] = probabilities(n, pr, bands, anc, cfg)
            cls = probs[n].astype(np.float32).argmax(0).astype(np.uint8)
            cls[~ok] = IGNORE_INDEX
            r[n] = score(cls)
        for b, members in BLENDS.items():
            c = min(probs[m].shape[0] for m in members)
            p = sum(probs[m][:c].astype(np.float32) for m in members) / len(members)
            cls = p.argmax(0).astype(np.uint8)
            cls[~ok] = IGNORE_INDEX
            r[b] = score(cls)
        rows[site] = r
        del probs

    lines = ["BLEND TEST on the 7 held-out estuaries (IoU). Blends are equal-weight averages fixed in advance;",
             "nothing is tuned on these estuaries. Verdict rule as for publishing: mean >= live and no drop > 0.05.", "",
             f"  {'estuary':20s} {'habitat':10s}" + "".join(f" {n[:10]:>10s}" for n in names)]
    vals = {n: [] for n in names}
    for site, r in rows.items():
        for k in BLUE_CARBON_KEYS:
            if r["live"].get(k) is None and all(r[n].get(k) is None for n in names[1:]):
                continue
            v = {n: r[n].get(k) or 0.0 for n in names}
            for n in names:
                vals[n].append(v[n])
            lines.append(f"  {site:20s} {k:10s}" + "".join(f" {v[n]:10.2f}" for n in names))
    lines += ["", "  model / blend              mean   largest drop vs live   verdict"]
    lv = np.array(vals["live"])
    for n in names:
        a = np.array(vals[n])
        drop = float((a - lv).min()) if n != "live" else 0.0
        ok = a.mean() >= lv.mean() and drop >= -0.05
        verdict = "-" if n == "live" else ("PUBLISH" if ok else "REVIEW")
        lines.append(f"  {n:24s} {a.mean():6.3f}   {drop:+6.2f}                {verdict}")
    text = "\n".join(lines)
    print("\n" + text, flush=True)
    Path("blend_test.txt").write_text(text + "\n")
    OUT.mkdir(parents=True, exist_ok=True)
    shutil.copy("blend_test.txt", OUT / "blend_test.txt")
    print(f"\nSaved to Google Drive: {OUT / 'blend_test.txt'}", flush=True)


if __name__ == "__main__":
    main()
