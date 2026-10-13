"""Blend the U-Net and the LightGBM model per habitat, tuned on validation tiles.

The two models fail differently (in the Oct 13 run LightGBM was far better on Mission Bay seagrass, the U-Net on
Tampa Bay seagrass and on salt marsh), so for each class c the blended probability is
w_c * p_unet + (1 - w_c) * p_lgbm. The weights are picked on validation chips only; the held-out estuaries
are scored afterwards, never used for picking. The blend is switched on only if it beats the better single model
on validation by at least MIN_GAIN.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import BLUE_CARBON_KEYS, CLASSES, IGNORE_INDEX, N_CLASSES

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_GAIN = 0.005
PIXELS_PER_CHIP = 4000
MAX_PIXELS = 3_000_000  # keeps tuning to minutes and memory to ~200 MB


def _confusion(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    k = y != IGNORE_INDEX
    return np.bincount(y[k].astype(np.int64) * N_CLASSES + pred[k], minlength=N_CLASSES ** 2).reshape(N_CLASSES, N_CLASSES)


def blue_iou(cm: np.ndarray) -> float:
    """Mean IoU over the blue carbon classes present in the reference (the release score)."""
    keys = [i for i, c in enumerate(CLASSES) if c.key in BLUE_CARBON_KEYS]
    v = []
    for i in keys:
        if cm[i].sum() == 0:
            continue
        union = cm[i].sum() + cm[:, i].sum() - cm[i, i]
        v.append(cm[i, i] / union if union else 0.0)
    return float(np.mean(v)) if v else 0.0


def score(pu: np.ndarray, ps: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """pu, ps: (N, C) probabilities of sampled pixels; y: (N,) labels."""
    p = w[None, :] * pu + (1 - w[None, :]) * ps
    return blue_iou(_confusion(y, p.argmax(1)))


def tune(pu: np.ndarray, ps: np.ndarray, y: np.ndarray, passes: int = 3) -> tuple[np.ndarray, float]:
    """Coordinate descent over per-class weights on the grid (single classes and pairs); starts from the best of
    U-Net only, LightGBM only and an even split."""
    starts = [np.ones(N_CLASSES, np.float32), np.zeros(N_CLASSES, np.float32), np.full(N_CLASSES, 0.5, np.float32)]
    best_w = max(starts, key=lambda w: score(pu, ps, y, w)).copy()
    best = score(pu, ps, y, best_w)
    moves = [((c,), (g,)) for c in range(N_CLASSES) for g in GRID]
    moves += [((c, d), (g, h)) for c in range(N_CLASSES) for d in range(c + 1, N_CLASSES)
              for g in GRID for h in GRID]  # pairs too: two habitats often have to switch together
    for _ in range(passes):
        changed = False
        for cs, gs in moves:
            w = best_w.copy()
            w[list(cs)] = gs
            s = score(pu, ps, y, w)
            if s > best + 1e-6:
                best, best_w, changed = s, w, True
        if not changed:
            break
    return best_w, best


def sample_chip(ens, rec, rng, tta: bool):
    """Probabilities of both models at a random sample of labelled pixels of one chip."""
    from .data import load_chip, load_chip_anc

    img, lab = load_chip(rec.path)
    pu, ps = ens.probs(img, 256, 0, tta, anc=load_chip_anc(rec.path) if ens.needs_ancillary else None)
    idx = np.flatnonzero(lab.ravel() != IGNORE_INDEX)
    if idx.size == 0:
        return None
    idx = rng.choice(idx, min(PIXELS_PER_CHIP, idx.size), replace=False)
    n = min(pu.shape[0], ps.shape[0])
    return (pu[:n].reshape(n, -1)[:, idx].T.astype(np.float32), ps[:n].reshape(n, -1)[:, idx].T.astype(np.float32),
            lab.ravel()[idx])


def collect(ens, records, split: str, tta: bool, seed: int = 0, log=print):
    rng = np.random.default_rng(seed)
    pus, pss, ys, skipped = [], [], [], 0
    for r in (r for r in records if r.split == split):
        try:
            s = sample_chip(ens, r, rng, tta)
        except ValueError:  # a layer one of the models needs is missing for this chip
            skipped += 1
            continue
        if s:
            pus.append(s[0])
            pss.append(s[1])
            ys.append(s[2])
    if skipped:
        log(f"ensemble: skipped {skipped} {split} chips missing a layer")
    if not ys:
        return None
    pu, ps, y = np.concatenate(pus), np.concatenate(pss), np.concatenate(ys)
    if y.size > MAX_PIXELS:
        k = rng.choice(y.size, MAX_PIXELS, replace=False)
        pu, ps, y = pu[k], ps[k], y[k]
    return pu, ps, y


def per_site(ens, records, split: str = "test", tta: bool = True) -> tuple[dict, np.ndarray]:
    """Full-chip scores per site for the blended model (same form as metrics.json test_per_site), and the pooled
    confusion matrix."""
    from .data import load_chip, load_chip_anc
    from .metrics import confusion, summarize

    out, pooled = {}, np.zeros((N_CLASSES, N_CLASSES), np.int64)
    for site in sorted({r.site for r in records if r.split == split}):
        cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
        for r in (r for r in records if r.split == split and r.site == site):
            img, lab = load_chip(r.path)
            try:
                pred, _ = ens.predict(img, 256, 0, tta, anc=load_chip_anc(r.path) if ens.needs_ancillary else None)
            except ValueError:
                continue
            cm += confusion(lab, pred)
        if cm.sum():
            out[site] = summarize(cm)
            pooled += cm
    return out, pooled


def build(model_dir: Path, records, device: str = "auto", tta: bool = True, log=print, deep: str = "model.pt") -> dict:
    """Tune, score and write model_dir/ensemble.ensemble.json + ensemble_metrics.json. Returns the metrics.
    `deep` is the network blended with LightGBM (the main U-Net, or whichever competitor did best)."""
    from .predictors import EnsemblePredictor, SpectralPredictor, TorchPredictor

    unet, lgbm = TorchPredictor(model_dir / deep, device), SpectralPredictor(model_dir / "spectral.json")
    ens = EnsemblePredictor(unet=unet, lgbm=lgbm, weights=np.ones(N_CLASSES))
    val = collect(ens, records, "val", tta, log=log)
    if val is None:
        raise RuntimeError("no validation chips both models can run on")
    pu, ps, y = val
    s_unet, s_lgbm = score(pu, ps, y, np.ones(N_CLASSES)), score(pu, ps, y, np.zeros(N_CLASSES))
    w, s_blend = tune(pu, ps, y)
    enabled = s_blend >= max(s_unet, s_lgbm) + MIN_GAIN
    log(f"ensemble val blue-carbon IoU: {deep} {s_unet:.3f}  LightGBM {s_lgbm:.3f}  blend {s_blend:.3f} "
        f"-> {'ON' if enabled else 'off (not better enough)'}")
    log("  U-Net share per class: " + ", ".join(f"{c.key} {x:.2f}" for c, x in zip(CLASSES, w, strict=False)))
    ens.weights = w
    from .metrics import summarize

    sites, cm = per_site(ens, records, "test", tta)
    metrics = {"val_blue_iou": {"unet": s_unet, "spectral": s_lgbm, "ensemble": s_blend}, "enabled": bool(enabled),
               "weights": {c.key: float(x) for c, x in zip(CLASSES, w, strict=False)}, "min_gain": MIN_GAIN,
               "test": summarize(cm) if cm.sum() else None, "test_confusion": cm.tolist(), "test_per_site": sites}
    metrics["deep"] = deep
    spec = {"unet": deep, "spectral": "spectral.json", "weights": [float(x) for x in w], "metrics": metrics}
    (model_dir / "ensemble.ensemble.json").write_text(json.dumps(spec, indent=2))
    (model_dir / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
