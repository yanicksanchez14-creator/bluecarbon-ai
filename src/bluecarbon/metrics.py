"""Segmentation metrics from a confusion matrix, plus error-adjusted area estimation.

Confusion matrices are indexed [reference, predicted].
"""

from __future__ import annotations

import numpy as np

from .schema import CLASS_KEYS, IGNORE_INDEX, N_CLASSES


def confusion(ref: np.ndarray, pred: np.ndarray, n: int = N_CLASSES) -> np.ndarray:
    m = ref != IGNORE_INDEX
    r, p = ref[m].astype(np.int64), pred[m].astype(np.int64)
    return np.bincount(r * n + p, minlength=n * n).reshape(n, n)


def summarize(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    ref_tot, pred_tot = cm.sum(1), cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (ref_tot + pred_tot - tp)
        precision = tp / pred_tot  # user's accuracy
        recall = tp / ref_tot  # producer's accuracy
        f1 = 2 * precision * recall / (precision + recall)
    # a class that exists in the reference but is never (correctly) predicted scores 0, not NaN
    precision = np.where((pred_tot == 0) & (ref_tot > 0), 0.0, precision)
    f1 = np.where((tp == 0) & (ref_tot > 0), 0.0, f1)
    total = cm.sum()
    oa = tp.sum() / total if total else float("nan")
    pe = (ref_tot * pred_tot).sum() / total**2 if total else float("nan")
    kappa = (oa - pe) / (1 - pe) if total and pe < 1 else float("nan")
    present = ref_tot > 0

    def per(v):
        return {k: (None if not present[i] or np.isnan(v[i]) else round(float(v[i]), 4))
                for i, k in enumerate(CLASS_KEYS)}

    return {
        "overall_accuracy": round(float(oa), 4),
        "kappa": round(float(kappa), 4),
        "mIoU": round(float(np.nanmean(iou[present])), 4) if present.any() else None,
        "macro_f1": round(float(np.nanmean(f1[present])), 4) if present.any() else None,
        "iou": per(iou),
        "f1": per(f1),
        "precision": per(precision),
        "recall": per(recall),
        "support_px": {k: int(ref_tot[i]) for i, k in enumerate(CLASS_KEYS)},
    }


def error_adjusted_area(cm: np.ndarray, mapped_px: np.ndarray, px_area_ha: float) -> dict:
    """Olofsson et al. (2014) stratified estimator of class area with 95% CI.

    `cm` is a validation confusion matrix [reference, map] and `mapped_px` the per-class
    pixel counts of the map being reported. Returns hectares per class.
    """
    cm = cm.astype(np.float64)
    W = mapped_px / mapped_px.sum()  # map-class proportions (strata weights)
    n_map = cm.sum(0)  # samples per map class
    with np.errstate(divide="ignore", invalid="ignore"):
        pij = np.where(n_map > 0, W * cm / n_map, 0.0)  # estimated proportion (ref i, map j)
    p_ref = pij.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(n_map > 1, cm / n_map, 0.0)
        var = np.where(n_map > 1, W**2 * frac * (1 - frac) / (n_map - 1), 0.0).sum(1)
    total_ha = mapped_px.sum() * px_area_ha
    se = np.sqrt(var) * total_ha
    out = {
        k: {
            "mapped_ha": round(float(mapped_px[i] * px_area_ha), 2),
            "adjusted_ha": round(float(p_ref[i] * total_ha), 2),
            "ci95_ha": round(float(1.96 * se[i]), 2),
        }
        for i, k in enumerate(CLASS_KEYS)
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.diag(cm) / cm.sum(0)
        rec = np.diag(cm) / cm.sum(1)
    return cap_adjusted(out, dict(zip(CLASS_KEYS, np.nan_to_num(prec).tolist(), strict=True)),
                        dict(zip(CLASS_KEYS, np.nan_to_num(rec).tolist(), strict=True)))


def cap_adjusted(adj: dict, precision: dict, recall: dict) -> dict:
    """Keep the blue carbon area correction within what the model's measured accuracy can support.

    The stratified estimator moves area between classes using test-set error rates pooled over all sites.
    At a site that is mostly open water, a pooled 'water that was really seagrass' rate turns a large lake
    of water into thousands of hectares of seagrass nobody mapped (e.g. 721 ha mapped -> 6,685 ha). The
    most the map can be under-reporting a class is mapped x precision / recall (the hits it finds, scaled
    up by the share of the class it is known to miss), so the corrected blue carbon area is capped there.
    """
    from .schema import BLUE_CARBON_KEYS

    out = {k: dict(v) for k, v in adj.items()}
    for k in BLUE_CARBON_KEYS:
        v, p, r = out.get(k), precision.get(k) or 0.0, recall.get(k) or 0.0
        if not v:
            continue
        cap = v["mapped_ha"] * p / r if r > 0 else v["mapped_ha"]
        if v["adjusted_ha"] > cap:
            scale = cap / v["adjusted_ha"] if v["adjusted_ha"] else 0.0
            v.update(adjusted_ha=round(cap, 2), ci95_ha=round(v["ci95_ha"] * scale, 2), capped=True)
    return out
