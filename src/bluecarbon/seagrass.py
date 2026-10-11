"""Seagrass specialist: a second opinion on every pixel the main model calls water or seagrass.

Seagrass is the weakest habitat (about 0.37 IoU): meadows in murky or deeper water look like water in a
yearly median. The specialist is a binary LightGBM model trained only on water and seagrass pixels, with
features built for seeing the seafloor:

  clear-water bands and their depth-insensitive log ratios (Lyzenga / Stumpf),
  darkness of the bottom (meadows are darker than sand), local texture at two scales (meadows are patchy),
  and distance from land (meadows hug the shore).

It only overrides the main model where the main model said water or seagrass, and only with a threshold
tuned on validation chips. If it does not improve seagrass IoU on validation it is saved switched off, and
the held-out estuaries report both numbers, so nothing is published that the data does not support.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import IGNORE_INDEX, KEY_TO_ID

WATER, SEAGRASS = KEY_TO_ID["water"], KEY_TO_ID["seagrass"]
FEATURES = ["B2", "B3", "B4", "B8", "NDWI", "MNDWI", "B2_CLEAR", "B3_CLEAR", "B4_CLEAR", "LN_B2_B3_CLEAR",
            "LN_B3_B4_CLEAR", "DARK_CLEAR", "B3C_MEAN5", "B3C_STD5", "B3C_MEAN15", "B3C_STD15", "LN23_MEAN15",
            "DIST_LAND"]
PIX_PER_CHIP = 1500
MIN_GAIN = 0.01  # seagrass IoU the specialist must add on validation to be switched on


def seagrass_features(img: np.ndarray, anc: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt, uniform_filter

    from .features import compute_features

    names = ["B2", "B3", "B4", "B8", "NDWI", "MNDWI", "B2_CLEAR", "B3_CLEAR", "B4_CLEAR", "LN_B2_B3_CLEAR",
             "LN_B3_B4_CLEAR"]
    f = compute_features(img, anc=anc, names=names)
    b3c = f[names.index("B3_CLEAR")]
    dark = f[names.index("B2_CLEAR")] + b3c

    def mean_std(a, k):
        m = uniform_filter(a, k)
        return m, np.sqrt(np.clip(uniform_filter(a * a, k) - m * m, 0, None))

    m5, s5 = mean_std(b3c, 5)
    m15, s15 = mean_std(b3c, 15)
    ln15 = uniform_filter(f[names.index("LN_B2_B3_CLEAR")], 15)
    land = f[names.index("MNDWI")] < 0
    dist = np.clip(distance_transform_edt(~land) * 10 / 3000, 0, 1) if land.any() else np.ones_like(b3c)
    return np.concatenate([f, np.stack([dark, m5, s5, m15, s15, ln15, dist])]).astype(np.float32)


def waterlike(main_cls: np.ndarray) -> np.ndarray:
    return np.isin(main_cls, [WATER, SEAGRASS])


def fuse(main_cls: np.ndarray, p_seagrass: np.ndarray, threshold: float) -> np.ndarray:
    out = main_cls.copy()
    m = waterlike(main_cls)
    out[m] = np.where(p_seagrass[m] >= threshold, SEAGRASS, WATER)
    return out


def _iou(ref, pred, k):
    ok = ref != IGNORE_INDEX
    a, b = (ref == k) & ok, (pred == k) & ok
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else None


class SeagrassSpecialist:
    def __init__(self, booster, threshold: float, enabled: bool, metrics: dict | None = None):
        self.booster, self.threshold, self.enabled, self.metrics = booster, threshold, enabled, metrics or {}

    def proba(self, img: np.ndarray, anc: np.ndarray) -> np.ndarray:
        x = seagrass_features(img, anc)
        n, h, w = x.shape
        return self.booster.predict(x.reshape(n, -1).T).reshape(h, w).astype(np.float32)

    def apply(self, main_cls: np.ndarray, img: np.ndarray, anc: np.ndarray | None) -> np.ndarray:
        if not self.enabled or anc is None:
            return main_cls
        return fuse(main_cls, self.proba(img, anc), self.threshold)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"features": FEATURES, "booster": self.booster.model_to_string(),
                                          "threshold": self.threshold, "enabled": self.enabled,
                                          "metrics": self.metrics}))

    @classmethod
    def load(cls, path: Path) -> SeagrassSpecialist:
        import lightgbm as lgb

        d = json.loads(Path(path).read_text())
        return cls(lgb.Booster(model_str=d["booster"]), d["threshold"], d["enabled"], d.get("metrics"))


def specialist_path_for(model_path: str | Path) -> Path:
    """The specialist travels next to the model it was tuned with: model.pt -> model.seagrass.json."""
    p = Path(model_path)
    return p.with_name(p.stem + ".seagrass.json")


def train_specialist(records, predict_main, seed: int = 0) -> SeagrassSpecialist:
    """records: chip records (with ancillary); predict_main(img, anc) -> main model class map for a chip."""
    import lightgbm as lgb

    from .data import load_chip, load_chip_anc

    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for r in records:
        if r.split != "train":
            continue
        img, lab = load_chip(r.path)
        anc = load_chip_anc(r.path)
        if anc is None:
            continue
        m = np.isin(lab, [WATER, SEAGRASS])
        if not (lab == SEAGRASS).any() and rng.random() > 0.3:  # keep some all-water chips as negatives
            continue
        idx = np.flatnonzero(m.ravel())
        if idx.size == 0:
            continue
        pick = idx if idx.size <= PIX_PER_CHIP else rng.choice(idx, PIX_PER_CHIP, replace=False)
        x = seagrass_features(img, anc)
        xs.append(x.reshape(x.shape[0], -1)[:, pick].T)
        ys.append((lab.ravel()[pick] == SEAGRASS).astype(np.int8))
    if not xs:
        raise RuntimeError("no water/seagrass training pixels with ancillary layers")
    X, y = np.concatenate(xs), np.concatenate(ys)
    pos = max(1, int(y.sum()))
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, min_data_in_leaf=50, scale_pos_weight=float((len(y) - pos) / pos) ** 0.5, verbose=-1)
    booster = lgb.train(params, lgb.Dataset(X, y), num_boost_round=400)
    sp = SeagrassSpecialist(booster, 0.5, True)

    def collect(split):
        refs, mains, probs = [], [], []
        for r in records:
            if r.split != split:
                continue
            img, lab = load_chip(r.path)
            anc = load_chip_anc(r.path)
            if anc is None:
                continue
            refs.append(lab)
            mains.append(predict_main(img, anc))
            probs.append(sp.proba(img, anc))
        return refs, mains, probs

    def score(refs, mains, probs, t):
        ref = np.concatenate([a.ravel() for a in refs])
        pred = np.concatenate([(m if t is None else fuse(m, p, t)).ravel() for m, p in zip(mains, probs, strict=True)])
        return _iou(ref, pred, SEAGRASS), _iou(ref, pred, WATER)

    refs, mains, probs = collect("val")
    base = score(refs, mains, probs, None)
    grid = [round(t, 2) for t in np.arange(0.2, 0.9, 0.05)]
    scored = {t: score(refs, mains, probs, t) for t in grid}
    best = max(grid, key=lambda t: scored[t][0] or 0)
    gain = (scored[best][0] or 0) - (base[0] or 0)
    sp.threshold, sp.enabled = best, gain >= MIN_GAIN
    tr, tm, tp = collect("test")
    t_base, t_fused = score(tr, tm, tp, None), score(tr, tm, tp, best)
    sp.metrics = {"val_seagrass_iou_main": base[0], "val_seagrass_iou_fused": scored[best][0],
                  "val_water_iou_main": base[1], "val_water_iou_fused": scored[best][1], "threshold": best,
                  "test_seagrass_iou_main": t_base[0], "test_seagrass_iou_fused": t_fused[0],
                  "test_water_iou_main": t_base[1], "test_water_iou_fused": t_fused[1],
                  "enabled": sp.enabled, "train_pixels": int(len(y)), "train_seagrass_pixels": pos}
    return sp
