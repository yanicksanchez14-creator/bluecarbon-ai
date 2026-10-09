"""Spectral-context model: gradient-boosted trees on per-pixel spectra + neighbourhood context.

Why this exists: with limited labels (a few thousand marsh / seagrass pixels), a per-pixel
LightGBM classifier beats a U-Net trained from scratch by a wide margin. On the Mission Bay
spatial cross-validation it lifted salt marsh IoU from 0.13 to 0.72 and seagrass from 0.00 to
0.53 (see docs/EXPERIMENTS.md). Each pixel sees:

  * the 14 standard features (10 bands + NDVI, NDWI, MNDWI, NDMI)
  * 5 water-column / pigment ratios that help separate submerged seagrass from open water
    (log B2/B3, log B3/B4, log B2/B4 are the depth-invariant terms of Lyzenga's method)
  * local means of all of the above at 3x3 and 9x9 pixels (30 m / 90 m context)

Class probabilities are smoothed with a 5x5 window before the argmax, which removes
salt-and-pepper noise without blurring habitat edges much.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

from .features import compute_features, valid_mask
from .schema import CLASS_KEYS, IGNORE_INDEX, N_CLASSES, compatible

EPS = 1e-4
CONTEXT_SCALES = (3, 9)
SMOOTH = 5
KIND = "spectral-lgbm"


def ratio_features(feats: np.ndarray) -> np.ndarray:
    b2, b3, b4, b5 = feats[0], feats[1], feats[2], feats[3]
    lg = np.log
    return np.stack([
        lg(b2 + EPS) - lg(b3 + EPS),
        lg(b3 + EPS) - lg(b4 + EPS),
        lg(b2 + EPS) - lg(b4 + EPS),
        (b5 - b4) / (b5 + b4 + EPS),  # red-edge chlorophyll
        b3 / (b2 + EPS),
    ]).astype(np.float32)


def pixel_features(bands: np.ndarray, anc: np.ndarray | None = None, names: list[str] | None = None) -> np.ndarray:
    """(C,H,W) raw S2 bands (+ optional ancillary) -> (N,H,W) feature stack for the spectral model."""
    f = compute_features(bands, anc=anc, names=names)
    base = np.concatenate([f, ratio_features(f)])
    out = [base]
    for s in CONTEXT_SCALES:
        out.append(np.stack([uniform_filter(x, s, mode="reflect") for x in base]))
    return np.concatenate(out).astype(np.float32)


def sample_pixels(bands: np.ndarray, label: np.ndarray, per_class: int, rng,
                  anc: np.ndarray | None = None, names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced pixel sample from one scene / chip."""
    x = pixel_features(bands, anc, names)
    ok = (label != IGNORE_INDEX) & valid_mask(bands)
    xs, ys = [], []
    for c in range(N_CLASSES):
        idx = np.flatnonzero(ok.ravel() & (label.ravel() == c))
        if idx.size == 0:
            continue
        pick = idx if idx.size <= per_class else rng.choice(idx, per_class, replace=False)
        xs.append(x.reshape(x.shape[0], -1)[:, pick].T)
        ys.append(np.full(pick.size, c, np.int64))
    if not xs:
        return np.zeros((0, x.shape[0]), np.float32), np.zeros(0, np.int64)
    return np.concatenate(xs), np.concatenate(ys)


class SpectralModel:
    """LightGBM habitat classifier with the same predict() contract as the U-Net."""

    kind = KIND

    def __init__(self, booster=None, metrics: dict | None = None, extra: dict | None = None,
                 uses_ancillary: bool = False, features: list[str] | None = None):
        from .features import FEATURE_NAMES, FEATURE_NAMES_ANC_V1

        self.booster = booster
        self.metrics = metrics or {}
        self.extra = extra or {}
        self.uses_ancillary = uses_ancillary
        # models saved before feature names were recorded used the v1 context layers
        self.features = features or (FEATURE_NAMES_ANC_V1 if uses_ancillary else FEATURE_NAMES)

    # ------------------------------------------------------------------ training
    @staticmethod
    def default_params() -> dict:
        return dict(objective="multiclass", num_class=N_CLASSES, learning_rate=0.05, num_leaves=31,
                    bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.7, min_data_in_leaf=40,
                    verbose=-1, num_threads=0)

    def fit(self, X: np.ndarray, y: np.ndarray, n_estimators: int = 300, params: dict | None = None,
            weighting: str = "sqrt_inverse") -> SpectralModel:
        import lightgbm as lgb

        cnt = np.bincount(y, minlength=N_CLASSES).astype(float)
        if weighting == "none":
            w = None
        else:
            cw = 1 / (np.sqrt(np.maximum(cnt, 1)) if weighting == "sqrt_inverse" else np.maximum(cnt, 1))
            w = cw[y]
        p = {**self.default_params(), **(params or {})}
        self.booster = lgb.train(p, lgb.Dataset(X, y, weight=w), num_boost_round=n_estimators)
        return self

    # ------------------------------------------------------------------ inference
    def _proba_block(self, bands: np.ndarray, anc: np.ndarray | None) -> np.ndarray:
        x = pixel_features(bands, anc if self.uses_ancillary else None, self.features)
        n, h, w = x.shape
        raw = self.booster.predict(x.reshape(n, -1).T).T
        p = np.zeros((N_CLASSES, h * w), np.float32)
        p[: raw.shape[0]] = raw
        return p.reshape(N_CLASSES, h, w)

    def predict_proba(self, bands: np.ndarray, anc: np.ndarray | None = None, strip: int = 512,
                      max_block_px: int = 2_000_000) -> np.ndarray:
        """Class probabilities. Large scenes are processed in row strips (with a halo wide enough for the
        context filters and smoothing) so memory stays bounded."""
        _, H, W = bands.shape
        halo = max(CONTEXT_SCALES) // 2 + SMOOTH // 2 + 1
        if H * W <= max_block_px:
            p = self._proba_block(bands, anc)
        else:
            p = np.zeros((N_CLASSES, H, W), np.float32)
            for y0 in range(0, H, strip):
                a, b = max(0, y0 - halo), min(H, y0 + strip + halo)
                blk = self._proba_block(bands[:, a:b], None if anc is None else anc[:, a:b])
                p[:, y0 : min(H, y0 + strip)] = blk[:, y0 - a : y0 - a + min(strip, H - y0)]
        if SMOOTH > 1:
            p = np.stack([uniform_filter(c, SMOOTH, mode="reflect") for c in p])
        return p

    def predict(self, bands: np.ndarray, anc: np.ndarray | None = None, **_) -> tuple[np.ndarray, np.ndarray]:
        prob = self.predict_proba(bands, anc)
        cls = prob.argmax(0).astype(np.uint8)
        cls[~valid_mask(bands)] = IGNORE_INDEX
        return cls, prob.max(0)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({
            "kind": KIND, "classes": CLASS_KEYS, "context_scales": list(CONTEXT_SCALES), "smooth": SMOOTH,
            "metrics": self.metrics, "extra": self.extra, "uses_ancillary": self.uses_ancillary,
            "features": self.features,
            "booster": self.booster.model_to_string(),
        }))

    @classmethod
    def load(cls, path: str | Path) -> SpectralModel:
        import lightgbm as lgb

        d = json.loads(Path(path).read_text())
        if d.get("kind") != KIND or not compatible(d["classes"]):
            raise ValueError(f"{path} is not a compatible spectral model")
        m = cls(lgb.Booster(model_str=d["booster"]), d.get("metrics"), d.get("extra"), d.get("uses_ancillary", False),
                d.get("features"))
        m.n_model_classes = len(d["classes"])
        return m

    @property
    def info(self) -> dict:
        return {"arch": "LightGBM", "encoder": "spectral-context", "kind": KIND}


def train_spectral(records, out_dir: str | Path, per_class_per_chip: int = 400, n_estimators: int = 300,
                   seed: int = 42, log=print) -> dict:
    """Train on chip records (same index as the U-Net) and evaluate on val / test chips."""
    from .data import chip_feature_names, chips_have_ancillary, load_chip, load_chip_anc
    from .metrics import confusion, summarize

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tr = [r for r in records if r.split == "train"]
    use_anc = chips_have_ancillary(records)
    names = chip_feature_names(records, use_anc)
    xs, ys = [], []
    for r in tr:
        img, lab = load_chip(r.path)
        x, y = sample_pixels(img, lab, per_class_per_chip, rng, load_chip_anc(r.path) if use_anc else None, names)
        xs.append(x)
        ys.append(y)
    X, y = np.concatenate(xs), np.concatenate(ys)
    log(f"spectral model: {len(y):,} training pixels from {len(tr)} chips, class counts "
        f"{np.bincount(y, minlength=N_CLASSES).tolist()}")
    model = SpectralModel(uses_ancillary=use_anc, features=names).fit(X, y, n_estimators=n_estimators)

    def evaluate(split, site=None):
        cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
        for r in (r for r in records if r.split == split and (site is None or r.site == site)):
            img, lab = load_chip(r.path)
            pred, _ = model.predict(img, anc=load_chip_anc(r.path) if use_anc else None)
            cm += confusion(lab, pred)
        return cm

    results = {}
    for split in ("val", "test"):
        cm = evaluate(split)
        if cm.sum():
            results[split] = summarize(cm)
            results[f"{split}_confusion"] = cm.tolist()
            log(f"spectral {split}: mIoU {results[split]['mIoU']}  IoU {results[split]['iou']}")
    results["test_per_site"] = {site: summarize(evaluate("test", site))
                                for site in sorted({r.site for r in records if r.split == "test"})}
    model.metrics = {k: results.get(k) for k in ("val", "test", "test_confusion", "test_per_site")}
    model.save(out / "spectral.json")
    (out / "spectral_metrics.json").write_text(json.dumps(results, indent=2))
    return results
