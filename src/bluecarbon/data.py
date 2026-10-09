"""PyTorch dataset over .npz chips."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import N_FEATURES, Normalizer, compute_features, valid_mask
from .schema import IGNORE_INDEX, N_CLASSES
from .tiling import ChipRecord


def load_chip(path: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        return z["image"], z["label"]


def load_chip_anc(path: str) -> np.ndarray | None:
    with np.load(path) as z:
        return z["anc"] if "anc" in z.files else None


def chips_have_ancillary(records: list[ChipRecord]) -> bool:
    return bool(records) and all(load_chip_anc(r.path) is not None for r in records[:50])


def chip_feature_names(records: list[ChipRecord], use_anc: bool) -> list[str]:
    """Feature names the chips support (all chips must share one ancillary layout)."""
    from .features import default_feature_names

    if not use_anc:
        return default_feature_names(None)
    step = max(1, len(records) // 200)  # sample across all sites, not every chip
    sets = {tuple(default_feature_names(load_chip_anc(r.path))) for r in records[::step]}
    return list(min(sets, key=len))


class ChipDataset(Dataset):
    def __init__(self, records: list[ChipRecord], normalizer: Normalizer, augment: bool = False,
                 use_anc: bool = False, names: list[str] | None = None):
        self.records = records
        self.norm = normalizer
        self.augment = augment
        self.use_anc = use_anc
        self.names = names

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        img, lab = load_chip(self.records[i].path)
        anc = load_chip_anc(self.records[i].path) if self.use_anc else None
        x = self.norm(compute_features(img, anc=anc, names=self.names))
        y = lab.astype(np.int64)
        if self.augment:
            # spectral features come first; never jitter the context layers (elevation, tide, depth)
            x, y = _augment(x, y, n_spectral=N_FEATURES)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def _augment(x: np.ndarray, y: np.ndarray, rng=np.random, n_spectral: int | None = None):
    """Dihedral (8 orientations) + mild per-chip spectral jitter.

    Spectral jitter mimics atmosphere / sun-angle differences between sites.
    """
    k = rng.randint(4)
    x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k)
    if rng.rand() < 0.5:
        x, y = x[:, :, ::-1], y[:, ::-1]
    n = x.shape[0] if n_spectral is None else n_spectral  # never jitter elevation / tide
    gain = np.ones((x.shape[0], 1, 1), np.float32)
    bias = np.zeros((x.shape[0], 1, 1), np.float32)
    gain[:n] += rng.normal(0, 0.05, size=(n, 1, 1)).astype(np.float32)
    bias[:n] += rng.normal(0, 0.05, size=(n, 1, 1)).astype(np.float32)
    return x * gain + bias, y


def fit_normalizer(records: list[ChipRecord], max_chips: int = 400, seed: int = 0,
                   use_anc: bool = False, names: list[str] | None = None) -> Normalizer:
    rng = np.random.default_rng(seed)
    pick = rng.permutation(len(records))[:max_chips]
    feats, masks = [], []
    for i in pick:
        img, _ = load_chip(records[i].path)
        feats.append(compute_features(img, anc=load_chip_anc(records[i].path) if use_anc else None, names=names))
        masks.append(valid_mask(img))
    return Normalizer.fit(feats, masks)


def class_frequencies(records: list[ChipRecord]) -> np.ndarray:
    counts = np.zeros(N_CLASSES, np.int64)
    for r in records:
        _, lab = load_chip(r.path)
        counts += np.bincount(lab[lab != IGNORE_INDEX].ravel(), minlength=N_CLASSES)[:N_CLASSES]
    return counts
