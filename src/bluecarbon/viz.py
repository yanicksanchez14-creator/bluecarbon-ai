"""Rendering helpers for maps, the app and README figures."""

from __future__ import annotations

import numpy as np

from .features import REFLECTANCE_SCALE, S2_BANDS
from .schema import CLASSES, IGNORE_INDEX, palette_rgb


def true_color(bands: np.ndarray, band_names: list[str] | None = None, p=(2, 98), gamma: float = 0.9,
               rgb_bands: tuple[str, str, str] = ("B4", "B3", "B2")) -> np.ndarray:
    """(C,H,W) S2 bands -> (H,W,4) uint8 RGBA with a percentile stretch.

    rgb_bands=("B8", "B4", "B3") gives the classic false-color infrared view, where healthy
    vegetation (marsh, mangrove) shows up bright red.
    """
    names = band_names or S2_BANDS
    idx = [names.index(b) for b in rgb_bands]
    rgb = np.nan_to_num(bands[idx].astype(np.float32))
    if rgb.max() > 2:
        rgb = rgb / REFLECTANCE_SCALE
    valid = (rgb > 0).any(0)
    out = np.zeros((*rgb.shape[1:], 4), np.uint8)
    if not valid.any():
        return out
    lo, hi = np.percentile(rgb[:, valid], p)
    s = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) ** gamma
    out[..., :3] = (s.transpose(1, 2, 0) * 255).astype(np.uint8)
    out[..., 3] = np.where(valid, 255, 0)
    return out


def class_rgba(class_map: np.ndarray, alpha: int = 255, only: list[int] | None = None) -> np.ndarray:
    pal = np.array(palette_rgb() + [(0, 0, 0)] * (256 - len(CLASSES)), np.uint8)
    out = np.zeros((*class_map.shape, 4), np.uint8)
    out[..., :3] = pal[class_map]
    show = class_map != IGNORE_INDEX
    if only is not None:
        show &= np.isin(class_map, only)
    out[..., 3] = np.where(show, alpha, 0)
    return out


def change_rgba(t0: np.ndarray, t1: np.ndarray, blue_ids: list[int]) -> np.ndarray:
    """Green = blue carbon gained, red = lost, transparent = unchanged."""
    b0, b1 = np.isin(t0, blue_ids), np.isin(t1, blue_ids)
    valid = (t0 != IGNORE_INDEX) & (t1 != IGNORE_INDEX)
    out = np.zeros((*t0.shape, 4), np.uint8)
    out[valid & b1 & ~b0] = (34, 197, 94, 230)
    out[valid & b0 & ~b1] = (239, 68, 68, 230)
    return out


def confidence_rgba(conf_pct: np.ndarray, class_map: np.ndarray, only: list[int] | None = None) -> np.ndarray:
    """Where to doubt the map: low model confidence shown warm and opaque, high confidence clear.
    conf_pct is the model's probability for the chosen class (0-100); 255 = no data."""
    c = conf_pct.astype(float)
    valid = (conf_pct != 255) & (class_map != 255)
    if only is not None:
        valid &= np.isin(class_map, only)
    doubt = np.clip((90 - c) / 50, 0, 1)            # 90%+ -> clear, 40% or less -> full colour
    out = np.zeros((*conf_pct.shape, 4), np.uint8)
    out[..., 0], out[..., 1], out[..., 2] = 242, (166 - 90 * doubt).astype(np.uint8), 90
    out[..., 3] = np.where(valid, (40 + 200 * doubt) * (doubt > 0), 0).astype(np.uint8)
    return out


def save_png(rgba: np.ndarray, path) -> None:
    import matplotlib.pyplot as plt

    plt.imsave(path, rgba)


def plot_confusion(cm: np.ndarray, labels: list[str], path, title: str = "Confusion matrix (row-normalized)"):
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i].sum() > 0:
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
