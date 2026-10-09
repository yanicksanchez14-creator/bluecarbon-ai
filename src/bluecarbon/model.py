"""Model construction and self-describing checkpoints.

A checkpoint bundles weights + architecture + normalization stats + class schema + metrics,
so inference never depends on a separate stats file (a v1 failure mode).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .features import FEATURE_NAMES, Normalizer
from .schema import CLASS_KEYS, N_CLASSES

CHECKPOINT_VERSION = 2


def build_model(arch: str = "Unet", encoder: str = "resnet34", encoder_weights: str | None = "imagenet",
                in_channels: int = len(FEATURE_NAMES), n_classes: int = N_CLASSES) -> nn.Module:
    import segmentation_models_pytorch as smp

    cls = getattr(smp, arch)
    try:
        return cls(encoder_name=encoder, encoder_weights=encoder_weights, in_channels=in_channels,
                   classes=n_classes)
    except Exception as e:  # offline / blocked download -> random init rather than crash
        if encoder_weights is None:
            raise
        print(f"[bluecarbon] could not load {encoder_weights} weights ({e.__class__.__name__}); "
              "training encoder from scratch")
        return cls(encoder_name=encoder, encoder_weights=None, in_channels=in_channels, classes=n_classes)


def save_checkpoint(path: str | Path, model: nn.Module, arch: str, encoder: str, normalizer: Normalizer,
                    metrics: dict | None = None, extra: dict | None = None,
                    features: list[str] | None = None, class_bias: list[float] | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "version": CHECKPOINT_VERSION,
        "state_dict": model.state_dict(),
        "arch": arch,
        "encoder": encoder,
        "features": features or FEATURE_NAMES,
        "classes": CLASS_KEYS,
        "normalizer": normalizer.to_dict(),
        "metrics": metrics or {},
        "extra": extra or {},
        "class_bias": class_bias,
    }, path)


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[nn.Module, Normalizer, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"{path} is not a v2 bluecarbon checkpoint")
    from .features import KNOWN_FEATURES
    from .schema import compatible

    if not set(ck["features"]) <= KNOWN_FEATURES or not compatible(ck["classes"]):
        raise ValueError("Checkpoint feature/class schema does not match this version of bluecarbon")
    model = build_model(ck["arch"], ck["encoder"], None, len(ck["features"]), len(ck["classes"]))
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, Normalizer.from_dict(ck["normalizer"]), ck


def resolve_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
