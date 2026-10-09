"""Training loop: CE + Dice, class weighting, AMP, cosine LR, early stopping on val mIoU."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Config
from .data import (
    ChipDataset,
    chip_feature_names,
    chips_have_ancillary,
    class_frequencies,
    fit_normalizer,
    load_chip,
)
from .metrics import confusion, summarize
from .model import build_model, resolve_device, save_checkpoint
from .schema import IGNORE_INDEX, N_CLASSES
from .tiling import ChipRecord


class DiceCELoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None, dice_weight: float = 0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=IGNORE_INDEX)
        self.dw = dice_weight

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = self.ce(logits, y)
        if self.dw <= 0:
            return loss
        valid = (y != IGNORE_INDEX).unsqueeze(1)
        prob = logits.float().softmax(1) * valid
        onehot = F.one_hot(y.clamp(max=N_CLASSES - 1), N_CLASSES).permute(0, 3, 1, 2) * valid
        inter = (prob * onehot).sum((0, 2, 3))
        den = prob.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
        present = onehot.sum((0, 2, 3)) > 0
        dice = 1 - (2 * inter + 1) / (den + 1)
        return loss + self.dw * (dice[present].mean() if present.any() else 0.0)


def class_weights(freq: np.ndarray, mode: str) -> np.ndarray | None:
    if mode == "none":
        return None
    f = np.maximum(freq, 1).astype(np.float64)
    w = 1 / f if mode == "inverse" else 1 / np.sqrt(f)
    w = w / w[freq > 0].mean()
    w[freq == 0] = 0.0
    return np.clip(w, 0, 10).astype(np.float32)


@torch.no_grad()
def _bias_tensor(bias, device):
    return None if bias is None else torch.as_tensor(np.asarray(bias, np.float32), device=device).view(1, -1, 1, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device, bias=None) -> np.ndarray:
    """Confusion matrix; `bias` (one value per class) is added to the log-probabilities before argmax."""
    model.eval()
    b = _bias_tensor(bias, device)
    cm = np.zeros((N_CLASSES, N_CLASSES), np.int64)
    for x, y in loader:
        logp = model(x.to(device)).float().log_softmax(1)
        pred = (logp if b is None else logp + b).argmax(1).cpu().numpy()
        cm += confusion(y.numpy(), pred)
    return cm


SEAGRASS_BIAS_GRID = [round(v, 2) for v in np.arange(-1.0, 3.01, 0.25)]


@torch.no_grad()
def tune_seagrass_bias(model: nn.Module, loader: DataLoader, device, log=print) -> tuple[float, list]:
    """Pick how readily the model calls seagrass, on the validation chips (never the test set).

    The network's raw argmax was cautious (it missed about half of seagrass in murky water). A positive
    bias on the seagrass log-probability trades a few false alarms for more detection; the value with
    the best validation mIoU (all classes, so false alarms on water count against it) is kept.
    """
    from .schema import KEY_TO_ID

    sg = KEY_TO_ID["seagrass"]
    model.eval()
    cms = {b: np.zeros((N_CLASSES, N_CLASSES), np.int64) for b in SEAGRASS_BIAS_GRID}
    for x, y in loader:
        logp = model(x.to(device)).float().log_softmax(1)
        yy = y.numpy()
        for b in SEAGRASS_BIAS_GRID:
            adj = logp.clone()
            adj[:, sg] += b
            cms[b] += confusion(yy, adj.argmax(1).cpu().numpy())
    table = []
    for b, cm in cms.items():
        sm = summarize(cm)
        table.append({"bias": b, "mIoU": sm["mIoU"], "seagrass_iou": sm["iou"].get("seagrass"),
                      "water_iou": sm["iou"].get("water")})
    best = max(table, key=lambda r: r["mIoU"])
    for r in table:
        log(f"  seagrass bias {r['bias']:+.2f}: val mIoU {r['mIoU']:.4f}  seagrass {r['seagrass_iou']}  "
            f"water {r['water_iou']}{'  <- best' if r is best else ''}")
    return float(best["bias"]), table


def rare_class_sampler(records: list[ChipRecord], factor: float):
    """Draw chips containing seagrass / salt marsh / mangrove more often (at most `factor` x)."""
    from torch.utils.data import WeightedRandomSampler

    from .schema import BLUE_CARBON_KEYS, KEY_TO_ID

    rare = [KEY_TO_ID[k] for k in BLUE_CARBON_KEYS]
    weights = []
    for r in records:
        _, lab = load_chip(r.path)
        frac = max(float((lab == c).mean()) for c in rare)
        weights.append(1.0 + (factor - 1.0) * min(1.0, frac * 20))  # 5% coverage of a rare class -> full boost
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


def train(cfg: Config, records: list[ChipRecord], out_dir: str | Path, log=print) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr = [r for r in records if r.split == "train"]
    va = [r for r in records if r.split == "val"]
    te = [r for r in records if r.split == "test"]
    if not tr or not va:
        raise ValueError(f"Need train and val chips (got train={len(tr)}, val={len(va)})")
    log(f"chips: train={len(tr)} val={len(va)} test={len(te)}")

    torch.manual_seed(cfg.chips.seed)
    np.random.seed(cfg.chips.seed)
    device = resolve_device(cfg.train.device)
    use_anc = cfg.model.use_ancillary and chips_have_ancillary(records)
    features = chip_feature_names(records, use_anc)
    log(f"inputs: {len(features)} features: {', '.join(features)}")
    norm = fit_normalizer(tr, use_anc=use_anc, names=features)
    freq = class_frequencies(tr)
    w = class_weights(freq, cfg.train.class_weighting)
    log(f"train class pixels: {freq.tolist()}  weights: {None if w is None else np.round(w, 2).tolist()}")

    t = cfg.train
    kw = dict(batch_size=t.batch_size, num_workers=t.num_workers, pin_memory=device.type == "cuda")
    sampler = rare_class_sampler(tr, t.rare_oversample) if t.rare_oversample > 1 else None
    dl_tr = DataLoader(ChipDataset(tr, norm, augment=True, use_anc=use_anc, names=features), shuffle=sampler is None,
                       sampler=sampler, drop_last=len(tr) > t.batch_size, **kw)
    dl_va = DataLoader(ChipDataset(va, norm, use_anc=use_anc, names=features), shuffle=False, **kw)

    m = cfg.model
    model = build_model(m.arch, m.encoder, m.encoder_weights, in_channels=len(features)).to(device)
    lossf = DiceCELoss(None if w is None else torch.tensor(w, device=device), t.dice_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=t.lr, weight_decay=t.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=t.lr, total_steps=max(1, t.epochs * len(dl_tr)),
                                                pct_start=0.1)
    use_amp = t.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_epoch, history = -1.0, -1, []
    ckpt = out / "model.pt"
    for epoch in range(1, t.epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = lossf(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1
        val = summarize(evaluate(model, dl_va, device))
        history.append({"epoch": epoch, "loss": tot / max(nb, 1), **{k: val[k] for k in ("mIoU", "macro_f1",
                                                                                           "overall_accuracy")}})
        log(f"epoch {epoch:3d} loss {tot / max(nb, 1):.4f} val mIoU {val['mIoU']:.4f} "
            f"F1 {val['macro_f1']:.4f} OA {val['overall_accuracy']:.4f} ({time.time() - t0:.0f}s)")
        if val["mIoU"] > best:
            best, best_epoch = val["mIoU"], epoch
            save_checkpoint(ckpt, model, m.arch, m.encoder, norm, {"val": val, "epoch": epoch}, features=features)
        elif epoch - best_epoch >= t.patience:
            log(f"early stop: no val improvement for {t.patience} epochs")
            break

    from .model import load_checkpoint

    model, norm, ck = load_checkpoint(ckpt, device)
    sg_bias, bias_table = tune_seagrass_bias(model, dl_va, device, log)
    from .schema import KEY_TO_ID

    bias = [0.0] * N_CLASSES
    bias[KEY_TO_ID["seagrass"]] = sg_bias
    val_tuned = summarize(evaluate(model, dl_va, device, bias))
    log(f"seagrass bias {sg_bias:+.2f}: val mIoU {ck['metrics']['val']['mIoU']:.4f} -> {val_tuned['mIoU']:.4f}")
    results = {"best_epoch": best_epoch, "val": val_tuned, "val_untuned": ck["metrics"]["val"],
               "class_bias": bias, "seagrass_bias_table": bias_table, "history": history}
    if te:
        cm_te = evaluate(model, DataLoader(ChipDataset(te, norm, use_anc=use_anc, names=features), shuffle=False, **kw),
                         device, bias)
        results["test"] = summarize(cm_te)
        results["test_confusion"] = cm_te.tolist()
        log(f"TEST mIoU {results['test']['mIoU']:.4f}  macro-F1 {results['test']['macro_f1']:.4f}  "
            f"OA {results['test']['overall_accuracy']:.4f}")
    per_site = {}
    for site in sorted({r.site for r in te}):
        rs = [r for r in te if r.site == site]
        per_site[site] = summarize(evaluate(model, DataLoader(ChipDataset(rs, norm, use_anc=use_anc, names=features),
                                                              shuffle=False, **kw), device, bias))
    results["test_per_site"] = per_site
    save_checkpoint(ckpt, model, m.arch, m.encoder, norm, features=features, class_bias=bias,
                    metrics={**ck["metrics"], "val": val_tuned, "test": results.get("test"),
                             "test_confusion": results.get("test_confusion")})
    (out / "metrics.json").write_text(json.dumps(results, indent=2))
    return results
