"""Cut image/label rasters into chips with a leakage-free spatial split.

v1 used overlapping chips (stride 128 on 256) and a random split, so near-identical pixels
ended up in both train and validation. v2 uses non-overlapping chips and assigns whole
spatial blocks (default 5 km) to train / val / test, plus fully held-out test sites.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .features import valid_mask
from .schema import IGNORE_INDEX


@dataclass
class ChipRecord:
    path: str
    site: str
    split: str
    row: int
    col: int
    labeled_frac: float


def block_split(site: str, bx: int, by: int, fractions=(0.7, 0.15, 0.15), seed: int = 42) -> str:
    """Deterministic split for a spatial block: same block -> same split, always."""
    h = hashlib.sha1(f"{seed}:{site}:{bx}:{by}".encode()).hexdigest()
    u = int(h[:8], 16) / 0xFFFFFFFF
    if u < fractions[0]:
        return "train"
    if u < fractions[0] + fractions[1]:
        return "val"
    return "test"


def make_chips(image_path: str | Path, label_path: str | Path, out_dir: str | Path, site: str,
               size: int = 256, stride: int = 256, min_labeled_frac: float = 0.2, block_km: float = 5,
               fractions=(0.7, 0.15, 0.15), seed: int = 42, force_split: str | None = None,
               ancillary_path: str | Path | None = None, tag: str = "",
               only_splits: set[str] | None = None, radar_path: str | Path | None = None) -> list[ChipRecord]:
    """`tag` names chips from an extra image of the same site (e.g. another year); the split is still
    decided by the site's spatial blocks, so a location is in the same split in every year.
    `only_splits`: keep only chips in these splits (extra years go to training only, so validation and
    test stay on the year the reference labels describe)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[ChipRecord] = []
    anc_ds = rasterio.open(ancillary_path) if ancillary_path and Path(ancillary_path).exists() else None
    rad_ds = (rasterio.open(radar_path) if anc_ds is not None and radar_path and Path(radar_path).exists() else None)
    if rad_ds is not None and (rad_ds.width, rad_ds.height) != (anc_ds.width, anc_ds.height):
        rad_ds.close()
        rad_ds = None
    with rasterio.open(image_path) as im, rasterio.open(label_path) as lb:
        if (im.width, im.height) != (lb.width, lb.height) or im.transform != lb.transform:
            raise ValueError(f"{image_path} and {label_path} are not on the same grid")
        res = abs(im.transform.a)
        block_px = max(size, int(round(block_km * 1000 / res)))
        for r in range(0, im.height - size + 1, stride):
            for c in range(0, im.width - size + 1, stride):
                win = Window(c, r, size, size)
                y = lb.read(1, window=win)
                x = im.read(window=win)
                y = np.where(valid_mask(x), y, IGNORE_INDEX).astype(np.uint8)
                frac = float((y != IGNORE_INDEX).mean())
                if frac < min_labeled_frac:
                    continue
                split = force_split or block_split(site, c // block_px, r // block_px, fractions, seed)
                if only_splits is not None and split not in only_splits:
                    continue
                p = out_dir / f"{site}{'_' + tag if tag else ''}_r{r:05d}_c{c:05d}.npz"
                extra = {}
                if anc_ds is not None:
                    a = anc_ds.read(window=win).astype(np.int16)
                    if rad_ds is not None:  # radar appended after the ancillary bands
                        a = np.concatenate([a, rad_ds.read(window=win).astype(np.int16)])
                    extra["anc"] = a
                np.savez_compressed(p, image=x.astype(np.uint16), label=y, **extra)
                records.append(ChipRecord(str(p), site, split, r, c, frac))
    for ds in (anc_ds, rad_ds):
        if ds is not None:
            ds.close()
    return records


def write_index(records: list[ChipRecord], path: str | Path) -> None:
    """Write the chip index with paths relative to it, so a chip folder can be moved (e.g. Colab -> PC)."""
    import os

    path = Path(path)
    rows = []
    for r in records:
        d = asdict(r)
        d["path"] = os.path.relpath(Path(r.path).resolve(), path.parent.resolve())
        rows.append(d)
    path.write_text(json.dumps(rows, indent=1))


def read_index(path: str | Path) -> list[ChipRecord]:
    path = Path(path)
    recs = [ChipRecord(**d) for d in json.loads(path.read_text())]
    for r in recs:
        if not Path(r.path).is_absolute():
            r.path = str(path.parent / r.path)
    return recs
