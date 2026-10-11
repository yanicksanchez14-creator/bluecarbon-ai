"""Seagrass specialist: learns dark, patchy seafloor the main model calls water; switches itself on only if it helps."""

import numpy as np
import pytest

from bluecarbon.features import ANCILLARY_BANDS
from bluecarbon.schema import KEY_TO_ID
from bluecarbon.tiling import ChipRecord

W, SG, LAND = KEY_TO_ID["water"], KEY_TO_ID["seagrass"], KEY_TO_ID["other_land"]


def _chip(path, rng):
    lab = np.full((64, 64), W, np.uint8)
    lab[:, :8] = LAND
    lab[20:44, 16:48] = SG
    img = np.full((10, 64, 64), 300, np.uint16)
    img[:, :, :8] = 2500                                   # land: bright, MNDWI < 0 via SWIR
    img[8:, :, :8] = 4000
    anc = np.zeros((len(ANCILLARY_BANDS), 64, 64), np.int16)
    anc[3:7] = 900                                         # sandy bottom, clear water
    anc[3:7, 20:44, 16:48] = 350                           # meadow: darker bottom
    anc[3:7] += rng.integers(-40, 40, anc[3:7].shape).astype(np.int16)
    np.savez_compressed(path, image=img, label=lab, anc=anc)


def test_specialist_learns_and_switches_on(tmp_path):
    pytest.importorskip("lightgbm")
    from bluecarbon.seagrass import SeagrassSpecialist, fuse, train_specialist

    rng = np.random.default_rng(0)
    recs = []
    for i, split in enumerate(["train"] * 6 + ["val"] * 2 + ["test"] * 2):
        p = tmp_path / f"c{i}.npz"
        _chip(p, rng)
        recs.append(ChipRecord(str(p), f"s{i}", split, 0, 0, 1.0))

    def main_says_water(img, anc):                         # the main model misses every meadow
        m = np.full(img.shape[1:], W, np.uint8)
        m[:, :8] = LAND
        return m

    sp = train_specialist(recs, main_says_water)
    assert sp.enabled and sp.metrics["test_seagrass_iou_fused"] > 0.9 and not sp.metrics["test_seagrass_iou_main"]
    sp.save(tmp_path / "m.seagrass.json")
    sp2 = SeagrassSpecialist.load(tmp_path / "m.seagrass.json")
    with np.load(recs[-1].path) as z:
        out = sp2.apply(main_says_water(z["image"], z["anc"]), z["image"], z["anc"])
    assert (out[:, :8] == LAND).all() and (out[30, 30] == SG)
    # never touches land
    assert (fuse(np.full((2, 2), LAND, np.uint8), np.ones((2, 2)), 0.5) == LAND).all()
