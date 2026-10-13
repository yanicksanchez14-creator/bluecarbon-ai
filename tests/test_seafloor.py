import numpy as np

from bluecarbon.features import compute_features, default_feature_names, seafloor_hidden, training_label
from bluecarbon.schema import IGNORE_INDEX, KEY_TO_ID

SG, W = KEY_TO_ID["seagrass"], KEY_TO_ID["water"]


def _anc(b2, b3, b4, b8=100):
    a = np.zeros((12, 40, 40), np.int16)  # elevation, tidal, abs_lat, B2/B3/B4/B8 clear, depth, S1 x2, ALOS x2
    a[3], a[4], a[5], a[6] = b2, b3, b4, b8
    return a


def test_visible_seagrass_kept_hidden_skipped():
    b2, b3, b4 = (np.full((40, 40), v) for v in (150, 120, 40))                         # deep, clear water
    b2[:, :20], b3[:, :20], b4[:, :20] = 320, 380, 120                                 # shallow seagrass: bottom shows
    b4[30:, 20:], b3[30:, 20:] = 700, 750                                              # brown, sediment-laden water
    anc = _anc(b2, b3, b4)
    lab = np.full((40, 40), SG, np.uint8)
    lab[:5] = W
    hid = seafloor_hidden(anc)
    assert not hid[:, :20].any() and hid[10:25, 25:].all()
    assert not hid[32:, 25:].any()                        # brown water no longer counts as hidden
    t = training_label(lab, anc)
    assert (t[5:, :20] == SG).all()                       # visible seagrass still trains
    n_sg = int((lab == SG).sum())
    assert (t == IGNORE_INDEX).sum() <= 0.25 * n_sg + 1   # at most a quarter skipped
    assert not (t == W)[5:].any()                         # never turned into water
    assert (t[:5] == W).all()


def test_psdb_feature_present_for_new_models():
    anc = _anc(np.full((40, 40), 300), np.full((40, 40), 400), np.full((40, 40), 100))
    names = default_feature_names(anc)
    assert "PSDB_CLEAR" in names
    f = compute_features(np.full((10, 40, 40), 500, np.uint16), anc=anc, names=names)
    assert np.isfinite(f).all()


def test_height_biomass_fit_recovers_power_law():
    from bluecarbon.gee import fit_height_biomass

    rng = np.random.default_rng(0)
    h = rng.uniform(3, 25, 300)
    r = fit_height_biomass(list(zip(h, 4 * h ** 1.3 * np.exp(rng.normal(0, 0.2, 300)), strict=True)))
    assert abs(r["hb_b"] - 1.3) < 0.1 and r["hb_r2_cv"] > 0.8
    assert fit_height_biomass([(5, 10)] * 10) == {}


def test_missing_lband_drops_sites_not_the_input(tmp_path):
    from bluecarbon.data import chip_feature_names, drop_unsupported
    from bluecarbon.tiling import ChipRecord

    recs = []
    for i in range(10):
        bands = 12 if i else 10          # site s0 has no L-band (failed download)
        p = tmp_path / f"c{i}.npz"
        np.savez(p, image=np.zeros((10, 4, 4), np.uint16), label=np.zeros((4, 4), np.uint8),
                 anc=np.zeros((bands, 4, 4), np.int16))
        recs.append(ChipRecord(path=str(p), site=f"s{i}", split="test" if i == 9 else "train", row=0, col=0,
                                labeled_frac=1.0))
    names = chip_feature_names(recs, True, log=lambda *_: None)
    assert "ALOS_HH" in names
    assert {r.site for r in drop_unsupported(recs, names)} == {f"s{i}" for i in range(1, 10)}
