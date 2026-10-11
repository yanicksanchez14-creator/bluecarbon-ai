"""Sentinel-1 radar: appended after the ancillary bands; old models (no radar features) are unaffected."""

import numpy as np

from bluecarbon.features import (
    ANCILLARY_BANDS,
    FEATURE_NAMES_ANC_V1,
    RADAR_FEATURES,
    compute_features,
    default_feature_names,
)


def test_radar_features():
    bands = np.full((10, 4, 4), 1000, np.uint16)
    anc = np.zeros((len(ANCILLARY_BANDS), 4, 4), np.int16)
    rad = np.stack([np.full((4, 4), -800), np.full((4, 4), -1600)]).astype(np.int16)   # VV -8 dB, VH -16 dB
    both = np.concatenate([anc, rad])
    assert default_feature_names(anc) == FEATURE_NAMES_ANC_V1
    assert default_feature_names(both)[-3:] == RADAR_FEATURES
    f = compute_features(bands, anc=both, names=RADAR_FEATURES)
    assert np.allclose(f[0], 22 / 35) and np.allclose(f[1], 19 / 35) and np.allclose(f[2], -8 / 20)
    # a model trained without radar asks for its own names and still works on a stack that has radar
    assert compute_features(bands, anc=both, names=FEATURE_NAMES_ANC_V1).shape[0] == len(FEATURE_NAMES_ANC_V1)
