import numpy as np

from bluecarbon.features import (
    ANCILLARY_BANDS,
    FEATURE_NAMES_ANC_RADAR,
    FEATURE_NAMES_ANC_RADAR_LBAND,
    LBAND_OFFSET,
    ancillary_features,
    default_feature_names,
)


def test_lband_features_follow_radar():
    n = len(ANCILLARY_BANDS)
    anc = np.zeros((n + 4, 8, 8), np.int16)
    anc[LBAND_OFFSET] = -800    # HH -8 dB: strong trunk scattering
    anc[LBAND_OFFSET + 1] = -1500
    f = ancillary_features(anc)
    assert {"ALOS_HH", "ALOS_HV", "ALOS_HV_HH"} <= set(f)
    assert np.allclose(f["ALOS_HH"], (-8 + 30) / 35)
    assert default_feature_names(anc)[:len(FEATURE_NAMES_ANC_RADAR_LBAND)] == FEATURE_NAMES_ANC_RADAR_LBAND
    assert default_feature_names(anc[: n + 2]) == FEATURE_NAMES_ANC_RADAR   # radar only: older models unchanged
