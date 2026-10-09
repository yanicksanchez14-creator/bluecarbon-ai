"""Seagrass survey polygons are burned over water only, and habitat fields are filtered."""

import numpy as np
from rasterio.transform import from_origin

from bluecarbon.schema import IGNORE_INDEX, KEY_TO_ID
from bluecarbon.surveys import burn_seagrass, seagrass_features


def _square(lon0, lat0, lon1, lat1, **props):
    return {"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": [[
        [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0]]]}}


def test_burn_only_over_water():
    h = w = 20
    tr = from_origin(0.0, 0.002, 0.0001, 0.0001)  # EPSG:4326, 0.0001 deg pixels
    label = np.full((h, w), KEY_TO_ID["water"], np.uint8)
    label[:, :5] = KEY_TO_ID["mangrove"]
    label[:, 15:] = IGNORE_INDEX
    image = np.zeros((10, h, w), np.uint16)
    image[1] = 800   # B3 green
    image[8] = 200   # B11 low -> MNDWI > 0 (looks like water)
    out, n = burn_seagrass(label, image, tr, "EPSG:4326", [_square(0.0, 0.0, 0.002, 0.002)])
    assert (out[:, :5] == KEY_TO_ID["mangrove"]).all()          # never over land / mangrove
    assert (out[:, 5:] == KEY_TO_ID["seagrass"]).all()          # water and unlabelled water
    assert n == h * (w - 5)


def test_seagrass_field_filter():
    feats = [_square(0, 0, 1, 1, HABITAT="Seagrass"), _square(0, 0, 1, 1, HABITAT="Sand"),
             _square(0, 0, 1, 1, HABITAT="No seagrass")]
    keep, how = seagrass_features(feats)
    assert len(keep) == 1 and "HABITAT" in how
    keep, _ = seagrass_features([_square(0, 0, 1, 1, SEAGRASS="Continuous")])
    assert len(keep) == 1


def test_negatives_near_meadows():
    h = w = 30
    tr = from_origin(0.0, 0.003, 0.0001, 0.0001)
    label = np.full((h, w), IGNORE_INDEX, np.uint8)       # unmapped water everywhere
    image = np.zeros((10, h, w), np.uint16)
    image[1], image[8] = 800, 200                         # looks like water
    meadow = _square(0.0, 0.002, 0.001, 0.003)            # top-left 10 x 10 px
    out, _ = burn_seagrass(label, image, tr, "EPSG:4326", [meadow], negatives_px=5)
    assert (out[:10, :10] == KEY_TO_ID["seagrass"]).all()
    assert out[12, 5] == KEY_TO_ID["water"]               # 3 px from the meadow -> open water
    assert out[25, 25] == IGNORE_INDEX                     # far away -> still unknown


def test_filter_features():
    from bluecarbon.surveys import filter_features

    feats = [_square(0, 0, 1, 1, BROADSCALE="SEAGRASS", LOCALSCALE="SEAGRASS perennial, dense"),
             _square(0, 0, 1, 1, BROADSCALE="SEAGRASS", LOCALSCALE="SEAGRASS perennial, sparse"),
             _square(0, 0, 1, 1, BROADSCALE="SILT", LOCALSCALE="SILT")]
    out = filter_features(feats, keep={"BROADSCALE": ["seagrass"]}, drop={"LOCALSCALE": ["sparse"]})
    assert len(out) == 1 and "dense" in out[0]["properties"]["LOCALSCALE"]
