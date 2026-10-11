import numpy as np
from rasterio.transform import from_origin

from bluecarbon.schema import IGNORE_INDEX, KEY_TO_ID
from bluecarbon.surveys import burn_wetland, esri_to_geojson


def _image(h=20, w=20):
    # 10 S2 bands, scaled ints: vegetated land (high NIR) on the left, water on the right
    img = np.full((10, h, w), 500, np.int16)
    img[6, :, :10] = 3000   # B8 NIR high -> vegetation
    img[2, :, :10] = 300    # B4 red low
    img[1, :, 10:] = 800    # B3 green higher than SWIR on the water side
    img[8, :, 10:] = 50     # B11 low -> MNDWI > 0
    img[6, :, 10:] = 100
    return img


def test_esri_rings_to_multipolygon():
    f = {"attributes": {"A": 1}, "geometry": {"rings": [
        [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]],      # clockwise: outer
        [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]],          # counter-clockwise: hole
        [[20, 0], [20, 5], [25, 5], [25, 0], [20, 0]]]}}   # second outer
    g = esri_to_geojson(f)["geometry"]
    assert g["type"] == "MultiPolygon" and len(g["coordinates"]) == 2 and len(g["coordinates"][0]) == 2


def test_burn_wetland_only_on_vegetation_and_authoritative():
    tr = from_origin(0, 20, 1, 1)
    lab = np.full((20, 20), KEY_TO_ID["other_land"], np.uint8)
    lab[:, 10:] = KEY_TO_ID["water"]
    lab[15:, :5] = KEY_TO_ID["saltmarsh"]  # global map says marsh where the survey does not
    poly = {"geometry": {"type": "Polygon", "coordinates": [[[0, 20], [20, 20], [20, 12], [0, 12], [0, 20]]]}}
    out, n = burn_wetland(lab, _image(), tr, "EPSG:4326", [poly], "saltmarsh", authoritative=True)
    assert (out[:8, :10] == KEY_TO_ID["saltmarsh"]).all()          # inside, vegetated
    assert (out[:8, 10:] == KEY_TO_ID["water"]).all()              # inside, water: untouched
    assert (out[15:, :5] == IGNORE_INDEX).all()                    # disagreement -> unlabelled
    assert n == 80


def test_authoritative_ignored_when_survey_empty():
    tr = from_origin(0, 20, 1, 1)
    lab = np.full((20, 20), KEY_TO_ID["saltmarsh"], np.uint8)
    out, n = burn_wetland(lab, _image(), tr, "EPSG:4326", [], "saltmarsh", authoritative=True)
    assert (out == lab).all() and n == 0
