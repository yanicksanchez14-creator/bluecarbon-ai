"""Sea-level-rise screen: open low land behind the wetland counts as room to migrate; towns and high ground don't."""

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_origin

from bluecarbon.schema import KEY_TO_ID
from bluecarbon.slr import slr_exposure


def test_room_to_migrate():
    cls = np.full((100, 100), KEY_TO_ID["water"], np.uint8)
    cls[:, 20:40] = KEY_TO_ID["mangrove"]           # 20 columns of mangrove
    cls[:, 40:] = KEY_TO_ID["other_land"]
    elev = np.zeros(cls.shape)
    elev[:, 40:60] = 1.0                              # low land just behind: room
    elev[:, 60:] = 10.0                               # high ground: no room
    t, crs = from_origin(500000, 4000000, 10, 10), CRS.from_epsg(32617)
    r = slr_exposure(cls, elev, None, t, crs)
    assert r["wetland_ha"] == 20.0 and r["room_ha"] == 20.0 and r["rating"] == "low"
    built = np.zeros(cls.shape, np.uint8)
    built[:, 40:58] = 1                               # a town on most of that low land
    r = slr_exposure(cls, elev, built, t, crs)
    assert r["room_ha"] == 2.0 and r["rating"] == "medium"
    built[:, 40:60] = 1
    assert slr_exposure(cls, elev, built, t, crs)["rating"] == "high"


def test_confidence_summary(tmp_path):
    import rasterio

    from bluecarbon.demo import confidence_summary

    cls = np.full((20, 20), KEY_TO_ID["water"], np.uint8)
    cls[:10] = KEY_TO_ID["mangrove"]
    conf = np.full((20, 20), 95, np.uint8)
    conf[:5] = 50                                      # half the mangrove mapped with low confidence
    p = tmp_path / "pred.tif"
    with rasterio.open(p, "w", driver="GTiff", width=20, height=20, count=2, dtype="uint8",
                       transform=from_origin(0, 0, 10, 10), crs=CRS.from_epsg(32617)) as ds:
        ds.write(cls, 1)
        ds.write(conf, 2)
    s = confidence_summary(p)
    assert s["mangrove"] == {"mean_pct": 72.5, "low_share": 0.5}
