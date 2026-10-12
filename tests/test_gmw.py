import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_gmw_agreement():
    from gmw_check import agreement

    gmw = {1996: 1000.0, 2010: 950.0, 2020: 900.0}
    hist = [{"year": 1995, "ha": 1100}, {"year": 2010, "ha": 1000}, {"year": 2020, "ha": 950}]
    a = agreement(gmw, hist)
    assert a["years_compared"] == 3 and a["same_direction"] and a["gmw_change_pct"] == -10.0
    assert agreement({1996: 1.0}, hist) is None
