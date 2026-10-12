import numpy as np

from bluecarbon.metrics import error_adjusted_area


def test_seagrass_correction_capped_by_measured_accuracy():
    cm = np.eye(7) * 100
    cm[3, 0] = 80  # reference seagrass often mapped as water (pooled test error)
    r = error_adjusted_area(cm, np.array([1e6, 0, 0, 1000, 0, 0, 0]), 0.01)
    s = r["seagrass"]
    # a huge water body must not become ~4,400 ha of seagrass when 10 ha were mapped
    assert s["capped"] and s["adjusted_ha"] == 18.0   # 10 ha x precision 1.0 / recall 100/180
    assert "capped" not in r["water"]
