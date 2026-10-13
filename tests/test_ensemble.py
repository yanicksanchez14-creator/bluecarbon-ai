"""U-Net + LightGBM blend: tuning picks the right model per class, and the saved blend loads and predicts."""

import json

import numpy as np
import rasterio

from bluecarbon.config import load_config
from bluecarbon.ensemble import blue_iou, build, score, tune
from bluecarbon.predictors import blend, is_ensemble, is_spectral, load_predictor, model_card
from bluecarbon.schema import CLASSES, N_CLASSES
from bluecarbon.tiling import make_chips, read_index, write_index

from .conftest import make_scene


def _onehot(c, n):
    p = np.full((n, N_CLASSES), 0.02, np.float32)
    p[np.arange(n), c] = 0.9
    return p


def test_tune_takes_each_class_from_the_model_that_gets_it_right():
    keys = [c.key for c in CLASSES]
    a, b = keys.index("mangrove"), keys.index("seagrass")
    y = np.array([a] * 50 + [b] * 50)
    unet = _onehot(np.where(y == a, a, keys.index("water")), 100)  # U-Net right on mangrove only
    lgbm = _onehot(np.where(y == b, b, keys.index("saltmarsh")), 100)  # LightGBM right on seagrass only
    w, s = tune(unet, lgbm, y)
    assert s > max(score(unet, lgbm, y, np.ones(N_CLASSES)), score(unet, lgbm, y, np.zeros(N_CLASSES)))
    assert s > 0.95


def test_blend_and_iou_basics():
    pu, ps = np.ones((N_CLASSES, 2, 2)), np.zeros((N_CLASSES, 2, 2))
    assert np.allclose(blend(pu, ps, np.full(N_CLASSES, 0.25)), 0.25)
    cm = np.zeros((N_CLASSES, N_CLASSES), int)
    assert blue_iou(cm) == 0.0
    assert is_ensemble("m/x.ensemble.json") and not is_spectral("m/x.ensemble.json")


def test_build_and_load(tmp_path):
    from bluecarbon.spectral import train_spectral
    from bluecarbon.train import train

    cfg = load_config(None, {
        "model": {"encoder": "resnet18", "encoder_weights": None},
        "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "lr": 3e-3, "device": "cpu", "amp": False},
    })
    recs = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        d = tmp_path / f"site{i}"
        d.mkdir()
        make_scene(d / "image.tif", d / "label.tif", size=256, seed=i)
        recs += make_chips(d / "image.tif", d / "label.tif", tmp_path / "chips", f"site{i}", size=128, stride=128,
                           force_split=split)
    write_index(recs, tmp_path / "chips" / "index.json")
    recs = read_index(tmp_path / "chips" / "index.json")
    out = tmp_path / "model"
    train(cfg, recs, out, log=lambda *_: None)
    train_spectral(recs, out, per_class_per_chip=100, n_estimators=20, log=lambda *_: None)
    m = build(out, recs, "cpu", tta=False, log=lambda *_: None)
    assert set(m["val_blue_iou"]) == {"unet", "spectral", "ensemble"}
    assert m["val_blue_iou"]["ensemble"] >= max(m["val_blue_iou"]["unet"], m["val_blue_iou"]["spectral"]) - 1e-9
    assert "site3" in m["test_per_site"] and m["test"] is not None
    path = out / "ensemble.ensemble.json"
    assert json.loads(path.read_text())["unet"] == "model.pt"
    pr = load_predictor(path, "cpu")
    with rasterio.open(tmp_path / "site3" / "image.tif") as s:
        bands = s.read()
    cls, conf = pr.predict(bands, tile=128, overlap=32, tta=False)
    assert cls.shape == bands.shape[1:] and conf.max() <= 1.0 + 1e-6
    card = model_card(path)
    assert card["kind"] == "ensemble" and "LightGBM" in card["arch"] and "test_per_site" in card["metrics"]
