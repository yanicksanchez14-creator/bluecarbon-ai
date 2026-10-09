"""End-to-end: chips -> train -> checkpoint -> predict -> report -> demo bundle, on synthetic data."""

import json

import numpy as np
import rasterio

from bluecarbon.config import load_config
from bluecarbon.demo import export_scene
from bluecarbon.model import load_checkpoint
from bluecarbon.predict import predict_raster
from bluecarbon.report import scene_report
from bluecarbon.tiling import make_chips, read_index, write_index
from bluecarbon.train import train

from .conftest import make_scene


def test_end_to_end(tmp_path):
    cfg = load_config(None, {
        "model": {"encoder": "resnet18", "encoder_weights": None},
        "train": {"epochs": 3, "batch_size": 4, "num_workers": 0, "lr": 3e-3, "device": "cpu", "amp": False},
        "carbon": {"monte_carlo": 200},
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

    res = train(cfg, recs, tmp_path / "model", log=lambda *_: None)
    assert (tmp_path / "model" / "model.pt").exists()
    assert "test" in res and res["test"]["mIoU"] is not None
    assert json.loads((tmp_path / "model" / "metrics.json").read_text())["best_epoch"] >= 1

    model, norm, ck = load_checkpoint(tmp_path / "model" / "model.pt")
    assert ck["metrics"]["test_confusion"] is not None

    scene = tmp_path / "scene"
    scene.mkdir()
    make_scene(scene / "image.tif", scene / "label.tif", size=200, seed=9)  # non-multiple of tile size
    predict_raster(model, norm, scene / "image.tif", scene / "pred.tif", tile=128, overlap=32, tta=True)
    with rasterio.open(scene / "pred.tif") as ds:
        cls, conf = ds.read(1), ds.read(2)
        assert cls.shape == (200, 200) and ds.crs.to_epsg() == 32611
    assert cls.max() < 6 and conf.max() <= 100

    rep = scene_report(scene / "pred.tif", cfg.carbon, ck["metrics"]["test_confusion"])
    assert np.isclose(sum(rep["areas_ha"].values()), 200 * 200 * 0.01)
    assert "error_adjusted_areas_ha" in rep

    out = export_scene(scene, tmp_path / "demo" / "syn", "Synthetic", cfg, tmp_path / "model" / "model.pt")
    meta = json.loads((out / "meta.json").read_text())
    assert meta["kind"] == "single" and (out / "rgb.png").exists() and (out / "classes.png").exists()
    (s, w), (n, e) = meta["bounds"]
    assert s < n and w < e


def test_spectral_model(tmp_path):
    from bluecarbon.predictors import load_predictor, model_card
    from bluecarbon.spectral import train_spectral

    recs = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        d = tmp_path / f"site{i}"
        d.mkdir()
        make_scene(d / "image.tif", d / "label.tif", size=256, seed=i)
        recs += make_chips(d / "image.tif", d / "label.tif", tmp_path / "chips", f"site{i}", size=128, stride=128,
                           force_split=split)
    res = train_spectral(recs, tmp_path / "m", per_class_per_chip=200, n_estimators=40, log=lambda *_: None)
    assert res["test"]["mIoU"] > 0.8  # synthetic classes are spectrally separable
    pr = load_predictor(tmp_path / "m" / "spectral.json")
    with rasterio.open(tmp_path / "site3" / "image.tif") as s:
        cls, conf = pr.predict(s.read())
    assert cls.shape == (256, 256) and conf.max() <= 1.0
    assert model_card(tmp_path / "m" / "spectral.json")["kind"] == "spectral-lgbm"


def test_ancillary_inputs(tmp_path):
    from bluecarbon.features import FEATURE_NAMES_ANC
    from bluecarbon.predictors import load_predictor, read_ancillary
    from bluecarbon.spectral import train_spectral

    from .conftest import write_ancillary

    cfg = load_config(None, {
        "model": {"encoder": "resnet18", "encoder_weights": None},
        "train": {"epochs": 2, "batch_size": 4, "num_workers": 0, "lr": 3e-3, "device": "cpu", "amp": False},
    })
    recs = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        d = tmp_path / f"s{i}"
        d.mkdir()
        _, lab = make_scene(d / "image.tif", d / "label.tif", size=256, seed=i)
        anc = write_ancillary(d / "image.tif", lab)
        recs += make_chips(d / "image.tif", d / "label.tif", tmp_path / "chips", f"s{i}", size=128, stride=128,
                           force_split=split, ancillary_path=anc)
    with np.load(recs[0].path) as z:
        assert "anc" in z.files and z["anc"].shape == (8, 128, 128)

    train(cfg, recs, tmp_path / "unet", log=lambda *_: None)
    pr = load_predictor(tmp_path / "unet" / "model.pt")
    assert pr.needs_ancillary
    _, _, ck = load_checkpoint(tmp_path / "unet" / "model.pt")
    assert ck["features"] == FEATURE_NAMES_ANC
    assert "ABSLAT" not in FEATURE_NAMES_ANC  # latitude is a rule (priors.py), never a model input

    train_spectral(recs, tmp_path / "spec", per_class_per_chip=100, n_estimators=20, log=lambda *_: None)
    sp = load_predictor(tmp_path / "spec" / "spectral.json")
    assert sp.needs_ancillary

    img = tmp_path / "s3" / "image.tif"
    with rasterio.open(img) as s:
        bands = s.read()
    anc = read_ancillary(img, bands.shape[1:])
    for p in (pr, sp):
        cls, _ = p.predict(bands, tile=128, overlap=32, anc=anc)
        assert cls.shape == (256, 256)
    try:
        pr.predict(bands)
        raise AssertionError("should require ancillary")
    except ValueError:
        pass


def test_spectral_strips_match_full(tmp_path):
    """Strip-wise prediction (used for large scenes) must equal whole-scene prediction."""
    from bluecarbon.spectral import SpectralModel, sample_pixels

    bands, lab = make_scene(tmp_path / "i.tif", tmp_path / "l.tif", size=300, seed=3)
    X, y = sample_pixels(bands, lab, 300, np.random.default_rng(0))
    m = SpectralModel().fit(X, y, n_estimators=10)
    whole = m.predict_proba(bands)
    strips = m.predict_proba(bands, strip=64, max_block_px=0)
    assert np.allclose(whole, strips)


def test_legacy_model_and_bias(tmp_path):
    """A model trained on the old 7-layer ancillary files keeps working on new 8-layer files, and a
    per-class bias changes how readily a class is predicted."""
    import torch

    from bluecarbon.features import FEATURE_NAMES_ANC_V1, Normalizer, compute_features
    from bluecarbon.model import build_model, save_checkpoint
    from bluecarbon.predictors import load_predictor, read_ancillary

    from .conftest import write_ancillary

    d = tmp_path / "s"
    d.mkdir()
    _, lab = make_scene(d / "image.tif", d / "label.tif", size=128, seed=1)
    write_ancillary(d / "image.tif", lab)  # new 8-layer file
    with rasterio.open(d / "image.tif") as s:
        bands = s.read()
    anc = read_ancillary(d / "image.tif", bands.shape[1:])
    feats = compute_features(bands, anc=anc, names=FEATURE_NAMES_ANC_V1)
    assert feats.shape[0] == len(FEATURE_NAMES_ANC_V1)
    norm = Normalizer.fit([feats], [np.ones(bands.shape[1:], bool)])
    net = build_model("Unet", "resnet18", None, len(FEATURE_NAMES_ANC_V1), 7)
    save_checkpoint(tmp_path / "old.pt", net, "Unet", "resnet18", norm, features=FEATURE_NAMES_ANC_V1)
    pr = load_predictor(tmp_path / "old.pt", "cpu")
    cls, _ = pr.predict(bands, tile=128, overlap=32, tta=False, anc=anc)
    assert cls.shape == (128, 128)
    ck = torch.load(tmp_path / "old.pt", weights_only=False)
    ck["class_bias"] = [0, 0, 0, 50.0, 0, 0, 0]  # huge seagrass bias -> everything valid is seagrass
    torch.save(ck, tmp_path / "biased.pt")
    cls2, _ = load_predictor(tmp_path / "biased.pt", "cpu").predict(bands, tile=128, overlap=32, tta=False, anc=anc)
    assert (cls2[cls2 != 255] == 3).all()


def test_extra_year_chips_train_only(tmp_path):
    """Extra-year chips reuse the site's spatial blocks and never enter validation or test."""
    from bluecarbon.tiling import make_chips

    d = tmp_path / "s"
    d.mkdir()
    make_chips_args = dict(size=64, stride=64, min_labeled_frac=0.0, block_km=0.64)
    make_scene(d / "image.tif", d / "label.tif", size=256, seed=2)
    base = make_chips(d / "image.tif", d / "label.tif", tmp_path / "c", "site", **make_chips_args)
    extra = make_chips(d / "image.tif", d / "label.tif", tmp_path / "c", "site", tag="2020",
                       only_splits={"train"}, **make_chips_args)
    train_cells = {(r.row, r.col) for r in base if r.split == "train"}
    assert extra and all(r.split == "train" for r in extra)
    assert {(r.row, r.col) for r in extra} == train_cells
    assert all("_2020_" in r.path for r in extra)
