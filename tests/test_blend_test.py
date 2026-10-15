"""scripts/blend_test.py runs end to end on tiny models and writes a verdict table."""

import importlib.util
import shutil
from pathlib import Path

import yaml

from bluecarbon.config import load_config
from bluecarbon.tiling import make_chips, read_index, write_index

from .conftest import make_scene

ROOT = Path(__file__).resolve().parents[1]


def test_blend_test_script(tmp_path, monkeypatch):
    from bluecarbon.spectral import train_spectral
    from bluecarbon.train import train

    work = tmp_path / "runs" / "default"
    recs = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        d = work / "sites" / f"s{i}"
        d.mkdir(parents=True)
        make_scene(d / "image.tif", d / "label.tif", size=256, seed=i)
        recs += make_chips(d / "image.tif", d / "label.tif", tmp_path / "chips", f"s{i}", size=128, stride=128,
                           force_split=split)
    write_index(recs, tmp_path / "chips" / "index.json")
    recs = read_index(tmp_path / "chips" / "index.json")
    model = work / "model"
    base = {"model": {"encoder": "resnet18", "encoder_weights": None},
            "train": {"epochs": 1, "batch_size": 4, "num_workers": 0, "device": "cpu", "amp": False}}
    train(load_config(None, base), recs, model, log=lambda *_: None)
    for name in ("unet_convnext", "segformer", "prithvi"):  # stand-ins: small networks under the real file names
        train(load_config(None, base), recs, model, log=lambda *_: None, name=name)
    train_spectral(recs, model, per_class_per_chip=50, n_estimators=10, log=lambda *_: None)
    (tmp_path / "models").mkdir()
    shutil.copy(model / "model.pt", tmp_path / "models" / "live.pt")
    (tmp_path / "models" / "current.txt").write_text("live.pt\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "sites.yaml").write_text(yaml.safe_dump({"sites": [{"name": "s3", "role": "test"}]}))
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    cfg["train"]["device"] = "cpu"
    cfg["predict"] = {"tile": 128, "overlap": 32, "tta": False}
    (tmp_path / "configs" / "default.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.chdir(tmp_path)

    spec = importlib.util.spec_from_file_location("blend_test", ROOT / "scripts" / "blend_test.py")
    bt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bt)
    bt.main()
    text = (tmp_path / "blend_test.txt").read_text()
    assert "all5" in text and "prithvi+segformer" in text and ("PUBLISH" in text or "REVIEW" in text)
    assert (work / "sites" / "_analysis" / "blend_test.txt").exists()
