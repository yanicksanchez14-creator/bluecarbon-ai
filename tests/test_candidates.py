"""Competing networks: Prithvi weight loading and a non-U-Net candidate trained, saved, reloaded and blended."""

import numpy as np
import rasterio
import torch

from bluecarbon.config import load_config
from bluecarbon.features import FEATURE_NAMES
from bluecarbon.tiling import make_chips, read_index, write_index

from .conftest import make_scene


def test_prithvi_loads_mae_checkpoint_layout(tmp_path, monkeypatch):
    """The released checkpoint is the whole MAE (Conv3d patch embed, decoder); only the encoder is taken."""
    from timm.models.vision_transformer import Block

    from bluecarbon import prithvi

    blocks = torch.nn.ModuleList([Block(768, 12, 4.0, qkv_bias=True) for _ in range(12)])
    sd = {f"blocks.{k}": v for k, v in blocks.state_dict().items()}
    sd |= {"patch_embed.proj.weight": torch.randn(768, 6, 1, 16, 16), "patch_embed.proj.bias": torch.zeros(768),
           "cls_token": torch.randn(1, 1, 768), "norm.weight": torch.ones(768), "norm.bias": torch.zeros(768),
           "pos_embed": torch.zeros(1, 589, 768), "decoder_embed.weight": torch.zeros(512, 768)}
    f = tmp_path / "Prithvi_EO_V1_100M.pt"
    torch.save(sd, f)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: str(f))
    m = prithvi.build_prithvi(len(FEATURE_NAMES), 7, FEATURE_NAMES, pretrained=True)
    assert m.pretrained_loaded
    assert torch.allclose(m.patch_embed.weight, sd["patch_embed.proj.weight"][:, :, 0])
    assert torch.allclose(m.blocks[3].attn.qkv.weight, sd["blocks.3.attn.qkv.weight"])
    assert m(torch.randn(1, len(FEATURE_NAMES), 64, 64)).shape == (1, 7, 64, 64)


def test_candidate_trains_and_blends(tmp_path):
    from bluecarbon.ensemble import build
    from bluecarbon.predictors import load_predictor, model_card
    from bluecarbon.spectral import train_spectral
    from bluecarbon.train import train

    cfg = load_config(None, {
        "model": {"arch": "Segformer", "encoder": "mit_b0", "encoder_weights": None},
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
    res = train(cfg, recs, out, log=lambda *_: None, name="segformer")
    assert (out / "segformer.pt").exists() and (out / "segformer_metrics.json").exists()
    assert not (out / "model.pt").exists() and res["arch"] == "Segformer"
    assert model_card(out / "segformer.pt")["arch"] == "Segformer"
    train_spectral(recs, out, per_class_per_chip=100, n_estimators=20, log=lambda *_: None)
    m = build(out, recs, "cpu", tta=False, log=lambda *_: None, deep="segformer.pt")
    assert m["deep"] == "segformer.pt"
    pr = load_predictor(out / "ensemble.ensemble.json", "cpu")
    assert "Segformer" in pr.meta["arch"]
    with rasterio.open(tmp_path / "site3" / "image.tif") as s:
        bands = s.read()
    cls, _ = pr.predict(bands, tile=128, overlap=32, tta=False)
    assert cls.shape == bands.shape[1:] and np.isin(np.unique(cls), list(range(7)) + [255]).all()


def test_big_model_stored_in_parts(tmp_path, monkeypatch):
    """A model over GitHub's limit is split into parts and still loads (joined once into a cache)."""
    from bluecarbon.features import Normalizer
    from bluecarbon.model import build_model, save_checkpoint, split_file
    from bluecarbon.predictors import load_predictor, model_card

    monkeypatch.setenv("BLUECARBON_CACHE", str(tmp_path / "cache"))
    f = FEATURE_NAMES
    net = build_model("Unet", "resnet18", None, len(f), 7)
    norm = Normalizer(np.zeros(len(f), np.float32), np.ones(len(f), np.float32))
    p = tmp_path / "models" / "big.pt"
    save_checkpoint(p, net, "Unet", "resnet18", norm, features=f)
    size = p.stat().st_size
    parts = split_file(p, part_mb=10)
    assert len(parts) >= 2 and not p.exists() and all(q.stat().st_size <= 10 * 1024 * 1024 for q in parts)
    pr = load_predictor(p, "cpu")
    assert pr.kind == "unet" and model_card(p)["arch"] == "Unet"
    assert next((tmp_path / "cache").glob("big-*.pt")).stat().st_size == size
