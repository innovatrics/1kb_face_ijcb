# SPDX-License-Identifier: MIT
"""Li-AE proxy auto-encoder: architecture and weight-file I/O (CPU)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from face1kb.adversarial import proxy as P  # noqa: E402

KEYS_HEAD = [
    "enc1.0.weight",
    "enc1.0.bias",
    "enc1.1.weight",
    "enc1.1.bias",
    "enc1.1.running_mean",
    "enc1.1.running_var",
    "enc1.1.num_batches_tracked",
]


def seeded_net(width: int = P.DEFAULT_WIDTH, seed: int = 0) -> P.ProxyAE:
    torch.manual_seed(seed)
    net = P.ProxyAE(width=width)
    # move the batch-norm statistics away from their initial values
    net.train()
    with torch.no_grad():
        net(torch.rand(4, 3, 112, 112))
    return net.eval()


def test_shapes_and_range():
    net = P.ProxyAE().eval()
    x = torch.rand(2, 3, 112, 112)
    with torch.no_grad():
        recon, feats = net(x)
        enc = net.encode(x)
    assert recon.shape == (2, 3, 112, 112)
    assert feats["mid"].shape == (2, 64, 28, 28)
    assert feats["code"].shape == (2, 256, 7, 7)
    assert torch.equal(enc["mid"], feats["mid"])
    assert torch.equal(enc["code"], feats["code"])
    assert float(recon.min()) > 0.0 and float(recon.max()) < 1.0


def test_parameter_count_and_keys():
    net = P.ProxyAE()
    assert P.count_parameters(net) == 1_381_443
    sd = net.state_dict()
    assert len(sd) == 51
    assert list(sd)[: len(KEYS_HEAD)] == KEYS_HEAD
    assert list(sd)[-2:] == ["dec4.0.weight", "dec4.0.bias"]


def test_width():
    net = P.ProxyAE(width=8).eval()
    with torch.no_grad():
        feats = net.encode(torch.rand(1, 3, 112, 112))
    assert feats["mid"].shape == (1, 16, 28, 28)
    assert feats["code"].shape == (1, 64, 7, 7)


def test_save_load_roundtrip(tmp_path):
    net = seeded_net()
    path = P.save_proxy(net, tmp_path / "sub" / "p.safetensors", {"n_ids": 3})
    back = P.load_proxy(path)
    assert not back.training
    assert all(not p.requires_grad for p in back.parameters())
    sd, bd = net.state_dict(), back.state_dict()
    assert list(sd) == list(bd)
    assert all(torch.equal(sd[k], bd[k]) and sd[k].dtype == bd[k].dtype for k in sd)
    _, meta = P.read_proxy_file(path)
    assert meta["face1kb.n_ids"] == "3"
    assert meta["face1kb.architecture"] == "ProxyAE"
    assert meta["face1kb.format_version"] == P.PROXY_FORMAT_VERSION
    assert meta["face1kb.width"] == "32"
    assert meta["face1kb.parameters"] == "1381443"
    x = torch.rand(2, 3, 112, 112)
    with torch.no_grad():
        assert torch.equal(net(x)[0], back(x)[0])


def test_width_from_metadata(tmp_path):
    path = P.save_proxy(seeded_net(width=8), tmp_path / "p8.safetensors")
    assert P.load_proxy(path).width == 8


def test_strict_loading(tmp_path):
    sd = {k: v.clone() for k, v in seeded_net().state_dict().items()}
    sd.pop("enc4.1.running_var")
    path = tmp_path / "bad.safetensors"
    save_file(sd, str(path), metadata={"face1kb.architecture": "ProxyAE"})
    with pytest.raises(RuntimeError, match="running_var"):
        P.load_proxy(path)


@pytest.mark.parametrize(
    "meta, match",
    [
        ({"face1kb.architecture": "Other"}, "architecture"),
        ({"face1kb.format_version": "99"}, "format version"),
    ],
)
def test_rejects_foreign_files(tmp_path, meta, match):
    sd = {k: v.clone() for k, v in seeded_net().state_dict().items()}
    path = tmp_path / "x.safetensors"
    save_file(sd, str(path), metadata=meta)
    with pytest.raises(ValueError, match=match):
        P.load_proxy(path)


def test_lfs_pointer(tmp_path):
    path = tmp_path / "liae_proxy.safetensors"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 5536012\n"
    )
    with pytest.raises(RuntimeError, match="git lfs pull"):
        P.load_proxy(path)


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        P.load_proxy(tmp_path / "none.safetensors")


def test_torch_checkpoint(tmp_path):
    net = seeded_net(width=8)
    ckpt = tmp_path / "proxy.pt"
    torch.save({"model": net.state_dict(), "width": 8, "n_ids": 5}, ckpt)
    back = P.load_proxy(ckpt)
    assert back.width == 8
    assert all(
        torch.equal(v, back.state_dict()[k]) for k, v in net.state_dict().items()
    )
    _, meta = P.read_proxy_file(ckpt)
    assert meta == {"face1kb.width": "8", "face1kb.n_ids": "5"}
    bare = tmp_path / "bare.pt"
    torch.save(seeded_net().state_dict(), bare)
    assert P.load_proxy(bare).width == P.DEFAULT_WIDTH


def test_default_path(monkeypatch, tmp_path):
    from face1kb import config

    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path)
    assert P.default_proxy_path() == tmp_path / "liae_proxy.safetensors"


def test_default_path_missing(monkeypatch, tmp_path):
    from face1kb import config

    monkeypatch.setattr(config, "WEIGHTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="released proxy weights not found"):
        P.load_proxy()
