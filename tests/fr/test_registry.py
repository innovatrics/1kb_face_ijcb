# SPDX-License-Identifier: MIT
"""Registry, constants and user hooks of face1kb.fr (no model files needed)."""

from __future__ import annotations

import numpy as np
import pytest

from face1kb import fr
from face1kb.fr import registry

PAPER_ROSTER = {
    "arcface_antelopev2",
    "cvlface_ir101",
    "cvlface_vit_b",
    "edgeface_base",
    "edgeface_s",
    "edgeface_xs",
    "edgeface_xxs",
    "lvface_b",
    "lvface_l",
    "lvface_s",
    "lvface_t",
    "topofr_r50",
    "topofr_r100",
    "topofr_r200",
}


@pytest.fixture()
def clean_registry():
    before = dict(fr.REGISTRY)
    yield
    fr.REGISTRY.clear()
    fr.REGISTRY.update(before)


def test_roster_matches_the_paper():
    assert set(fr.ROSTER) == PAPER_ROSTER
    assert len(fr.ROSTER) == 14
    assert fr.ROSTER[0] == fr.ARCFACE
    assert all(fr.REGISTRY[n].builtin for n in fr.ROSTER)
    assert set(fr.MODEL_LABELS) == PAPER_ROSTER


def test_anchor_constants():
    assert fr.ANCHORS == (
        "arcface_antelopev2",
        "lvface_l",
        "topofr_r100",
        "edgeface_xs",
    )
    assert (fr.ARCFACE, fr.LVFACE_L, fr.TOPOFR_R100, fr.EDGEFACE_XS) == fr.ANCHORS
    assert fr.HELDOUT in fr.ROSTER and fr.HELDOUT not in fr.ANCHORS
    # the public id-cos default is a built-in, independent of the anchors and of
    # the codec training models (EdgeFace)
    assert fr.IDCOS_DEFAULT in fr.ROSTER and fr.IDCOS_DEFAULT not in fr.ANCHORS
    assert fr.REGISTRY[fr.IDCOS_DEFAULT].family != "edgeface"


def test_families_and_variants():
    fam = {n: fr.REGISTRY[n].family for n in fr.ROSTER}
    assert sum(f == "lvface" for f in fam.values()) == 4
    assert sum(f == "topofr" for f in fam.values()) == 3
    assert sum(f == "edgeface" for f in fam.values()) == 4
    assert sum(f == "cvlface" for f in fam.values()) == 2
    assert fam["arcface_antelopev2"] == "arcface"
    assert fr.REGISTRY["cvlface_vit_b"].variant == "vit_b"


def test_unknown_model_raises_keyerror():
    with pytest.raises(KeyError, match="register_onnx"):
        fr.load("no_such_model", device="cpu")


def test_builtins_cannot_be_replaced(clean_registry, tmp_path):
    with pytest.raises(ValueError, match="built-in"):
        fr.register_onnx("lvface_l", tmp_path / "x.onnx")
    with pytest.raises(ValueError, match="built-in"):
        fr.unregister("arcface_antelopev2")


def test_register_embedder_roundtrip(clean_registry):
    class Const:
        def __init__(self, device):
            self.device = device

        def embed(self, crops):
            return np.ones((len(crops), 4), np.float32)

    spec = fr.register_embedder("const", Const, label="Constant")
    assert spec.family == "custom" and fr.label("const") == "Constant"
    emb = fr.load("const", device="cpu")
    assert emb.device == "cpu"
    assert emb.embed([np.zeros((112, 112, 3), np.uint8)] * 3).shape == (3, 4)
    with pytest.raises(ValueError, match="already registered"):
        fr.register_embedder("const", Const)
    fr.register_embedder("const", Const, overwrite=True)
    fr.unregister("const")
    assert "const" not in fr.REGISTRY


def test_register_embedder_checks_the_contract(clean_registry):
    fr.register_embedder("bad", lambda device: object())
    with pytest.raises(TypeError, match="embed"):
        fr.load("bad", device="cpu")


def test_register_onnx_missing_file(clean_registry, tmp_path):
    fr.register_onnx("mine", tmp_path / "missing.onnx")
    with pytest.raises(FileNotFoundError):
        fr.load("mine", device="cpu")


def test_register_torch(clean_registry):
    torch = pytest.importorskip("torch")

    class MeanNet(torch.nn.Module):
        def forward(self, x, scale=1.0):
            return (x.mean(dim=(2, 3)) * scale, None)

    fr.register_torch("meannet", MeanNet, forward_kwargs={"scale": 2.0})
    emb = fr.load("meannet", device="cpu")
    assert isinstance(emb.model, torch.nn.Module)
    crops = np.random.default_rng(0).integers(0, 256, (2, 112, 112, 3), dtype=np.uint8)
    out = emb.embed(crops)
    ref = (((crops.astype(np.float32) / 255.0) - 0.5) / 0.5).mean(axis=(1, 2)) * 2.0
    assert out.dtype == np.float32 and out.shape == (2, 3)
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_register_torch_factory_with_device(clean_registry):
    torch = pytest.importorskip("torch")
    seen = []

    def factory(device):
        seen.append(device)
        return torch.nn.Flatten()

    fr.register_torch("flat", factory, color="BGR", normalize="[0,1]")
    out = fr.load("flat", device="cpu").embed([np.full((112, 112, 3), 255, np.uint8)])
    assert seen == ["cpu"] and out.shape == (1, 3 * 112 * 112)
    np.testing.assert_array_equal(out, 1.0)


def test_torch_builder_rejects_non_torch_models():
    with pytest.raises(ValueError):
        fr.torch_builder("lvface_l")
    with pytest.raises(ValueError):
        fr.torch_builder("cvlface_ir101")


def test_edgeface_weights_follow_the_verify_flag(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("timm")
    from face1kb import config
    from face1kb.codec import identity_loss
    from face1kb.fr import download, families

    monkeypatch.setattr(config, "MODELS_ROOT", tmp_path)
    ref = identity_loss.build_edgeface("edgeface_xxs", pretrained=False)
    path = tmp_path / "edgeface" / "edgeface_xxs.pt"
    path.parent.mkdir()
    torch.save(ref.state_dict(), path)  # a valid state dict, but not the pinned file

    def codec_fetch(*args, **kwargs):
        raise AssertionError("fr.load must locate and verify the file itself")

    monkeypatch.setattr(identity_loss, "fetch_pretrained", codec_fetch)
    with pytest.raises(RuntimeError, match="--only edgeface_xxs --force"):
        families.build_edgeface_net("edgeface_xxs", download=False)
    hashed = []
    monkeypatch.setattr(download, "sha256_file", lambda p: hashed.append(p) or "0")
    net = families.build_edgeface_net("edgeface_xxs", download=False, verify=False)
    assert hashed == [] and not net.training
    got = net.state_dict()
    assert all(torch.equal(got[k], v) for k, v in ref.state_dict().items())


def test_import_is_lightweight():
    import subprocess
    import sys

    code = (
        "import sys, face1kb.fr; "
        "print(any(m in sys.modules for m in ('torch', 'onnxruntime', 'insightface')))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_registry_module_exports():
    assert registry.load is fr.load
    assert set(fr.__all__) <= set(dir(fr))


PLUGIN = """
import numpy as np
from face1kb import fr


class Ones:
    def __init__(self, device):
        self.device = device

    def embed(self, crops):
        return np.ones((len(crops), 8), np.float32)


fr.register_embedder("{name}", Ones, label="Plugin model")
"""


@pytest.fixture()
def no_plugins(clean_registry, monkeypatch):
    monkeypatch.setattr(registry, "_PLUGINS_LOADED", set())
    monkeypatch.delenv(fr.PLUGINS_ENV, raising=False)


def test_plugin_file_from_the_environment(no_plugins, monkeypatch, tmp_path):
    path = tmp_path / "my_matchers.py"
    path.write_text(PLUGIN.format(name="plugin_file_model"))
    monkeypatch.setenv(fr.PLUGINS_ENV, str(path))
    assert "plugin_file_model" not in fr.REGISTRY
    emb = fr.load("plugin_file_model", device="cpu")  # imports the plugin on demand
    assert emb.embed([np.zeros((112, 112, 3), np.uint8)]).shape == (1, 8)
    assert fr.label("plugin_file_model") == "Plugin model"
    assert fr.load_plugins() == []  # imported once per process
    assert "plugin_file_model" in fr.available()


def test_plugin_module_by_name(no_plugins, monkeypatch, tmp_path):
    (tmp_path / "face1kb_test_plugin_mod.py").write_text(
        PLUGIN.format(name="plugin_module_model")
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert fr.load_plugins("face1kb_test_plugin_mod, ") == ["face1kb_test_plugin_mod"]
    assert "plugin_module_model" in fr.REGISTRY
    monkeypatch.delitem(__import__("sys").modules, "face1kb_test_plugin_mod")


def test_missing_plugin_file(no_plugins, tmp_path):
    with pytest.raises(FileNotFoundError, match="plugin"):
        fr.load_plugins(str(tmp_path / "nope.py"))


def test_plugin_path_without_py_suffix(no_plugins, tmp_path):
    path = tmp_path / "my_matchers"
    path.write_text(PLUGIN.format(name="plugin_nosuffix_model"))
    with pytest.raises(ValueError, match="not a Python file"):
        fr.load_plugins(str(path))
    assert "plugin_nosuffix_model" not in fr.REGISTRY
