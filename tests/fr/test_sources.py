# SPDX-License-Identifier: MIT
"""Pinned sources of the evaluators (no network, no model files)."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

import pytest

from face1kb import fr
from face1kb.fr import sources

HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _sources():
    return [fr.model_source(n) for n in fr.ROSTER]


def test_every_builtin_has_a_source():
    assert set(sources.builtin_names()) == set(fr.ROSTER)


@pytest.mark.parametrize("name", fr.ROSTER)
def test_files_are_pinned(name):
    if name.startswith("edgeface"):
        pytest.importorskip("torch")  # the EdgeFace spec lives in face1kb.codec
    src = fr.model_source(name)
    assert src.files, name
    for f in src.files:
        assert HEX64.match(f.sha256), (name, f.path)
        assert f.size > 0, (name, f.path)
        p = PurePosixPath(f.path)
        assert not p.is_absolute() and ".." not in p.parts
        n_sources = bool(f.urls) + (f.archive is not None) + (f.gdrive_id is not None)
        assert n_sources == 1, f.path
        for url in f.urls:
            assert url.startswith("https://")
        if f.archive is not None:
            assert f.member and HEX64.match(f.archive.sha256)
        if f.gdrive_id is not None:
            assert f.manual_url and f.gdrive_id in f.manual_url
    assert src.weights_licence and src.homepage.startswith("https://")
    assert name in src.notice and "not part of face1kb" in src.notice
    assert src.notice.count(src.training_data) == 1  # named once, not twice


def test_hugging_face_and_github_downloads_are_pinned_to_a_revision():
    urls = [
        u
        for n in fr.ROSTER
        if not n.startswith("edgeface")
        for f in fr.model_source(n).files
        for u in f.urls
    ]
    for u in urls:
        if "huggingface.co" in u:
            assert re.search(r"/resolve/[0-9a-f]{40}/", u), u
        if "githubusercontent" in u:
            assert sources.TOPOFR_COMMIT in u


def test_paths_do_not_collide_across_models():
    seen: dict[str, str] = {}
    for n in fr.ROSTER:
        if n.startswith("edgeface"):
            continue
        for f in fr.model_source(n).files:
            if f.path in seen:
                # shared files (TopoFR code) must have the same content
                assert seen[f.path] == f.sha256
            seen[f.path] = f.sha256


def test_cvlface_snapshots_ship_the_files_the_loader_uses():
    for v in ("ir101", "vit_b"):
        rel = {f.path.split("/", 2)[2] for f in sources.cvlface_files(v)}
        assert {"wrapper.py", "models/__init__.py", "pretrained_model/model.pt"} <= rel
        assert "pretrained_model/model.yaml" in rel
        assert not any(r.endswith("model.safetensors") for r in rel)
    vit = {f.path.rsplit("/", 1)[-1] for f in sources.cvlface_files("vit_b")}
    assert {"kprpe_shared.py", "rpe_index.py", "setup.py"} <= vit


def test_arcface_is_glintr100_from_antelopev2():
    (f,) = fr.model_source("arcface_antelopev2").files
    assert f.path.endswith("antelopev2/glintr100.onnx")
    assert f.member == "antelopev2/glintr100.onnx"
    assert "w600k" not in f.path


def test_topofr_code_is_fetched_not_vendored():
    import face1kb

    pkg = Path(face1kb.__file__).parent
    assert not list(pkg.rglob("iresnet.py"))
    for n in ("topofr_r50", "topofr_r100", "topofr_r200"):
        paths = [f.path for f in fr.model_source(n).files]
        assert sum(p.startswith(sources.TOPOFR_CODE_DIR) for p in paths) == 4


def test_edgeface_sources_follow_the_codec_spec():
    pytest.importorskip("torch")
    from face1kb.codec.identity_loss import EDGEFACE_ARCHS, EDGEFACE_WEIGHTS

    for name in ("edgeface_xxs", "edgeface_xs", "edgeface_s", "edgeface_base"):
        spec = EDGEFACE_WEIGHTS[EDGEFACE_ARCHS[name]]
        (f,) = fr.model_source(name).files
        assert f.path == f"{spec.subdir}/{spec.filename}"
        assert f.sha256 == spec.sha256 and f.urls == tuple(spec.urls)
