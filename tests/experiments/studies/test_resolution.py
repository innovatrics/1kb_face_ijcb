# SPDX-License-Identifier: MIT
"""Resolution-information analysis: tags, spectra, decomposition, rendering."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb import config

from ._scripts import load
from .conftest import make_dataset

an = load("resolution", "analyze")
rd = load("resolution", "render")


def test_source_tags():
    tags = an.source_tags([])
    assert tags == sorted(f"aligned_{r}" for r in config.RESOLUTIONS)
    tags = an.source_tags(["webp"], budgets=(1024,))
    assert "webp_112_1024" in tags and "webp_112_A1_1024" not in tags
    assert all("_A" not in t and "tmp" not in t for t in tags)


def test_retention_is_monotonic_and_complete():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(224, 224)).astype(np.uint8)
    radial = an.radial_spectrum(img)
    rows = an.retention_rows("kk", radial, 1, 224)
    fr = [r["frac_energy_retained"] for r in rows]
    assert [r["res"] for r in rows] == list(config.RESOLUTIONS)
    assert np.all(np.diff(fr) > 0) and fr[-1] == 1.0
    assert [r["nyquist_cyc"] for r in rows] == [r // 2 for r in config.RESOLUTIONS]


def test_decomposition_and_eer_on_toy_embeddings(roots):
    pytest.importorskip("cv2")
    make_dataset("kk", n_subjects=5, per_subject=4, resolutions=(224,))
    subj = np.repeat(np.arange(5), 4)
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(5, 512)).astype(np.float32)
    for r in config.RESOLUTIONS:
        emb = centres[subj] + (0.1 + 1.0 / r) * rng.normal(size=(20, 512))
        path = config.embeddings_path("kk", "lvface_l", config.embedding_tag(r))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, emb.astype(np.float32))
    # stray files next to the arrays are ignored
    np.save(path.parent / "aligned_112_A1.npy", np.zeros((20, 512), np.float32))
    (path.parent / "aligned_112.npy.tmp").write_bytes(b"")
    tags = an.source_tags([])
    dec = pd.DataFrame(an.embedding_decomposition("kk", ["lvface_l"], tags))
    assert list(dec.tag) == tags
    assert np.isclose(dec.set_index("res").loc[112, "resize_cos"], 1.0)
    assert (dec.resize_cos <= 1.0 + 1e-6).all()
    eer = pd.DataFrame(an.eer_by_resolution("kk", ["lvface_l"], tags, 0))
    assert list(eer.columns) == [
        "dataset",
        "model",
        "tag",
        "kind",
        "res",
        "codec",
        "budget",
        "eer",
    ]
    assert eer.eer.between(0, 0.5).all()
    spec = pd.DataFrame(an.spectral_retention("kk", 8))
    assert (spec.n_images == 8).all()
    tex = rd.table(dec, eer, spec, "kk")
    lv = [ln for ln in tex.splitlines() if ln.startswith("\\quad LVFace-L")]
    assert len(lv) == 2 and lv[0].split(" & ")[3] == "1.000"
    arc = next(ln for ln in tex.splitlines() if ln.startswith("\\quad ArcFace"))
    assert arc.split(" & ")[1:] == ["--"] * 4 + ["-- \\\\"]


def test_clean_rows_drop_stray_tags():
    df = pd.DataFrame(
        {
            "tag": [
                "aligned_112",
                "aligned_112.npy.tmp",
                "aligned_112_A1",
                "webp_112_512",
            ]
        }
    )
    assert list(rd.clean_rows(df).tag) == ["aligned_112"]
