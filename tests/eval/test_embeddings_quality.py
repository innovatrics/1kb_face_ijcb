# SPDX-License-Identifier: MIT
"""Tests of source tags, the embeddings manifest and the quality helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb import config
from face1kb.eval import embeddings as E
from face1kb.eval import quality as Q


@pytest.mark.parametrize(
    "tag, exp",
    [
        ("aligned_112", ("aligned", 112, "", "aligned", 0)),
        ("aligned_112_adv_hfc_003", ("aligned", 112, "_adv_hfc_003", "aligned", 0)),
        ("jpeg_112_1024", ("compressed", 112, "", "jpeg", 1024)),
        ("jpeg_xl_64_512", ("compressed", 64, "", "jpeg_xl", 512)),
        ("jpeg2000_224_1024", ("compressed", 224, "", "jpeg2000", 1024)),
        ("jpeg_ai_112_tight_1024", ("compressed", 112, "_tight", "jpeg_ai", 1024)),
        (
            "ours_accurate_112_adv_liae_006_512",
            ("compressed", 112, "_adv_liae_006", "ours_accurate", 512),
        ),
        (
            "neural_mbt2018_mean_112_A1_1024",
            ("compressed", 112, "_A1", "neural_mbt2018_mean", 1024),
        ),
    ],
)
def test_parse_tag(tag, exp):
    m = E.parse_tag(tag)
    assert (m["kind"], m["res"], m["suffix"], m["codec"], m["budget"]) == exp


def test_parse_tag_unknown_and_aligned_codec():
    assert E.parse_tag("neural_unknown_112_1024") is None
    assert E.parse_tag("aligned_96", aligned_codec="")["codec"] == ""
    assert len(E.CODECS) == 12 and len(set(E.CODECS)) == 12


@pytest.fixture
def work_root(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "WORK_ROOT", tmp_path)
    monkeypatch.setattr(config, "LAYOUT", "public")
    for model in ("arcface_antelopev2", "lvface_l", "zzz"):
        d = config.embeddings_dir("colorferet", model)
        d.mkdir(parents=True)
    e = np.ones((4, E.EMB_DIM), np.float32)
    e[1] = np.nan
    np.save(config.embeddings_path("colorferet", "lvface_l", "aligned_112"), e)
    np.save(config.embeddings_path("colorferet", "lvface_l", "webp_112_1024"), e)
    np.save(config.embeddings_path("colorferet", "lvface_l", "whatever"), e)
    return tmp_path


def test_models_and_sources(work_root):
    assert E.resolve_models("anchor", "colorferet") == [
        "arcface_antelopev2",
        "lvface_l",
    ]
    assert E.resolve_models("all", "colorferet") == [
        "arcface_antelopev2",
        "lvface_l",
        "zzz",
    ]
    assert E.resolve_models("zzz, nope", "colorferet") == ["zzz"]
    src = E.list_sources("colorferet", "lvface_l")
    assert [p.stem for p, _ in src] == ["aligned_112", "webp_112_1024"]
    assert E.load_embeddings("colorferet", "lvface_l", "nope") is None
    with pytest.raises(ValueError):
        E.check_rows(E.load_embeddings("colorferet", "lvface_l", "aligned_112"), 5)


def test_manifest_roundtrip(work_root):
    df = E.scan_manifest("colorferet")
    assert list(df.columns) == list(E.MANIFEST_COLUMNS)
    assert df.source_tag.tolist() == ["aligned_112", "webp_112_1024", "whatever"]
    row = df.iloc[0]
    assert (row.n_images, row.n_ok, row.n_missing, row.dim) == (4, 3, 1, E.EMB_DIM)
    assert row.codec == "" and df.iloc[2].kind == "?"
    E.write_manifest(df, "colorferet", merge=False)
    back = E.read_manifest("colorferet")
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    E.write_manifest(df.iloc[:1].assign(n_ok=0), "colorferet")
    assert E.read_manifest("colorferet").n_ok.tolist() == [0, 3, 3]


def test_aggregate_quality_medians_and_allowlist():
    rows = []
    for codec, n in (("webp", 3), ("neural_unknown", 2), ("ours_fast", 4)):
        for i in range(n):
            rows.append(
                {
                    "image": f"i{i}",
                    "subject": "s",
                    "codec": codec,
                    "res": 112,
                    "budget": 1024,
                    **{k: float(i) for k in Q.METRIC_NAMES},
                }
            )
    df = Q.aggregate_quality(pd.DataFrame(rows), "kk")
    assert list(df.columns) == list(Q.QUALITY_COLUMNS)
    assert df.codec.tolist() == ["ours_fast", "webp"]
    assert df.n.tolist() == [4, 3] and df.psnr.tolist() == [1.5, 1.0]
    assert len(Q.aggregate_quality(pd.DataFrame(rows), "kk", codecs=None)) == 3


def test_score_images_batching():
    torch = pytest.importorskip("torch")

    calls = []

    def mean_abs(t, r):
        calls.append(t.shape[0])
        return (t - r).abs().flatten(1).mean(1)

    rng = np.random.default_rng(0)
    items = []
    for i in range(5):
        a = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
        items.append(({"image": str(i)}, a, np.zeros_like(a)))
    recs = Q.score_images(items, {"m": mean_abs}, batch=2, device="cpu")
    assert calls == [2, 2, 1]
    assert [r["image"] for r in recs] == ["0", "1", "2", "3", "4"]
    for (_, a, _), r in zip(items, recs):
        exp = torch.from_numpy(a.astype(np.float32) / 255).mean().item()
        assert r["m"] == pytest.approx(exp, abs=1e-6)
    t = Q.to_tensor(items[0][1])
    assert t.shape == (3, 8, 8) and t.dtype == torch.float32


def test_msssim_configuration_runs_at_64px():
    pytest.importorskip("piq")
    import piq
    import torch

    x = torch.rand(2, 3, 64, 64)
    y = (x + 0.05 * torch.rand_like(x)).clamp(0, 1)
    v = piq.multi_scale_ssim(
        x,
        y,
        kernel_size=Q.MSSSIM_KERNEL_SIZE,
        scale_weights=torch.tensor(Q.MSSSIM_SCALE_WEIGHTS),
        data_range=1.0,
        reduction="none",
    )
    assert v.shape == (2,) and bool(((v > 0.5) & (v <= 1)).all())
