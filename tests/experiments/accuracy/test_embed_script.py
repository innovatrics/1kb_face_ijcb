# SPDX-License-Identifier: MIT
"""experiments/embed/compute_embeddings.py: discovery, decoding rules, row layout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb import config

from ._scripts import load

PIL = pytest.importorskip("PIL.Image")
pytest.importorskip("cv2")

REL = ["00001/a.png", "00001/b.png", "00002/c.png"]


class _MeanModel:
    """Stand-in evaluator: the per-channel mean of every 112 px crop, padded."""

    def embed(self, crops):
        x = np.stack(crops).astype(np.float32)
        assert x.shape[1:] == (112, 112, 3)
        out = np.zeros((len(crops), 512), dtype=np.float32)
        out[:, :3] = x.mean(axis=(1, 2))
        return out


@pytest.fixture
def layout(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(config, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(config, "LAYOUT", "public")
    rng = np.random.default_rng(0)
    for rel in REL:  # aligned 64 px crops
        p = config.aligned_dir("kk", 64) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        PIL.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)).save(p)
    # a WebP cell with one missing file
    for rel in REL[:2]:
        img = PIL.open(config.aligned_dir("kk", 64) / rel)
        p = (config.compressed_dir("kk", 64, 512, "webp") / rel).with_suffix(".webp")
        p.parent.mkdir(parents=True, exist_ok=True)
        img.save(p, quality=50)
    # a JPEG-AI cell: three bitstreams, a decoded cache holding two of them
    for rel in REL:
        p = config.compressed_dir("kk", 64, 512, "jpeg_ai") / rel
        p = p.with_suffix(".jpegai")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"not decoded here")
    for rel in REL[1:]:
        p = config.decoded_dir("kk", 64, 512, "jpeg_ai") / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        PIL.open(config.aligned_dir("kk", 64) / rel).save(p)
    return tmp_path


def _sources(m, codecs=("webp", "jpeg_ai", "ours_fast")):
    kinds = ["aligned", "compressed"]
    return m.discover_sources(["kk"], kinds, [64], [""], codecs, [512])


def test_discovery_tags_and_cache(layout):
    m = load("compute_embeddings", "embed")
    src = {s["tag"]: s for s in _sources(m)}
    assert sorted(src) == ["aligned_64", "jpeg_ai_64_512", "webp_64_512"]
    assert src["jpeg_ai_64_512"]["ext"] == ".jpegai"
    assert src["jpeg_ai_64_512"]["cache"] is not None
    assert src["webp_64_512"]["cache"] is None
    assert [m.decode_cost(src[t]) for t in sorted(src)] == [0, 1, 1]
    assert m.parallel_reads(src["jpeg_ai_64_512"])


def test_embed_source_rows(layout):
    m = load("compute_embeddings", "embed")
    src = {s["tag"]: s for s in _sources(m)}
    model = _MeanModel()
    out, n_ok, n_miss = m.embed_source(model, src["aligned_64"], REL, 2, 1, 0)
    assert out.shape == (3, 512) and out.dtype == np.float32
    assert (n_ok, n_miss) == (3, 0)
    assert not np.isnan(out).any()
    # missing WebP file -> NaN row at its id
    out, n_ok, n_miss = m.embed_source(model, src["webp_64_512"], REL, 256, 4, 0)
    assert (n_ok, n_miss) == (2, 1)
    assert np.isnan(out[2]).all() and not np.isnan(out[:2]).any()
    # JPEG-AI with a cache folder: a bitstream without cached PNG is a missing crop,
    # the cached ones are read from the cache (identical to the aligned crop here)
    out_ai, n_ok, n_miss = m.embed_source(model, src["jpeg_ai_64_512"], REL, 256, 4, 0)
    ref, _, _ = m.embed_source(model, src["aligned_64"], REL, 256, 1, 0)
    assert (n_ok, n_miss) == (2, 1)
    assert np.isnan(out_ai[0]).all()
    np.testing.assert_array_equal(out_ai[1:], ref[1:])
    # --limit keeps the other rows NaN
    out, n_ok, _ = m.embed_source(model, src["aligned_64"], REL, 256, 1, 1)
    assert n_ok == 1 and np.isnan(out[1:]).all()


def test_dry_run_and_manifest(layout):
    m = load("compute_embeddings", "embed")
    assert m.main(["--datasets", "kk", "--models", "anchor", "--dry-run"]) == 0
    assert not config.embeddings_manifest("kk").parent.exists()  # nothing computed
    edir = config.embeddings_dir("kk", "lvface_l")
    edir.mkdir(parents=True)
    arr = np.zeros((3, 512), np.float32)
    arr[1] = np.nan
    np.save(edir / "aligned_64.npy", arr)
    assert m.main(["--datasets", "kk", "--rebuild-manifest"]) == 0
    row = pd.read_csv(config.embeddings_manifest("kk")).iloc[0]
    got = (row.model, row.source_tag, row.kind, row.res, row.budget)
    assert got == ("lvface_l", "aligned_64", "aligned", 64, 0)
    assert (row.n_images, row.n_ok, row.n_missing, row.dim) == (3, 2, 1, 512)


def test_resolve_models():
    m = load("compute_embeddings", "embed")
    from face1kb import fr

    assert m.resolve_models("anchor") == list(fr.ANCHORS)
    assert m.resolve_models("all") == list(fr.ROSTER)
    with pytest.raises(SystemExit):
        m.resolve_models("no_such_model")


def test_undecodable_file_is_missing_and_reported(layout, caplog):
    m = load("compute_embeddings", "embed")
    bad = (config.compressed_dir("kk", 64, 512, "webp") / REL[2]).with_suffix(".webp")
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(b"not a webp file")
    src = {s["tag"]: s for s in _sources(m)}
    with caplog.at_level("WARNING"):
        out, n_ok, n_miss = m.embed_source(
            _MeanModel(), src["webp_64_512"], REL, 256, 4, 0
        )
    assert (n_ok, n_miss) == (2, 1) and np.isnan(out[2]).all()
    assert "1 files could not be decoded" in caplog.text
