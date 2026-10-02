# SPDX-License-Identifier: MIT
"""Differential tests against stored benchmark bitstreams and decoded caches.

They run when ``FACE1KB_DATA_ROOT`` / ``FACE1KB_WORK_ROOT`` hold the aligned crops,
the index and compressed cells (e.g. the paper's store with ``FACE1KB_LAYOUT=legacy``,
or cells regenerated with the same library versions), and skip otherwise. Each test
uses the first crops of the canonical index order.

Byte identity of the classical codecs needs the library builds of the paper
(Pillow 12.2.0, pillow-heif 1.4.0, pillow-jxl-plugin 1.3.7); with other versions
those tests are skipped.
"""

from __future__ import annotations

from importlib import metadata
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from face1kb import config
from face1kb.baselines import classical, decode_file, decoded_cache_path, get_codec

N = 3
PAPER_LIBS = {"pillow": "12.2.0", "pillow-heif": "1.4.0", "pillow-jxl-plugin": "1.3.7"}
CELLS = [("colorferet", 112, 1024), ("colorferet", 224, 512), ("kk", 224, 1024)]


def _version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def _paper_libs() -> bool:
    return all(_version(d) == v for d, v in PAPER_LIBS.items())


def _items(ds, res, budget, codec, n=N):
    """``(crop, stored bitstream)`` pairs of the first ``n`` index rows."""
    try:
        idx = config.read_index(ds)
    except (FileNotFoundError, OSError):
        pytest.skip(f"no index for {ds}")
    ext = get_codec(codec).ext if not codec.startswith("ours_") else config.OURS_EXT
    out = []
    for rel in idx.rel_path[:n]:
        src = config.aligned_dir(ds, res) / rel
        stored = config.compressed_dir(ds, res, budget, codec) / Path(rel).with_suffix(
            ext
        )
        if src.is_file() and stored.is_file():
            out.append((src, stored))
    if not out:
        pytest.skip(f"no stored {codec} bitstreams for {ds} {res} px / {budget} B")
    return out


def _load(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


@pytest.mark.data
@pytest.mark.parametrize(("ds", "res", "budget"), CELLS)
@pytest.mark.parametrize("codec", (*classical.CODECS, "jpeg_fzt"))
def test_cpu_codecs_reproduce_stored_bitstreams(ds, res, budget, codec):
    if not _paper_libs():
        pytest.skip(f"byte identity needs {PAPER_LIBS}")
    entry = get_codec(codec)
    for src, stored in _items(ds, res, budget, codec):
        data, info = entry.encode_to_budget(_load(src), budget)
        assert data == stored.read_bytes(), stored
        assert info["size"] == len(data)


@pytest.mark.data
@pytest.mark.parametrize(("ds", "res", "budget"), CELLS)
def test_jpeg_fzt_decode_of_stored_files(ds, res, budget):
    from face1kb.baselines import jpeg_fzt

    for _src, stored in _items(ds, res, budget, "jpeg_fzt"):
        out = decode_file(stored)
        assert out.shape == (res, res, 3)
        low = _load(stored)
        ref = jpeg_fzt.decode_stage(low, (res, res), jpeg_fzt.load_ibf(), h=2, rad=1)
        assert np.array_equal(out, ref)


@pytest.mark.data
@pytest.mark.gpu
@pytest.mark.parametrize("codec", ("neural_bmshj2018", "neural_mbt2018_mean"))
def test_compressai_reproduces_stored_bitstreams(codec):
    pytest.importorskip("compressai")
    entry = get_codec(codec)
    for src, stored in _items("colorferet", 112, 1024, codec):
        data, info = entry.encode_to_budget(_load(src), 1024)
        assert data == stored.read_bytes(), stored
        a, b = decode_file(stored), decode_file(stored)
        assert a.shape == (112, 112, 3)
        # CUDA decodes repeat up to 1 grey level (non-deterministic cuDNN kernels)
        assert np.abs(a.astype(int) - b.astype(int)).max() <= 1


def _jpegai_ready() -> bool:
    from face1kb.baselines import jpeg_ai

    return jpeg_ai.available()


@pytest.mark.data
@pytest.mark.gpu
@pytest.mark.parametrize(("ds", "res", "budget"), [("colorferet", 224, 512)])
def test_jpegai_decode_matches_cache(ds, res, budget):
    if not _jpegai_ready():
        pytest.skip("JPEG-AI reference software not set up")
    checked = 0
    for _src, stored in _items(ds, res, budget, "jpeg_ai", n=2):
        cache = decoded_cache_path(stored)
        if cache is None or not cache.is_file():
            continue
        assert np.array_equal(decode_file(stored), _load(cache)), stored
        checked += 1
    if not checked:
        pytest.skip("no decoded JPEG-AI cache")


@pytest.mark.data
@pytest.mark.gpu
def test_jpegai_reproduces_stored_bitstreams():
    # The first Color FERET crops at 224 px / 512 B, whose stored streams match the
    # benchmark fit; other Color FERET cells have a prefix of streams written with
    # another fit (docs/baselines.md, Reproducibility limits).
    if not _jpegai_ready():
        pytest.skip("JPEG-AI reference software not set up")
    for src, stored in _items("colorferet", 224, 512, "jpeg_ai", n=2):
        data, info = get_codec("jpeg_ai").encode_to_budget(_load(src), 512)
        assert data == stored.read_bytes(), stored
        assert info["fitted"]


@pytest.mark.data
@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.parametrize("codec", ("ours_fast", "ours_accurate"))
def test_ours_decode_dispatch_matches_cache(codec):
    checked = 0
    for _src, stored in _items("kk", 224, 512, codec):
        cache = decoded_cache_path(stored)
        if cache is None or not cache.is_file():
            continue
        assert np.array_equal(decode_file(stored), _load(cache)), stored
        checked += 1
    if not checked:
        pytest.skip(f"no decoded {codec} cache")
