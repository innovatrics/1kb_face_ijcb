# SPDX-License-Identifier: MIT
"""Decode stored paper bitstreams and compare with the cached decoded PNGs.

Needs the paper bitstreams and their decoded PNG cache in the layout of
:mod:`face1kb.config` (``decoded_dir``); cells without a cache are skipped. The
cached PNGs are what every downstream embedding of the paper was computed from, so
the decode must match them exactly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

import face1kb  # noqa: E402
from face1kb import config  # noqa: E402

pytestmark = [pytest.mark.gpu, pytest.mark.weights, pytest.mark.data]
N_IMAGES = 4


@pytest.fixture(scope="module", params=list(config.CODEC_VARIANTS))
def codec(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return face1kb.load(request.param, device="cuda")


@pytest.mark.parametrize("dataset", config.DATASETS)
@pytest.mark.parametrize("res", config.RESOLUTIONS)
@pytest.mark.parametrize("budget", config.BUDGETS)
def test_decode_matches_cache(codec, dataset, res, budget):
    from PIL import Image

    name = f"ours_{codec.variant}"
    src = config.compressed_dir(dataset, res, budget, name)
    cache = config.decoded_dir(dataset, res, budget, name)
    if not config.index_csv(dataset).is_file() or not cache.is_dir():
        pytest.skip(f"no decoded cache {cache}")
    index = config.read_index(dataset).iloc[:N_IMAGES]
    for r in index.itertuples(index=False):
        rel = Path(r.rel_path)
        bitstream = src / rel.parent / (rel.stem + config.OURS_EXT)
        png = cache / rel.parent / (rel.stem + ".png")
        if not (bitstream.is_file() and png.is_file()):
            pytest.skip(f"missing {bitstream} or {png}")
        out = codec.decode(bitstream.read_bytes())
        ref = np.asarray(Image.open(png).convert("RGB"))
        assert out.shape == ref.shape
        assert np.array_equal(out, ref), (dataset, res, budget, r.rel_path)
