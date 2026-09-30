# SPDX-License-Identifier: MIT
"""Golden tests: re-encode stored paper bitstreams and fixed synthetic vectors.

``test_paper_bitstreams`` needs the aligned crops and the paper bitstreams (produced
by the Ours compression stage) in the layout of :mod:`face1kb.config`; it re-encodes
the first images of the canonical index with ``paper_compat=True`` and requires
byte-identical output. ``test_synthetic_vectors`` compares the encodings of a fixed
synthetic image with hashes recorded on an NVIDIA Turing GPU (sm_75, torch 2.10,
compressai 1.2.8); a failure on other hardware means the bitstreams of that machine
are not interchangeable with the reference ones.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

import face1kb  # noqa: E402
from face1kb import config  # noqa: E402

from ._synthetic import smooth_face  # noqa: E402

GOLDEN = Path(__file__).with_name("golden_synthetic.json")
CELLS = [(64, 1024), (96, 512), (112, 1024), (168, 1024), (224, 512)]
N_IMAGES = 3


@pytest.fixture(scope="module", params=list(config.CODEC_VARIANTS))
def codec(request):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return face1kb.load(request.param, device="cuda")


def _index(dataset):
    p = config.index_csv(dataset)
    if not p.is_file():
        pytest.skip(f"no index for {dataset}: {p}")
    return config.read_index(dataset).iloc[:N_IMAGES]


def _crop(dataset, res, rel_path):
    from PIL import Image

    p = config.aligned_dir(dataset, res) / Path(rel_path).with_suffix(".png")
    if not p.is_file():
        pytest.skip(f"missing crop {p}")
    return np.asarray(Image.open(p).convert("RGB"))


@pytest.mark.gpu
@pytest.mark.weights
@pytest.mark.data
@pytest.mark.parametrize("dataset", config.DATASETS)
@pytest.mark.parametrize("res,budget", CELLS)
def test_paper_bitstreams(codec, dataset, res, budget):
    codec_name = f"ours_{codec.variant}"
    cell = config.compressed_dir(dataset, res, budget, codec_name)
    for r in _index(dataset).itertuples(index=False):
        rel = Path(r.rel_path)
        stored_p = cell / rel.parent / (rel.stem + config.OURS_EXT)
        if not stored_p.is_file():
            pytest.skip(f"missing paper bitstream {stored_p}")
        data = codec.encode(_crop(dataset, res, r.rel_path), budget, paper_compat=True)
        assert data == stored_p.read_bytes(), (dataset, res, budget, r.rel_path)


def _synthetic_hashes(codec) -> dict:
    out = {}
    for res, budget in CELLS:
        data = codec.encode(smooth_face(res), budget, paper_compat=True)
        out[f"{codec.variant}/{res}/{budget}"] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return out


@pytest.mark.gpu
@pytest.mark.weights
def test_synthetic_vectors(codec):
    expected = json.loads(GOLDEN.read_text())["vectors"]
    got = _synthetic_hashes(codec)
    for key, val in got.items():
        assert val == expected[key], key
