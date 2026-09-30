# SPDX-License-Identifier: MIT
"""Differential check against embedding arrays computed earlier (maintainers).

For every evaluator whose files are present, embed the first 64 aligned 112 px
crops (index order) of each dataset and compare them with rows 0..63 of
``<work>/<dataset>/embeddings/<model>/aligned_112.npy``. The paper's
``aligned_112`` arrays are reproduced bit for bit by batches of 256 on the GPU model
and software stack that computed them. Other batch sizes, GPU models or library
versions change the kernel selection and give float noise only, hence the cosine
threshold instead of an exact comparison.
"""

from __future__ import annotations

import importlib.util

import pytest

from face1kb import config, fr

from ._helpers import run_helper

MIN_COS = 0.9999


@pytest.mark.data
@pytest.mark.gpu
@pytest.mark.parametrize("name", fr.ROSTER)
def test_first_crops_match_the_stored_arrays(name):
    for mod in ("torch", "onnxruntime"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} is not installed")
    if not fr.present(name):
        pytest.skip(f"{name} is not under FACE1KB_MODELS_ROOT")
    if not any(
        config.embeddings_path(ds, name, "aligned_112").is_file()
        for ds in config.DATASETS
    ):
        pytest.skip(f"no stored aligned_112 embeddings of {name}")
    res = run_helper("_stored.py", name, "--n", "64", "--batch", "64")
    checked = {ds: r for ds, r in res.items() if r is not None}
    assert checked
    for ds, r in checked.items():
        assert r["rows"] == 64, (ds, r)
        assert r["min_cos"] >= MIN_COS, (ds, r)
