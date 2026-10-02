# SPDX-License-Identifier: MIT
"""Installation check: embeddings of two synthetic crops vs reference values.

The reference summaries in ``golden_embeddings.json`` were produced with the paper
environment (CUDA, RTX 2080 Ti). Each evaluator runs in its own process (the two
CVLface snapshots cannot share one) and is skipped when its files are not under
``FACE1KB_MODELS_ROOT`` (``python scripts/fetch_models.py``). The tolerance
(1e-3) is far below the differences between evaluators and far above the CPU/GPU
float noise (~1e-6).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from face1kb import fr

from ._helpers import run_helper

HERE = Path(__file__).resolve().parent
GOLDEN = json.loads((HERE / "golden_embeddings.json").read_text())
TOL = 1e-3


def test_golden_file_covers_the_roster():
    assert set(GOLDEN) == set(fr.ROSTER)
    for name, g in GOLDEN.items():
        assert g["dim"] == fr.EMB_DIM, name
        assert len(g["norm"]) == 2 and len(g["head"]) == 2, name


@pytest.mark.parametrize("name", fr.ROSTER)
def test_evaluator_reproduces_the_reference_embeddings(name, device):
    for mod in ("torch", "onnxruntime"):
        if importlib.util.find_spec(mod) is None:
            pytest.skip(f"{mod} is not installed")
    if not fr.present(name):
        pytest.skip(f"{name} is not under FACE1KB_MODELS_ROOT (fetch_models.py)")
    got = run_helper("_golden.py", name, "--device", device)[name]
    ref = GOLDEN[name]
    assert got["dim"] == ref["dim"]
    np.testing.assert_allclose(got["norm"], ref["norm"], rtol=TOL)
    np.testing.assert_allclose(got["head"], ref["head"], atol=TOL)
    assert abs(got["cos01"] - ref["cos01"]) < TOL
