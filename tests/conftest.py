# SPDX-License-Identifier: MIT
"""Shared pytest configuration: skip marked tests whose resources are missing.

Markers (declared in ``pyproject.toml``):

* ``gpu`` -- needs a CUDA device (skipped when ``torch.cuda.is_available()`` is
  false or torch is not installed);
* ``data`` -- needs the datasets (skipped unless ``FACE1KB_DATA_ROOT`` is set and
  exists; tests may skip further when specific files are absent);
* ``weights`` -- needs the released weights (skipped when ``weights/*.safetensors``
  are missing or are still git-lfs pointer files).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def cuda_available() -> bool:
    """Return True if torch is importable and sees a CUDA device."""
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def weights_available() -> bool:
    """Return True if both released weight files are present (not LFS pointers)."""
    from face1kb import config

    for variant in config.CODEC_VARIANTS:
        p = config.weights_path(variant)
        if not p.is_file() or p.stat().st_size < 1024:
            return False
        with open(p, "rb") as f:
            if f.read(40).startswith(b"version https://git-lfs"):
                return False
    return True


def data_available() -> bool:
    """Return True if FACE1KB_DATA_ROOT is set and points to an existing folder."""
    root = os.environ.get("FACE1KB_DATA_ROOT", "").strip()
    return bool(root) and Path(root).is_dir()


def pytest_collection_modifyitems(config, items):
    """Attach skip markers to tests whose resources are not available."""
    checks = {
        "gpu": (cuda_available, "no CUDA device"),
        "weights": (weights_available, "released weights missing (git lfs pull)"),
        "data": (data_available, "FACE1KB_DATA_ROOT not set or missing"),
    }
    cache: dict[str, bool] = {}
    for item in items:
        for name, (check, reason) in checks.items():
            if item.get_closest_marker(name) is not None:
                if name not in cache:
                    cache[name] = check()
                if not cache[name]:
                    item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture(scope="session")
def device() -> str:
    """``"cuda"`` when available, else ``"cpu"``."""
    return "cuda" if cuda_available() else "cpu"
