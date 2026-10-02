# SPDX-License-Identifier: MIT
"""Load the experiments/figures scripts as modules (they are not a package)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIGURES_DIR = REPO / "experiments" / "figures"
RESULTS = REPO / "results"


def load(name: str):
    """Import ``experiments/figures/<name>.py`` under a private module name."""
    key = f"_face1kb_figures_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, FIGURES_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod
