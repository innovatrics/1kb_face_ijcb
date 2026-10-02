# SPDX-License-Identifier: MIT
"""Load the experiments/accuracy and experiments/embed scripts as modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIRS = {
    "accuracy": REPO / "experiments" / "accuracy",
    "embed": REPO / "experiments" / "embed",
}


def load(name: str, area: str = "accuracy"):
    """Import ``experiments/<area>/<name>.py`` under a private module name."""
    key = f"_face1kb_{area}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, SCRIPT_DIRS[area] / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod
