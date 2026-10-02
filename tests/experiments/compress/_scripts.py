# SPDX-License-Identifier: MIT
"""Load the experiments/compress and experiments/speed scripts as modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIRS = {
    "compress": REPO / "experiments" / "compress",
    "speed": REPO / "experiments" / "speed",
}
#: Published tables the generators reproduce (copies of the arXiv v1 sources).
EXPECTED = Path(__file__).resolve().parent / "expected"


def load(name: str, area: str = "compress"):
    """Import ``experiments/<area>/<name>.py`` under a private module name."""
    key = f"_face1kb_{area}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, SCRIPT_DIRS[area] / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod
