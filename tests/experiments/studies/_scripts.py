# SPDX-License-Identifier: MIT
"""Load the experiment scripts of the study areas as modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
AREAS = ("recompression", "preprocessing", "resolution", "difficulty")


def script_path(area: str, name: str) -> Path:
    """Path of ``experiments/<area>/<name>.py``."""
    return REPO / "experiments" / area / f"{name}.py"


def load(area: str, name: str):
    """Import ``experiments/<area>/<name>.py`` under a private module name."""
    key = f"_face1kb_studies_{area}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, script_path(area, name))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod
