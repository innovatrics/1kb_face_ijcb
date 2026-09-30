# SPDX-License-Identifier: MIT
"""Load the quality, fairness and codec-comparison experiment scripts as modules."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
#: Published sources of the tables the generators reproduce (arXiv v1): the table
#: files, plus the tabulars typed inline in the fairness section.
EXPECTED = Path(__file__).resolve().parent / "expected"


def load(area: str, name: str):
    """Import ``experiments/<area>/<name>.py`` under a private module name."""
    key = f"_face1kb_c4_{area}_{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = REPO / "experiments" / area / f"{name}.py"
    spec = importlib.util.spec_from_file_location(key, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[key] = mod
    spec.loader.exec_module(mod)
    return mod
