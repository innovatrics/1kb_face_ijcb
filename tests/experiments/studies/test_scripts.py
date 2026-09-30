# SPDX-License-Identifier: MIT
"""Hygiene of the study scripts: headers, imports, paths, command lines."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys

import pytest

from ._scripts import AREAS, REPO

SCRIPTS = sorted(p for a in AREAS for p in (REPO / "experiments" / a).glob("*.py"))
RUN_SH = [REPO / "experiments" / a / "run.sh" for a in AREAS]
#: Text that must not appear in the public study code: absolute host paths, the
#: excluded CompressAI model and internal model names (assembled from parts so that
#: this file itself does not contain them).
_PARTS = (
    ("/m", "nt/"),
    ("/s", "rv/"),
    ("work", "place"),
    ("chen", "g2020"),
    ("inno", "[-_]bal"),
    ("Inno", "Embedder"),
    ("inno", "_eval"),
)
FORBIDDEN = re.compile("|".join(a + b for a, b in _PARTS))


def test_every_area_has_scripts_and_run_sh():
    assert len(SCRIPTS) >= 11
    for p in RUN_SH:
        assert p.is_file(), p
        text = p.read_text()
        assert text.startswith("#!/usr/bin/env bash\n# SPDX-License-Identifier: MIT")
        assert "GPU" in text and "CPU" in text


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_header_imports_and_paths(path):
    text = path.read_text()
    assert text.startswith("# SPDX-License-Identifier: MIT\n")
    assert not FORBIDDEN.search(text), FORBIDDEN.search(text).group()
    tree = ast.parse(text)
    assert ast.get_docstring(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            assert top != "inno" and node.level == 0
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] != "inno" for a in node.names)
    assert "argparse" in text and "def main(" in text


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_help(path):
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    out = subprocess.run(
        [sys.executable, str(path), "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "usage:" in out.stdout
