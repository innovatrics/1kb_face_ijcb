# SPDX-License-Identifier: MIT
"""Run the helper scripts of this folder in a fresh interpreter."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def run_helper(script: str, *args: str, timeout: int = 900) -> dict:
    """Run a helper script of this folder in a fresh interpreter; parse its JSON."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO), env.get("PYTHONPATH", "")) if p
    )
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    out = subprocess.run(
        [sys.executable, str(HERE / script), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if out.returncode != 0:
        raise AssertionError(f"{script} {args} failed:\n{out.stderr[-4000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])
