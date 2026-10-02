# SPDX-License-Identifier: MIT
"""experiments/compress/decode_cache.py: per-file SIGALRM timeout."""

from __future__ import annotations

import signal
import time

import numpy as np
import pytest

import face1kb.baselines as baselines

from ._scripts import load

decode_cache = load("decode_cache")

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"), reason="SIGALRM is Unix-only"
)


def test_hanging_decode_times_out_and_run_continues(tmp_path, monkeypatch):
    hang, ok = tmp_path / "hang.bin", tmp_path / "ok.bin"
    hang.write_bytes(b"x")
    ok.write_bytes(b"y")

    def fake_decode(src, device=None):
        if src.name == "hang.bin":
            time.sleep(30)
        return np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr(baselines, "decode_file", fake_decode)
    monkeypatch.setattr(
        baselines, "decoded_cache_path", lambda p: p.with_suffix(".png")
    )
    t0 = time.monotonic()
    done, failed = decode_cache._worker(None, [str(hang), str(ok)], {}, timeout_s=1)
    assert time.monotonic() - t0 < 10
    assert (done, failed) == (1, 1)
    assert (tmp_path / "ok.png").exists()
    assert not (tmp_path / "hang.png").exists()
    assert signal.alarm(0) == 0  # no alarm left pending
