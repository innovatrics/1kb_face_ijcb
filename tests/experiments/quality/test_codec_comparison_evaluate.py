# SPDX-License-Identifier: MIT
"""CPU checks of the codec-comparison driver: sampling and codec selection."""

from __future__ import annotations

import random

import numpy as np
import pytest

from ._scripts import load

ev = load("codec_comparison", "evaluate")


def _crops(root, n_subjects=3, per_subject=4):
    from PIL import Image

    for s in range(n_subjects):
        d = root / "aligned_112" / f"s{s:02d}"
        d.mkdir(parents=True)
        for k in range(per_subject):
            Image.fromarray(np.zeros((4, 4, 3), np.uint8)).save(d / f"img{k}.png")


def test_sample_is_seeded_over_sorted_files(tmp_path, monkeypatch):
    _crops(tmp_path)
    monkeypatch.setattr(
        ev.config, "aligned_dir", lambda ds, res: tmp_path / f"aligned_{res}"
    )
    files = sorted((tmp_path / "aligned_112").glob("*/*.png"))
    got = ev.sample_crops("kk", 112, 5, 0)
    assert got == random.Random(0).sample(files, 5)
    assert len(ev.sample_crops("kk", 112, 100, 0)) == len(files)


def test_empty_sample_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr(ev.config, "aligned_dir", lambda ds, res: tmp_path / "none")
    with pytest.raises(SystemExit, match="no aligned crops"):
        ev.sample_crops("kk", 112, 64, 0)


def test_codec_selection():
    enc = ev.Encoders.__new__(ev.Encoders)
    enc.codecs = None
    assert all(enc._on(c) for c in ev.CODECS)
    enc.codecs = {"webp", "jpeg_ai"}
    assert enc._on("webp") and enc._on("jpeg_ai") and not enc._on("jpeg")
    assert len(ev.CODECS) == len(set(ev.CODECS)) == 10


def test_unknown_codec_is_rejected(monkeypatch):
    monkeypatch.setattr(ev.fr, "load", lambda *a, **k: object())
    monkeypatch.setattr(ev.jpeg_ai, "available", lambda: False)
    args = ev.argparse.Namespace(
        id_onnx=None,
        id_model="x",
        device="cpu",
        no_jpegai=True,
        codecs="webp,not_a_codec",
        ours="",
    )
    with pytest.raises(SystemExit, match="not_a_codec"):
        ev.run(args)


def _row(ds, codec, budget, v):
    return {"dataset": ds, "codec": codec, "budget": budget, "id_cos": v}


def test_merge_replaces_only_recomputed_rows():
    old = [
        _row("kk", "jpeg", 1024, 0.1),
        _row("kk", "webp", 1024, 0.2),
        _row("kk", "webp", 512, 0.3),
    ]
    new = [_row("kk", "webp", 1024, 0.9), _row("colorferet", "webp", 1024, 0.8)]
    got = ev.merge_rows(old, new)
    assert got == [
        _row("kk", "jpeg", 1024, 0.1),
        _row("kk", "webp", 1024, 0.9),
        _row("kk", "webp", 512, 0.3),
        _row("colorferet", "webp", 1024, 0.8),
    ]
    assert ev.merge_rows([], new) == new


def test_main_merges_with_codecs_and_rewrites_without(tmp_path, monkeypatch):
    import json

    out = tmp_path / "comparison.json"
    full = [_row("kk", "jpeg", 1024, 0.1), _row("kk", "webp", 1024, 0.2)]
    out.write_text(json.dumps(full))
    monkeypatch.setattr(ev, "run", lambda args: [_row("kk", "webp", 1024, 0.9)])
    ev.main(["--codecs", "webp", "--out-dir", str(tmp_path)])
    assert json.loads(out.read_text()) == [
        _row("kk", "jpeg", 1024, 0.1),
        _row("kk", "webp", 1024, 0.9),
    ]
    ev.main(["--out-dir", str(tmp_path)])
    assert json.loads(out.read_text()) == [_row("kk", "webp", 1024, 0.9)]
