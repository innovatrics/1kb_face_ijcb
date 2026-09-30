# SPDX-License-Identifier: MIT
"""Command-line interface (python -m face1kb.codec)."""

from __future__ import annotations

import json
import warnings

import pytest

from face1kb.codec import container as C
from face1kb.codec.cli import main


def test_info(tmp_path, capsys):
    p = tmp_path / "x.f1k"
    p.write_bytes(C.pack(C.VARIANT_ACCURATE, 12, 168, b"s" * 8, b"y" * 30, b"z" * 9))
    assert main(["info", str(p)]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["variant"] == "accurate" and rec["res"] == 168
    assert rec["side_bytes"] == 8 and rec["raw_geometry"] is True
    assert rec["bytes"] == 3 + 8 + 4 + 30 + 9 + 4


@pytest.mark.weights
def test_encode_decode_round_trip(tmp_path, capsys, device):
    pytest.importorskip("torch")
    from PIL import Image

    from ._synthetic import smooth_face

    src = tmp_path / "face.png"
    Image.fromarray(smooth_face(112)).save(src)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert main(["encode", str(src), "--budget", "512", "--device", device]) == 0
        enc = json.loads(capsys.readouterr().out)
        f1k = tmp_path / "face.f1k"
        assert enc["output"] == str(f1k) and f1k.stat().st_size <= 512
        out = tmp_path / "back.png"
        assert main(["decode", str(f1k), str(out), "--device", device]) == 0
    img = Image.open(out)
    assert img.size == (112, 112) and img.mode == "RGB"


def test_info_on_truncated_container_reports_error(tmp_path, capsys):
    p = tmp_path / "trunc.f1k"
    p.write_bytes(C.pack(C.VARIANT_FAST, 1, 112, b"", b"y" * 40, b"z" * 8)[:20])
    assert main(["info", str(p)]) == 1
    assert "error: container truncated" in capsys.readouterr().err


def test_decode_refuses_to_overwrite_its_input(tmp_path, capsys):
    p = tmp_path / "x.png"
    p.write_bytes(C.pack(C.VARIANT_IDENTITY_ONLY, 0, 112))
    assert main(["decode", str(p), str(p)]) == 1
    assert "error: refusing to overwrite" in capsys.readouterr().err


def test_output_path_may_follow_the_options(tmp_path, capsys):
    f1k = tmp_path / "io.f1k"
    f1k.write_bytes(C.pack(C.VARIANT_IDENTITY_ONLY, 0, 112))  # decodes without a model
    out = tmp_path / "late.png"
    assert main(["decode", str(f1k), "--device", "cpu", str(out)]) == 0
    assert json.loads(capsys.readouterr().out)["output"] == str(out)
    assert out.is_file()
    with pytest.raises(SystemExit) as exc:
        main(["decode", str(f1k), str(out), "--device", "cpu", "extra.png"])
    assert exc.value.code == 2
    assert "unrecognized arguments: extra.png" in capsys.readouterr().err


def test_decode_default_output_does_not_replace_the_source(tmp_path, capsys):
    pytest.importorskip("torch")
    from PIL import Image

    src = tmp_path / "face.png"
    Image.new("RGB", (112, 112), (200, 150, 120)).save(src)
    before = src.read_bytes()
    f1k = tmp_path / "face.f1k"
    f1k.write_bytes(C.pack(C.VARIANT_IDENTITY_ONLY, 0, 112))  # decodes without a model
    assert main(["decode", str(f1k), "--device", "cpu"]) == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["output"] == str(tmp_path / "face_decoded.png")
    assert {"load_ms", "decode_ms"} <= set(rec)
    assert src.read_bytes() == before
    assert Image.open(tmp_path / "face_decoded.png").size == (112, 112)


@pytest.mark.weights
def test_encode_warns_on_identity_only(tmp_path, capsys, device):
    pytest.importorskip("torch")
    import numpy as np
    from PIL import Image

    src = tmp_path / "noise.png"
    rng = np.random.default_rng(0)
    Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)).save(src)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rc = main(["encode", str(src), "--budget", "512", "--device", device])
    assert rc == 0
    cap = capsys.readouterr()
    rec = json.loads(cap.out)
    assert rec["identity_only"] and rec["bytes"] == 3
    assert "identity-only container" in cap.err


def test_missing_or_unwritable_files_report_errors(tmp_path, capsys):
    missing = tmp_path / "missing.f1k"
    for argv in (["info", str(missing)], ["decode", str(missing)]):
        assert main(argv) == 1
        assert "error:" in capsys.readouterr().err
    f1k = tmp_path / "io.f1k"
    f1k.write_bytes(C.pack(C.VARIANT_IDENTITY_ONLY, 0, 112))
    (tmp_path / "out.png").mkdir()
    assert main(["decode", str(f1k), str(tmp_path / "out.png")]) == 1
    assert "error:" in capsys.readouterr().err
    (tmp_path / "bad.png").write_bytes(b"not an image")
    pytest.importorskip("torch")
    for name in ("missing.png", "bad.png"):
        assert main(["encode", str(tmp_path / name), "--device", "cpu"]) == 1
        assert "error:" in capsys.readouterr().err


class _StubCodec:
    """Stand-in for a loaded codec whose container overshoots by the trailer."""

    def encode(self, img, budget, paper_compat, return_info, overflow):
        n = budget + 3
        return b"x" * n, {
            "fitted": True,
            "over_budget": False,
            "identity_only": False,
            "within_budget": False,
            "rate_index": 5,
            "gain": 0.5,
            "bytes": n,
        }


def test_encode_warns_whenever_the_file_exceeds_the_budget(
    tmp_path, capsys, monkeypatch
):
    pytest.importorskip("torch")
    from PIL import Image

    import face1kb

    monkeypatch.setattr(face1kb, "load", lambda *a, **k: _StubCodec())
    src = tmp_path / "f.png"
    Image.new("RGB", (96, 96), (120, 100, 90)).save(src)
    argv = ["encode", str(src), "--budget", "512", "--paper-compat", "--device", "cpu"]
    assert main(argv) == 0
    cap = capsys.readouterr()
    rec = json.loads(cap.out)
    assert rec["fitted"] and not rec["within_budget"] and rec["bytes"] == 515
    assert "exceeds the 512 B budget" in cap.err and "geometry trailer" in cap.err
    # An output path that is a directory is reported, not a traceback.
    (tmp_path / "o.f1k").mkdir()
    assert main(["encode", str(src), str(tmp_path / "o.f1k"), "--device", "cpu"]) == 1
    assert "error:" in capsys.readouterr().err


def test_identity_only_decode_does_not_import_torch(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    f1k = tmp_path / "io.f1k"
    f1k.write_bytes(C.pack(C.VARIANT_IDENTITY_ONLY, 0, 96))
    code = (
        "import sys; from face1kb.codec.cli import main; rc = main(['decode', "
        "sys.argv[1]]); assert 'torch' not in sys.modules, 'torch imported'; "
        "sys.exit(rc)"
    )
    path = os.pathsep.join(filter(None, [str(repo), os.environ.get("PYTHONPATH")]))
    env = dict(os.environ, PYTHONPATH=path, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(
        [sys.executable, "-c", code, str(f1k)], capture_output=True, text=True, env=env
    )
    assert r.returncode == 0, r.stderr
    rec = json.loads(r.stdout)
    assert rec["variant"] == "identity-only" and rec["load_ms"] == 0.0
