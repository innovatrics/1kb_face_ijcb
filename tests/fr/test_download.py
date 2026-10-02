# SPDX-License-Identifier: MIT
"""Download / verification logic of face1kb.fr.download (file:// URLs, no network)."""

from __future__ import annotations

import hashlib
import os
import sys
import types
import zipfile

import pytest

from face1kb.fr import download
from face1kb.fr.sources import Archive, RemoteFile

PAYLOAD = b"face1kb test payload\n" * 1000


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture()
def origin(tmp_path):
    src = tmp_path / "origin" / "weights.bin"
    src.parent.mkdir()
    src.write_bytes(PAYLOAD)
    return src


def _no_leftovers(root):
    return not [p for p in root.rglob("*") if p.name.endswith(".part")]


def test_fetch_file_from_url(tmp_path, origin):
    root = tmp_path / "models"
    f = RemoteFile(
        "x/weights.bin", _sha(PAYLOAD), len(PAYLOAD), urls=(origin.as_uri(),)
    )
    assert download.check_file(f, root).endswith("missing")
    dst = download.fetch_file(f, root)
    assert dst == root / "x" / "weights.bin" and dst.read_bytes() == PAYLOAD
    assert download.check_file(f, root) is None
    assert download.fetch_file(f, root) == dst  # idempotent
    assert _no_leftovers(root)


def test_second_url_is_tried(tmp_path, origin):
    root = tmp_path / "models"
    bad = (tmp_path / "nope.bin").as_uri()
    f = RemoteFile("w.bin", _sha(PAYLOAD), len(PAYLOAD), urls=(bad, origin.as_uri()))
    assert download.fetch_file(f, root).read_bytes() == PAYLOAD


def test_hash_mismatch_leaves_nothing(tmp_path, origin):
    root = tmp_path / "models"
    f = RemoteFile("w.bin", "0" * 64, len(PAYLOAD), urls=(origin.as_uri(),))
    with pytest.raises(RuntimeError, match="sha256"):
        download.fetch_file(f, root)
    assert not (root / "w.bin").exists() and _no_leftovers(root)


def test_corrupted_existing_file_is_reported(tmp_path, origin):
    root = tmp_path / "models"
    f = RemoteFile("w.bin", _sha(PAYLOAD), len(PAYLOAD), urls=(origin.as_uri(),))
    (root).mkdir()
    (root / "w.bin").write_bytes(PAYLOAD[:-1] + b"X")
    with pytest.raises(RuntimeError, match="sha256"):
        download.ensure_file(f, root, model="toy")
    # verify=False trusts the file
    assert download.ensure_file(f, root, verify=False) == root / "w.bin"


def test_ensure_file_offline(tmp_path, origin):
    f = RemoteFile("w.bin", _sha(PAYLOAD), len(PAYLOAD), urls=(origin.as_uri(),))
    with pytest.raises(FileNotFoundError, match="fetch_models.py --only toy"):
        download.ensure_file(f, tmp_path / "models", download=False, model="toy")
    assert download.ensure_file(f, tmp_path / "models").read_bytes() == PAYLOAD


def test_archive_member(tmp_path):
    zpath = tmp_path / "pack.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("pack/other.onnx", b"other")
        z.writestr("pack/model.onnx", PAYLOAD)
    data = zpath.read_bytes()
    arch = Archive(urls=(zpath.as_uri(),), sha256=_sha(data), size=len(data))
    f = RemoteFile(
        "pack/model.onnx",
        _sha(PAYLOAD),
        len(PAYLOAD),
        archive=arch,
        member="pack/model.onnx",
    )
    root = tmp_path / "models"
    assert download.fetch_file(f, root).read_bytes() == PAYLOAD
    assert sorted(p.name for p in (root / "pack").iterdir()) == ["model.onnx"]
    assert _no_leftovers(root)


def test_gdrive_failure_asks_for_a_manual_download(tmp_path, monkeypatch):
    fake = types.ModuleType("gdown")

    def fail(**kwargs):
        raise RuntimeError("Too many users have viewed or downloaded this file")

    fake.download = fail
    monkeypatch.setitem(sys.modules, "gdown", fake)
    f = RemoteFile(
        "topofr/ckpt.pt",
        "a" * 64,
        123,
        gdrive_id="FILEID",
        manual_url="https://drive.google.com/file/d/FILEID/view",
    )
    root = tmp_path / "models"
    with pytest.raises(download.ManualDownloadRequired) as ei:
        download.fetch_file(f, root)
    msg = str(ei.value)
    assert "drive.google.com/file/d/FILEID" in msg
    assert str(root / "topofr" / "ckpt.pt") in msg and "a" * 64 in msg
    assert _no_leftovers(root)


def test_gdrive_interrupted_transfer_leaves_nothing(tmp_path, monkeypatch):
    fake = types.ModuleType("gdown")

    def cut_off(id, output, quiet):  # noqa: A002 - gdown's signature
        # gdown streams into NamedTemporaryFile(prefix=basename(output),
        # suffix=".part", dir=dirname(output)) and keeps it when the stream breaks.
        with open(f"{output}{'x' * 8}.part", "wb") as fh:
            fh.write(PAYLOAD[:100])
        raise ConnectionError("connection reset by peer")

    fake.download = cut_off
    monkeypatch.setitem(sys.modules, "gdown", fake)
    f = RemoteFile("t/c.pt", _sha(PAYLOAD), len(PAYLOAD), gdrive_id="ID")
    with pytest.raises(download.ManualDownloadRequired, match="connection reset"):
        download.fetch_file(f, tmp_path)
    assert not (tmp_path / "t" / "c.pt").exists() and _no_leftovers(tmp_path)


def test_gdrive_download(tmp_path, monkeypatch):
    fake = types.ModuleType("gdown")

    def ok(id, output, quiet):  # noqa: A002 - gdown's signature
        with open(output, "wb") as fh:
            fh.write(PAYLOAD)
        return output

    fake.download = ok
    monkeypatch.setitem(sys.modules, "gdown", fake)
    f = RemoteFile("t/c.pt", _sha(PAYLOAD), len(PAYLOAD), gdrive_id="ID")
    assert download.fetch_file(f, tmp_path).read_bytes() == PAYLOAD


def test_verification_is_cached_per_process(tmp_path, origin, monkeypatch):
    f = RemoteFile("w.bin", _sha(PAYLOAD), len(PAYLOAD), urls=(origin.as_uri(),))
    download.fetch_file(f, tmp_path)
    calls = []
    real = download.sha256_file
    monkeypatch.setattr(download, "sha256_file", lambda p: calls.append(p) or real(p))
    assert download.check_file(f, tmp_path) is None
    assert calls == []  # verified when it was fetched
    dst = tmp_path / "w.bin"
    dst.write_bytes(PAYLOAD)
    st = dst.stat()  # force a new mtime (timestamps can be coarse) -> hashed again
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert download.check_file(f, tmp_path) is None
    assert len(calls) == 1
