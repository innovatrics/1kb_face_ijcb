# SPDX-License-Identifier: MIT
"""measure_quality.py on a tiny synthetic cell, with stand-in metrics (CPU)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from face1kb import config

from ._scripts import load

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")


def _fake_metrics(device="cpu"):
    """Five cheap metrics with the ``(test, ref) -> tensor[N]`` contract."""

    def mse(t, r):
        return ((t - r) ** 2).flatten(1).mean(1)

    return {
        "psnr": lambda t, r: -10 * torch.log10(mse(t, r) + 1e-12),
        "ssim": lambda t, r: 1 - mse(t, r),
        "ms_ssim": lambda t, r: 1 - 2 * mse(t, r),
        "lpips": mse,
        "dists": lambda t, r: 2 * mse(t, r),
    }


@pytest.fixture
def cell(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path / "data")
    monkeypatch.setattr(config, "WORK_ROOT", tmp_path / "work")
    monkeypatch.setattr(config, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(config, "LAYOUT", "public")
    rng = np.random.default_rng(0)
    comp = config.compressed_dir("colorferet", 64, 1024, "webp")
    for subj in ("00001", "00002"):
        for k in range(3):
            img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
            ref = config.aligned_dir("colorferet", 64) / subj / f"{subj}_{k}.png"
            ref.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img).save(ref)
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="WEBP", quality=50)
            out = comp / subj / f"{subj}_{k}.webp"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(buf.getvalue())
    (comp / "00002" / "00002_9.webp").write_bytes(b"not a webp")  # no reference
    (comp / "00001" / "00001_1.webp").write_bytes(b"broken")  # fails to decode
    return tmp_path


def test_measure_and_aggregate(cell, monkeypatch):
    mq = load("quality", "measure_quality")
    monkeypatch.setattr(mq.quality, "make_metrics", _fake_metrics)
    shards = cell / "shards"
    args = [
        "--datasets",
        "colorferet",
        "--codecs",
        "webp,jpeg",
        "--resolutions",
        "64",
        "--budgets",
        "1024",
        "--device",
        "cpu",
        "--shard-dir",
        str(shards),
    ]
    assert mq.main(args) == 0
    per = pd.read_parquet(shards / "webp_64_1024.parquet")
    assert list(per.columns) == list(mq.quality.PER_IMAGE_COLUMNS)
    assert len(per) == 5  # 6 crops, one broken bitstream; the orphan file is skipped
    assert not (shards / "jpeg_64_1024.parquet").exists()  # no such cell
    q = pd.read_csv(config.output_dir("quality") / "quality_colorferet.csv")
    assert q.n.tolist() == [5] and q.codec.tolist() == ["webp"]
    assert q.psnr.iloc[0] == pytest.approx(per.psnr.median())
    summary = pd.read_csv(config.output_dir("quality") / "quality_summary.csv")
    assert len(summary) == 1
    # an existing shard is kept unless --overwrite
    mtime = (shards / "webp_64_1024.parquet").stat().st_mtime_ns
    assert mq.main(args) == 0
    assert (shards / "webp_64_1024.parquet").stat().st_mtime_ns == mtime


def test_aggregate_excludes_unknown_codecs(cell):
    mq = load("quality", "measure_quality")
    shards = cell / "shards"
    shards.mkdir()
    rec = {
        "image": "a",
        "subject": "s",
        "res": 112,
        "budget": 1024,
        "psnr": 30.0,
        "ssim": 0.9,
        "ms_ssim": 0.9,
        "lpips": 0.1,
        "dists": 0.1,
    }
    pd.DataFrame([{**rec, "codec": "webp"}]).to_parquet(
        shards / "webp_112_1024.parquet"
    )
    pd.DataFrame([{**rec, "codec": "neural_unknown_codec"}]).to_parquet(
        shards / "neural_unknown_codec_112_1024.parquet"
    )
    mq.aggregate("colorferet", shards, cell / "agg")
    q = pd.read_csv(cell / "agg" / "quality_colorferet.csv")
    assert q.codec.tolist() == ["webp"]


def test_legacy_layout_needs_shard_dir(monkeypatch):
    mq = load("quality", "measure_quality")
    monkeypatch.setattr(config, "LAYOUT", "legacy")
    with pytest.raises(SystemExit):
        mq.shard_dir("kk", None)
