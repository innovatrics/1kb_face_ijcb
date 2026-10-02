# SPDX-License-Identifier: MIT
"""experiments/compress/compress.py: task grid, resume, failure rows, manifests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from face1kb import config
from face1kb.baselines.verify import synthetic_image

from ._scripts import load

compress = load("compress")


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """Public-layout data/work roots with two 112 px crops of two subjects."""
    data, work = tmp_path / "data", tmp_path / "work"
    monkeypatch.setattr(config, "LAYOUT", "public")
    monkeypatch.setattr(config, "DATA_ROOT", data)
    monkeypatch.setattr(config, "WORK_ROOT", work)
    rows = []
    for i, (subj, stem) in enumerate([("00001", "a_fa"), ("00002", "b_fb")]):
        d = config.aligned_dir("colorferet", 112) / subj
        d.mkdir(parents=True)
        Image.fromarray(synthetic_image(112, seed=i)).save(d / f"{stem}.png")
        rows.append({"id": i, "rel_path": f"{subj}/{stem}.png", "subject": subj})
    pd.DataFrame(rows).to_csv(config.index_csv("colorferet"), index=False)
    return tmp_path


def test_parse_codecs_groups_and_order():
    assert compress.parse_codecs("cpu") == list(compress.CPU_GROUP)
    assert compress.parse_codecs("webp,jpeg,webp") == ["webp", "jpeg"]
    assert compress.parse_codecs("ours") == ["ours_fast", "ours_accurate"]
    assert len(compress.parse_codecs("all")) == 12
    with pytest.raises(SystemExit):
        compress.parse_codecs("neural_unknown")


def test_select_rows_shard_offset_limit():
    rows = list(range(20))
    assert compress.select_rows(rows, (0, 1), 0, 0) == rows
    assert compress.select_rows(rows, (1, 4), 0, 0) == [1, 5, 9, 13, 17]
    assert compress.select_rows(rows, (1, 4), 1, 2) == [5, 9]
    assert compress.select_rows(rows, (0, 1), 0, 3) == [0, 1, 2]
    with pytest.raises(ValueError):
        compress.select_rows(rows, (4, 4), 0, 0)


def test_manifest_tag():
    tag = compress.manifest_tag(["jpeg", "webp"], [112, 224], [1024, 512])
    assert tag == "jpeg_webp_112_224_1024_512"
    assert compress.manifest_tag(["ours_fast"], [112], [768], "_A1", (2, 7)) == (
        "ours_fast_112_A1_768_sh2of7"
    )
    assert compress.manifest_tag(["jpeg"], [112], [1024], offset=5, limit=3) == (
        "jpeg_112_1024_o5n3"
    )


def test_paths_follow_config(tree):
    rows = [{"rel_path": "00001/a_fa.png", "subject": 1}]
    (task,) = compress.build_tasks(
        "colorferet", rows, ["jpeg_fzt"], [112], [512], "_A1"
    )
    assert task[2] == "00001"  # subject is the crop folder, not the index value
    assert compress.output_path(task) == (
        config.compressed_dir("colorferet", 112, 512, "jpeg_fzt", "_A1")
        / "00001"
        / "a_fa.fzt"
    )
    assert compress.source_path(task) == (
        config.aligned_dir("colorferet", 112, "_A1") / "00001" / "a_fa.png"
    )


def _opts(**kw):
    return {
        "device": "cpu",
        "overwrite": False,
        "compressai_deterministic": "auto",
        **kw,
    }


def test_existing_and_missing_rows(tree):
    rows = [{"rel_path": "00001/a_fa.png"}, {"rel_path": "00009/none.png"}]
    t_ok, t_missing = compress.build_tasks("colorferet", rows, ["jpeg"], [112], [1024])
    out = compress.output_path(t_ok)
    out.parent.mkdir(parents=True)
    out.write_bytes(b"x" * 1100)
    r = compress.process_task(t_ok, _opts())
    assert r["status"] == "existing" and r["bytes"] == 1100 and not r["fitted"]
    assert out.read_bytes() == b"x" * 1100  # never re-encoded without --overwrite
    r = compress.process_task(t_missing, _opts())
    assert r["status"] == "missing" and r["bytes"] == -1
    assert not compress.output_path(t_missing).exists()


def test_jpegai_failure_writes_no_file(tree, monkeypatch):
    from face1kb.baselines.jpeg_ai import JpegAIError

    def boom(*_a, **_k):
        raise JpegAIError("no plausible stream")

    monkeypatch.setattr(compress, "encode_crop", boom)
    (task,) = compress.build_tasks(
        "colorferet", [{"rel_path": "00001/a_fa.png"}], ["jpeg_ai"], [112], [1024]
    )
    r = compress.process_task(task, _opts())
    assert r["status"] == "failed" and "no plausible" in r["error"]
    assert not compress.output_path(task).exists()


def test_fit_is_judged_on_disk(tree, monkeypatch):
    # a CompressAI-like result: the search fitted on entropy bytes, the file does not
    monkeypatch.setattr(
        compress,
        "encode_crop",
        lambda *a: (b"y" * 1030, {"setting": 3, "size": 1000, "fitted": True}),
    )
    (task,) = compress.build_tasks(
        "colorferet",
        [{"rel_path": "00002/b_fb.png"}],
        ["neural_bmshj2018"],
        [112],
        [1024],
    )
    r = compress.process_task(task, _opts())
    assert r["status"] == "encoded"
    assert (r["bytes"], r["fitted"]) == (1030, False)
    assert (r["search_bytes"], r["search_fitted"]) == (1000, True)
    assert compress.output_path(task).stat().st_size == 1030
    assert not list(compress.output_path(task).parent.glob("*.part*"))


def test_compressai_deterministic_rule(tree, monkeypatch):
    seen = []
    monkeypatch.setattr(
        compress,
        "encode_crop",
        lambda c, s, b, d, det: (
            seen.append((b, det)) or b"z",
            {
                "setting": 1,
                "size": 1,
                "fitted": True,
            },
        ),
    )
    tasks = compress.build_tasks(
        "colorferet",
        [{"rel_path": "00001/a_fa.png"}],
        ["neural_mbt2018_mean"],
        [112],
        [1024, 768, 960],
    )
    for t in tasks:
        compress.process_task(t, _opts(overwrite=True))
    assert seen == [(1024, False), (768, True), (960, True)]


def test_main_cpu_grid_and_resume(tree):
    argv = [
        "--dataset",
        "colorferet",
        "--codecs",
        "jpeg,webp,jpeg_fzt",
        "--resolutions",
        "112",
        "--budgets",
        "1024,512",
        "--workers",
        "1",
    ]
    assert compress.main(argv) == 0
    man = (
        compress.manifest_dir("colorferet") / "jpeg_webp_jpeg_fzt_112_1024_512.parquet"
    )
    df = pd.read_parquet(man)
    assert list(df.columns) == list(compress.COLUMNS)
    assert len(df) == 2 * 3 * 2 and set(df.status) == {"encoded"}
    for r in df.itertuples():
        f = config.compressed_dir("colorferet", 112, r.budget, r.codec) / Path(
            r.rel_path
        ).with_suffix({"jpeg": ".jpg", "webp": ".webp", "jpeg_fzt": ".fzt"}[r.codec])
        assert f.stat().st_size == r.bytes
        assert r.fitted == (r.bytes <= r.budget)
    # same bytes as the library call
    from face1kb.baselines import get_codec

    with Image.open(config.aligned_dir("colorferet", 112) / "00001" / "a_fa.png") as im:
        data, _ = get_codec("webp").encode_to_budget(im.convert("RGB"), 512)
    stored = (
        config.compressed_dir("colorferet", 112, 512, "webp") / "00001" / "a_fa.webp"
    )
    assert stored.read_bytes() == data
    # resume: nothing is re-encoded
    assert compress.main(argv) == 0
    assert set(pd.read_parquet(man).status) == {"existing"}


def test_legacy_store_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LAYOUT", "legacy")
    monkeypatch.setattr(config, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(config, "WORK_ROOT", tmp_path)
    with pytest.raises(SystemExit):
        compress._guard_legacy_store()


def test_decode_cache_plan(tree):
    decode_cache = load("decode_cache")
    from face1kb.baselines import decoded_cache_path

    rows = [{"rel_path": "00001/a_fa.png"}, {"rel_path": "00002/b_fb.png"}]
    for t in compress.build_tasks("colorferet", rows, ["jpeg"], [112], [1024]):
        compress.process_task(t, _opts())
    cells = decode_cache.plan(["colorferet"], ["jpeg", "jpeg_ai"], [112], [1024])
    assert len(cells) == 1
    ds, res, budget, codec, files, todo = cells[0]
    assert (ds, res, budget, codec, len(files), len(todo)) == (
        "colorferet",
        112,
        1024,
        "jpeg",
        2,
        2,
    )
    dst = decoded_cache_path(files[0])
    assert (
        dst
        == config.decoded_dir("colorferet", 112, 1024, "jpeg") / "00001" / "a_fa.png"
    )
    assert decode_cache._worker(None, [str(f) for f in files], {}) == (2, 0)
    arr = np.asarray(Image.open(dst).convert("RGB"))
    from face1kb.baselines import decode_file

    assert np.array_equal(arr, decode_file(files[0]))
    assert decode_cache.plan(["colorferet"], ["jpeg"], [112], [1024])[0][5] == []
