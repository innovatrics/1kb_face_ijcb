# SPDX-License-Identifier: MIT
"""compute_accuracy / compute_significance / idcos_tail on a synthetic dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from face1kb import config
from face1kb.eval.significance import SIGNIFICANCE_COLUMNS
from face1kb.eval.verification import METRICS_COLUMNS

from ._scripts import load

MODEL = "arcface_antelopev2"
N_SUBJ, PER_SUBJ = 10, 5
#: tag -> noise level of the synthetic embeddings
TAGS = {
    "aligned_112": 0.4,
    "webp_112_512": 0.9,
    "jpeg_ai_112_512": 0.7,
    "ours_fast_112_512": 0.6,
    "avif_112_512": 1.2,
    "ours_accurate_112_512": 0.5,
}


def _write(dataset: str, rng) -> None:
    subj = np.repeat(np.arange(1, N_SUBJ + 1), PER_SUBJ)
    shot = np.tile(range(PER_SUBJ), N_SUBJ)
    rel = [f"{s:05d}/{s:05d}_{k}.png" for s, k in zip(subj, shot)]
    config.dataset_dir(dataset).mkdir(parents=True)
    pd.DataFrame({"id": np.arange(subj.size), "rel_path": rel, "subject": subj}).to_csv(
        config.index_csv(dataset), index=False
    )
    i1, i2 = np.triu_indices(subj.size, 1)
    pd.DataFrame(
        {
            "idx1": i1.astype(np.int32),
            "idx2": i2.astype(np.int32),
            "label": (subj[i1] == subj[i2]).astype(np.int8),
        }
    ).to_parquet(config.pairs_parquet(dataset), index=False)
    centers = rng.normal(size=(N_SUBJ, 512))
    config.embeddings_dir(dataset, MODEL).mkdir(parents=True)
    for tag, noise in TAGS.items():
        e = centers[subj - 1] + noise * rng.normal(size=(subj.size, 512))
        e = e.astype(np.float32)
        if tag == "webp_112_512":
            e[7] = np.nan
        np.save(config.embeddings_path(dataset, MODEL, tag), e)


@pytest.fixture
def synth(tmp_path, monkeypatch):
    roots = {"DATA_ROOT": "data", "WORK_ROOT": "work", "OUTPUT_ROOT": "out"}
    for name, sub in roots.items():
        monkeypatch.setattr(config, name, tmp_path / sub)
    monkeypatch.setattr(config, "LAYOUT", "public")
    rng = np.random.default_rng(0)
    for ds in config.DATASETS:
        _write(ds, rng)
    return tmp_path


@pytest.mark.filterwarnings("ignore:All-NaN slice")
def test_compute_accuracy_writes_metrics(synth):
    m = load("compute_accuracy")
    out = synth / "acc"
    args = ["--device", "cpu", "--bootstrap", "5", "--sample-nonmated", "300"]
    assert m.main(args + ["--out-dir", str(out)]) == 0
    df = pd.read_csv(out / "metrics.csv")
    assert list(df.columns) == list(METRICS_COLUMNS)
    assert len(df) == 2 * len(TAGS)
    assert list(df.dataset.unique()) == ["colorferet", "kk"]
    al = df[df.tag == "aligned_112"]
    assert (al.delta_eer == 0).all()
    assert df.eer_lo.notna().all()  # arcface is a bootstrap model, suffix empty
    assert df.median_bytes.isna().all()
    # merging the per-dataset files gives the same text as the combined write
    text = (out / "metrics.csv").read_text()
    assert m.main(["--merge", "--out-dir", str(out)]) == 0
    assert (out / "metrics.csv").read_text() == text


@pytest.mark.filterwarnings("ignore:All-NaN slice")
def test_compute_accuracy_tags_keep_baseline(synth):
    m = load("compute_accuracy")
    assert m.with_baselines(["webp_112_512", "aligned_96_A1", "jpeg_xl_168_1024"]) == [
        "webp_112_512",
        "aligned_96_A1",
        "jpeg_xl_168_1024",
        "aligned_112",
        "aligned_96",
        "aligned_168",
    ]
    out = synth / "acc_tags"
    args = ["--device", "cpu", "--bootstrap", "0", "--sample-nonmated", "300"]
    args += ["--datasets", "colorferet", "--tags", "webp_112_512,ours_fast_112_512"]
    assert m.main(args + ["--out-dir", str(out)]) == 0
    df = pd.read_csv(out / "metrics.csv")
    assert set(df.tag) == {"aligned_112", "webp_112_512", "ours_fast_112_512"}
    assert df.eer_aligned.notna().all()
    full = synth / "acc_full"
    args_full = ["--device", "cpu", "--bootstrap", "0", "--sample-nonmated", "300"]
    assert m.main(args_full + ["--datasets", "colorferet", "--out-dir", str(full)]) == 0
    ref = pd.read_csv(full / "metrics.csv").set_index("tag")
    got = df.set_index("tag")
    pd.testing.assert_frame_equal(got, ref.loc[got.index])


def test_compute_accuracy_refuses_to_shrink(synth, tmp_path):
    m = load("compute_accuracy")
    out = tmp_path / "acc2"
    out.mkdir()
    pd.DataFrame({"model": ["arcface_antelopev2", "lvface_l"]}).to_csv(
        out / "metrics.csv", index=False
    )
    args = ["--device", "cpu", "--bootstrap", "0", "--sample-nonmated", "300"]
    assert m.main(args + ["--out-dir", str(out)]) == 1
    assert m.main(args + ["--out-dir", str(out), "--allow-shrink"]) == 0


def test_compute_significance_per_cell(synth):
    m = load("compute_significance")
    out = synth / "sig"
    args = ["--models", MODEL, "--res", "112", "--budgets", "512"]
    args += ["--sample-nonmated", "300", "--out-dir", str(out)]
    assert m.main(args) == 0
    for ds in config.DATASETS:
        df = pd.read_csv(out / f"significance_{ds}.csv")
        assert list(df.columns) == list(SIGNIFICANCE_COLUMNS)
        assert len(df) == len(TAGS) * (len(TAGS) - 1) // 2
        assert set(df.codec_a) | set(df.codec_b) == {
            "aligned",
            "webp",
            "jpeg_ai",
            "ours_fast",
            "avif",
            "ours_accurate",
        }
        assert (df.p_adj >= df.mcnemar_p - 1e-15).all()
    merged = synth / "merged.csv"
    assert (
        m.main(
            [
                "--merge",
                str(merged),
                str(out / "significance_colorferet.csv"),
                str(out / "significance_kk.csv"),
            ]
        )
        == 0
    )
    assert len(pd.read_csv(merged)) == 2 * len(TAGS) * (len(TAGS) - 1) // 2


def test_significance_merge_keeps_text(tmp_path):
    """Merging write_significance_csv shards reproduces their text byte for byte."""
    from face1kb.eval.significance import write_significance_csv

    m = load("compute_significance")
    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame(
        {
            c: (rng.random(n) * 10.0 ** rng.integers(-60, 0, n))
            if c not in ("codec_a", "codec_b", "model", "dataset")
            else [f"{c}{i % 3}" for i in range(n)]
            for c in SIGNIFICANCE_COLUMNS
        }
    )
    shard_a, shard_b = tmp_path / "a.csv", tmp_path / "b.csv"
    write_significance_csv([frame.iloc[:25]], shard_a)
    write_significance_csv([frame.iloc[25:]], shard_b)
    single = tmp_path / "single.csv"
    assert m.main(["--merge", str(single), str(shard_a)]) == 0
    assert single.read_bytes() == shard_a.read_bytes()
    both = tmp_path / "both.csv"
    assert m.main(["--merge", str(both), str(shard_a), str(shard_b)]) == 0
    head, *rows_a = shard_a.read_text().splitlines(keepends=True)
    _, *rows_b = shard_b.read_text().splitlines(keepends=True)
    assert both.read_text() == head + "".join(rows_a + rows_b)


def test_idcos_tail_compute(synth):
    m = load("idcos_tail")
    df = m.compute(512)
    assert list(df.columns) == ["dataset", "codec", "budget", "n", "p5", "median"]
    assert len(df) == 2 * len(m.CODECS)
    web = df[(df.dataset == "kk") & (df.codec == "webp")].iloc[0]
    assert web.n == N_SUBJ * PER_SUBJ - 1  # the NaN row is dropped
    assert web.p5 <= web["median"]
    tex = m.table(df)
    assert tex.count("\\cellcolor{green!25}") == 4  # one best per column
