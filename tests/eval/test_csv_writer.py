# SPDX-License-Identifier: MIT
"""Tests of the merged-CSV writers (write_metrics_csv / write_significance_csv)."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from face1kb.eval import significance as S
from face1kb.eval import verification as V


def _rows(dataset: str, rng) -> list[dict]:
    rows = []
    for model in ("m1", "m2"):
        for tag, kind, codec, budget in (
            ("aligned_112", "aligned", "", 0),
            ("webp_112_1024", "compressed", "webp", 1024),
            ("jpeg_112_512", "compressed", "jpeg", 512),
        ):
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "tag": tag,
                    "kind": kind,
                    "res": 112,
                    "suffix": "",
                    "codec": codec,
                    "budget": budget,
                    "n_pos": 95839,
                    "n_neg": 5_000_000,
                    "eer": float(rng.integers(1, 95839)) / 95839 / 3,
                }
            )
    return rows


def _published_flow(frames) -> str:
    """Per-part CSV files, read back with the default parser, concatenated."""
    parts = []
    for f in frames:
        buf = io.StringIO()
        f.to_csv(buf, index=False)
        buf.seek(0)
        parts.append(pd.read_csv(buf))
    out = io.StringIO()
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    return out.getvalue()


def test_write_metrics_csv_matches_published_flow(tmp_path):
    rng = np.random.default_rng(0)
    frames = [V.metrics_frame(_rows(ds, rng)) for ds in ("colorferet", "kk")]
    path = tmp_path / "metrics.csv"
    V.write_metrics_csv(frames, path)
    assert path.read_text() == _published_flow(frames)
    back = pd.read_csv(path)
    assert list(back.columns[: len(V.METRICS_COLUMNS)]) == list(V.METRICS_COLUMNS)
    assert len(back) == sum(len(f) for f in frames)


def test_write_significance_csv_matches_published_flow(tmp_path):
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {c: np.nan for c in S.SIGNIFICANCE_COLUMNS}, index=range(6)
    ).astype(object)
    df["dataset"], df["model"], df["res"], df["budget"] = "kk", "m", 112, 1024
    df["mcnemar_p"] = rng.random(6) / 7
    path = tmp_path / "significance_kk.csv"
    S.write_significance_csv([df.iloc[:3], df.iloc[3:]], path)
    assert path.read_text() == _published_flow([df.iloc[:3], df.iloc[3:]])


def test_write_merged_csv_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        V.write_merged_csv([], tmp_path / "x.csv")
