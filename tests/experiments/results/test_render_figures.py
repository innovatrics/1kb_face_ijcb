# SPDX-License-Identifier: MIT
"""The summary figures render from results/ and reproduce derived_stats.json."""

from __future__ import annotations

import json

import pytest

from ._scripts import RESULTS, load

pytest.importorskip("matplotlib")
pytest.importorskip("scipy")

render = load("render_figures")

needs_results = pytest.mark.skipif(
    not (RESULTS / "accuracy" / "metrics.csv").is_file(),
    reason="results/ not present",
)


@needs_results
def test_render_all_from_results(tmp_path):
    stats = tmp_path / "derived_stats.json"
    rc = render.main(
        ["--from-results", "--out-dir", str(tmp_path), "--stats-out", str(stats)]
    )
    assert rc == 0
    for name in render.FIGURES:
        png = tmp_path / f"{name}.png"
        assert png.is_file() and png.stat().st_size > 20_000, name
        assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    # byte-identical in the paper environment; the Spearman p-values depend on the
    # SciPy version in their last digits, everything else is exact
    got = json.loads(stats.read_text())
    want = json.loads((RESULTS / "report_summary" / "derived_stats.json").read_text())
    assert got["fairness_amplification"] == want["fairness_amplification"]
    for budget, w in want["quality_vs_identity"].items():
        g = got["quality_vs_identity"][budget]
        assert (g["spearman_rho"], g["n"]) == (w["spearman_rho"], w["n"])
        assert g["p"] == pytest.approx(w["p"], rel=1e-9)


@needs_results
def test_only_subset_writes_no_stats(tmp_path):
    stats = tmp_path / "derived_stats.json"
    rc = render.main(
        [
            "--from-results",
            "--out-dir",
            str(tmp_path),
            "--stats-out",
            str(stats),
            "--only",
            "significance_matrix,codec_mean_rank",
        ]
    )
    assert rc == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "codec_mean_rank.png",
        "significance_matrix.png",
    ]


def test_unknown_figure_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        render.main(["--from-results", "--out-dir", str(tmp_path), "--only", "nope"])


def test_read_posthoc_scalars(tmp_path):
    p = tmp_path / "posthoc_scalars.txt"
    p.write_text(
        "n=14 k=3 friedman_chi2=20.5 friedman_p=1.31e-05 kendall_w=0.732\n"
        "mean_rank: JPEG-AI=1.14, Ours-ACCURATE=2.00, JPEG 2000=2.86\n"
    )
    head, ranks = render.read_posthoc_scalars(p)
    assert head == {
        "n": "14",
        "k": "3",
        "friedman_chi2": "20.5",
        "friedman_p": "1.31e-05",
        "kendall_w": "0.732",
    }
    assert ranks == {"JPEG-AI": 1.14, "Ours-ACCURATE": 2.0, "JPEG 2000": 2.86}
    p.write_text("garbage\n")
    with pytest.raises(ValueError):
        render.read_posthoc_scalars(p)


@needs_results
def test_significance_cell_is_in_merged_file():
    import pandas as pd

    ds, model, res, budget = render.SIGNIFICANCE_CELL
    df = pd.read_csv(RESULTS / "accuracy" / f"significance_{ds}.csv")
    cell = df[(df.model == model) & (df.res == res) & (df.budget == budget)]
    assert len(cell) == 55
    # the published matrix marks exactly these four pairs as not significant
    ns = {
        tuple(sorted((a, b)))
        for a, b, s in zip(cell.codec_a, cell.codec_b, cell.significant)
        if not s
    }
    assert ns == {
        ("avif", "webp"),
        ("heif", "jpeg_xl"),
        ("heif", "ours_fast"),
        ("jpeg_xl", "ours_fast"),
    }


def test_derived_stats_shape():
    shipped = RESULTS / "report_summary" / "derived_stats.json"
    if not shipped.is_file():
        pytest.skip("results/ not present")
    d = json.loads(shipped.read_text())
    assert set(d) == {"quality_vs_identity", "fairness_amplification"}
    assert set(d["quality_vs_identity"]) == {"1024", "512"}
