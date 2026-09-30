# SPDX-License-Identifier: MIT
"""The quality, codec-comparison and fairness generators reproduce the paper tables.

The tables are rendered from the shipped aggregates in ``results/`` and compared with
the arXiv v1 sources in ``expected/`` (``--paper-header`` writes the first comment
line of the published tables). Exceptions, asserted explicitly:

* ``fiq_grid`` matches with ``--shading raw`` (the rule of the published table);
* ``disparity_ci_kk``: the published WebP and JPEG 2000 rows do not match the
  shipped ``disparity_ci_kk.csv``; the reference row and the Ours rows do;
* the tabulars typed inline in the fairness section match up to whitespace.
"""

from __future__ import annotations

import re

import pytest

from face1kb import config

from ._scripts import EXPECTED, load

RESULTS = config.RESULTS_ROOT


def _need(*rel: str) -> None:
    missing = [r for r in rel if not (RESULTS / r).is_file()]
    if missing:
        pytest.skip(f"shipped results missing: {missing}")


def _expected(stem: str) -> str:
    return (EXPECTED / f"{stem}.tex").read_text()


def _norm(text: str) -> list[str]:
    return [
        re.sub(r"\s+", " ", ln).strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("%")
    ]


def _plain(text: str) -> str:
    """Drop the shading colour and the bold / underline marker."""
    text = re.sub(r"\\cellcolor\{[a-z]+!\d+\}", "", text)
    return re.sub(r"\\(?:textbf|underline)\{([^{}]*)\}", r"\1", text)


FAIRNESS_CSVS = (
    "fairness/fairness_colorferet.csv",
    "fairness/fairness_kk.csv",
    "fairness/fairness_disparity_colorferet.csv",
    "fairness/fmr_fairness_colorferet.csv",
    "fairness/disparity_ci_kk.csv",
)


@pytest.fixture(scope="module")
def fairness_tables(tmp_path_factory):
    _need(*FAIRNESS_CSVS)
    out = tmp_path_factory.mktemp("fairness_tables")
    assert (
        load("fairness", "tables").main(
            ["--from-results", "--paper-header", "--out-dir", str(out)]
        )
        == 0
    )
    return out


@pytest.mark.parametrize(
    "stem", ["fairness_disparity_cf", "fairness_disparity_512", "fmr_fairness_cf"]
)
def test_fairness_tables_byte_identical(fairness_tables, stem):
    assert (fairness_tables / f"{stem}.tex").read_text() == _expected(stem)


@pytest.mark.parametrize(
    "stem", ["fairness_subgroup_cf", "fairness_subgroup_kk", "fairness_disparity_kk"]
)
def test_inline_fairness_tables(fairness_tables, stem):
    assert _norm((fairness_tables / f"{stem}.tex").read_text()) == _norm(
        _expected(stem)
    )


def test_disparity_ci_rows(fairness_tables):
    def rows(text):
        return {ln.split("&")[0].strip(): ln for ln in _norm(_plain(text)) if "&" in ln}

    got = rows((fairness_tables / "disparity_ci_kk.tex").read_text())
    want = rows(_expected("disparity_ci_kk"))
    assert got.keys() == want.keys()
    stale = {"WebP", "JPEG~2000"}
    for key in want:
        if key in stale:
            assert got[key] != want[key]
        else:
            assert got[key] == want[key], key


def test_quality_tables(tmp_path):
    _need("quality/quality_colorferet.csv", "quality/quality_kk.csv")
    mod = load("quality", "quality_tables")
    assert (
        mod.main(["--from-results", "--paper-header", "--out-dir", str(tmp_path)]) == 0
    )
    for stem in ("quality_matrix", "budget_sweep"):
        assert (tmp_path / f"{stem}.tex").read_text() == _expected(stem), stem


def test_fiq_grid(tmp_path):
    _need("quality/fiq_colorferet.csv", "quality/fiq_kk.csv")
    mod = load("quality", "fiq_report")
    args = [
        "--from-results",
        "--paper-header",
        "--out-dir",
        str(tmp_path),
        "--no-figure",
    ]
    assert mod.main([*args, "--shading", "raw"]) == 0
    assert (tmp_path / "fiq_grid.tex").read_text() == _expected("fiq_grid")
    assert mod.main([*args, "--shading", "printed"]) == 0
    printed = (tmp_path / "fiq_grid.tex").read_text()
    assert printed != _expected("fiq_grid")
    assert _plain(printed) == _plain(_expected("fiq_grid"))  # same values


def test_codec_comparison_tables(tmp_path):
    _need(
        "codec_comparison/res112/comparison.json",
        "codec_comparison/res224/comparison.json",
        "quality/quality_summary.csv",
    )
    mod = load("codec_comparison", "tables")
    assert (
        mod.main(["--from-results", "--paper-header", "--out-dir", str(tmp_path)]) == 0
    )
    for stem in (
        "codec_comparison_112",
        "codec_comparison_224",
        "codec_results",
        "codec_results_112",
    ):
        assert (tmp_path / f"{stem}.tex").read_text() == _expected(stem), stem


def test_jpegai_tradeoff_points():
    _need("codec_comparison/jpegai_decoders.csv")
    mod = load("codec_comparison", "jpegai_tradeoff")
    points = mod.load_points(RESULTS / "codec_comparison" / "jpegai_decoders.csv")
    # the values printed in tab:jpegai-speed
    assert points == {
        "SOP": (572, 655, 34.99),
        "BOP": (599, 632, 35.36),
        "HOP": (1192, 656, 35.93),
    }


def test_default_header_names_the_generator(tmp_path):
    _need("quality/quality_colorferet.csv", "quality/quality_kk.csv")
    mod = load("quality", "quality_tables")
    assert mod.main(["--from-results", "--out-dir", str(tmp_path)]) == 0
    tex = (tmp_path / "quality_matrix.tex").read_text()
    assert tex.splitlines()[0] == "% generated by experiments/quality/quality_tables.py"
    assert tex.splitlines()[1:] == _expected("quality_matrix").splitlines()[1:]
