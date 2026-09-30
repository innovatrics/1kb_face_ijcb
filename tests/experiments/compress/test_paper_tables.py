# SPDX-License-Identifier: MIT
"""The generators reproduce the published tables from the shipped results/.

``expected/`` holds the arXiv v1 sources of the tables. ``fit_rate_kk_1024`` differs
by design: the published copy still lists bmshj2018 and mbt2018 as rows of ``--``
(AI-Solutions-KK has no CompressAI cell with at least 30 files), which the generator
omits.
"""

from __future__ import annotations

import pytest

from face1kb import config

from ._scripts import EXPECTED, load

RESULTS = config.RESULTS_ROOT


def _need(*rel):
    missing = [r for r in rel if not (RESULTS / r).is_file()]
    if missing:
        pytest.skip(f"shipped results missing: {missing}")


def _expected(stem: str) -> str:
    return (EXPECTED / f"{stem}.tex").read_text()


@pytest.mark.parametrize(
    "stem",
    [
        "fit_rate_colorferet_1024",
        "fit_rate_colorferet_512",
        "fit_rate_kk_512",
        "fit_rate_kk_1024",
    ],
)
def test_fit_rate_tables(stem, tmp_path):
    _need("codec_comparison/file_size_summary.csv")
    mod = load("generate_file_size_boxplots")
    assert (
        mod.main(
            [
                "--from-summary",
                "--from-results",
                "--no-figures",
                "--output-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    got = (tmp_path / "tables" / f"{stem}.tex").read_text()
    want = _expected(stem)
    if stem == "fit_rate_kk_1024":
        want = "".join(
            ln
            for ln in want.splitlines(keepends=True)
            if not ln.startswith(("bmshj2018 &", "mbt2018 &"))
        )
    assert got == want


def test_speed_table(tmp_path):
    _need("quality/speed_cpu_classical.csv", "quality/speed_gpu_learned.csv")
    mod = load("generate_speed_table", "speed")
    assert (
        mod.main(["--from-results", "--no-figure", "--output-root", str(tmp_path)]) == 0
    )
    assert (tmp_path / "tables" / "speed_benchmark.tex").read_text() == _expected(
        "speed_benchmark"
    )


def test_codec_properties_table(tmp_path):
    _need("codec_properties.csv", "codec_comparison/file_size_summary.csv")
    mod = load("generate_codec_properties_table")
    assert mod.main(["--from-results", "--output-root", str(tmp_path)]) == 0
    assert (tmp_path / "tables" / "codec_properties.tex").read_text() == _expected(
        "codec_properties"
    )


def test_jpegai_speed_table(tmp_path):
    _need("codec_comparison/jpegai_decoders.csv")
    mod = load("generate_jpegai_speed_table")
    assert mod.main(["--output-root", str(tmp_path)]) == 0
    assert (tmp_path / "tables" / "jpegai_speed.tex").read_text() == _expected(
        "jpegai_speed"
    )


def test_jpegai_kk_table(tmp_path):
    _need("codec_comparison/jpegai_kk.csv")
    mod = load("measure_jpegai_kk")
    assert (
        mod.main(["--phase", "table", "--from-results", "--output-root", str(tmp_path)])
        == 0
    )
    assert (tmp_path / "tables" / "jpegai_kk.tex").read_text() == _expected("jpegai_kk")
