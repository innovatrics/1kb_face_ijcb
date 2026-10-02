# SPDX-License-Identifier: MIT
"""The shipped results/ tree: file set, aggregate-only content, internal consistency."""

from __future__ import annotations

import csv
import io
import json
import re

import pandas as pd
import pytest

from face1kb.eval.embeddings import CODECS

from ._scripts import RESULTS, load

collect = load("collect_results")

#: Data rows per shipped CSV (a change means the aggregation grid changed).
EXPECTED_ROWS = {
    "ablation/preprocessing_ablation.csv": 52,
    "ablation/s7_ablation.csv": 32,
    "accuracy/metrics.csv": 4618,
    "accuracy/preproc_through_codec_kk.csv": 36,
    "accuracy/significance_colorferet.csv": 220,
    "accuracy/significance_kk.csv": 440,
    "adversarial/sanitization.csv": 6048,
    "annex/ci.csv": 44,
    "annex/scores.csv": 1152,
    "annex/scores_pop2000.csv": 984,
    "codec_comparison/file_size_summary.csv": 216,
    "codec_comparison/jpegai_decoders.csv": 6,
    "codec_comparison/jpegai_kk.csv": 8,
    "codec_properties.csv": 10,
    "contamination/contamination_summary.csv": 52,
    "difficulty/difficulty_contrast.csv": 6,
    "difficulty/difficulty_correlations.csv": 14,
    "fairness/disparity_ci_kk.csv": 40,
    "fairness/fairness_colorferet.csv": 12848,
    "fairness/fairness_disparity_colorferet.csv": 2336,
    "fairness/fairness_disparity_kk.csv": 1188,
    "fairness/fairness_kk.csv": 5656,
    "fairness/fmr_fairness_colorferet.csv": 3600,
    "fairness/fmr_fairness_kk.csv": 2352,
    "quality/codec_verify.csv": 40,
    "quality/fiq_colorferet.csv": 44,
    "quality/fiq_kk.csv": 44,
    "quality/idcos_tail_512.csv": 10,
    "quality/quality_colorferet.csv": 108,
    "quality/quality_kk.csv": 94,
    "quality/quality_summary.csv": 202,
    "quality/speed_benchmark.csv": 61,
    "quality/speed_cpu_classical.csv": 70,
    "quality/speed_cpu_jpegai.csv": 5,
    "quality/speed_cpu_learned.csv": 12,
    "quality/speed_gpu_jpegai.csv": 2,
    "quality/speed_gpu_learned.csv": 40,
    "recompression/recompression_colorferet_1024.csv": 64,
    "recompression/recompression_colorferet_512.csv": 64,
    "recompression/recompression_colorferet_arcface_antelopev2.csv": 128,
    "recompression/recompression_kk_1024.csv": 64,
    "recompression/recompression_kk_512.csv": 64,
    "report_summary/key_findings.csv": 18,
    "resolution_information/eer_by_resolution.csv": 120,
    "resolution_information/embedding_decomposition.csv": 116,
    "resolution_information/spectral_retention.csv": 10,
}
#: Column names that would identify an image, a subject or a file.
PER_ITEM_COLUMNS = {
    "subject",
    "identity",
    "stem",
    "image",
    "rel_path",
    "path",
    "file",
    "filename",
    "eval_identity",
    "eval_crop",
    "webface_row",
    "webface_subject_id",
    "original_png",
    "decoded_png",
}
#: Absolute paths and per-item file names must not appear in any shipped file.
LEAK = re.compile(
    r"(?:^|[\s,\"'(=])/(?:mnt|srv|home|tmp|data|workspace|root|Users)/"
    r"|\.(?:png|jpe?g|bin|jpegai|npy|pt|tmp)\b",
    re.IGNORECASE,
)

needs_results = pytest.mark.skipif(
    not (RESULTS / "accuracy" / "metrics.csv").is_file(),
    reason="results/ not present",
)


def _shipped():
    return sorted(
        p.relative_to(RESULTS).as_posix() for p in RESULTS.rglob("*") if p.is_file()
    )


def _csvs():
    return [r for r in collect.FILES if r.endswith(".csv")]


# ------------------------------------------------------------------ the tree
@needs_results
def test_file_set_is_the_manifest():
    assert _shipped() == sorted(set(collect.FILES) | set(collect.STATIC_FILES))


@needs_results
def test_row_counts():
    assert set(EXPECTED_ROWS) == set(_csvs())
    for rel, n in EXPECTED_ROWS.items():
        assert len(pd.read_csv(RESULTS / rel)) == n, rel


@needs_results
@pytest.mark.parametrize("rel", sorted(EXPECTED_ROWS))
def test_no_per_item_columns(rel):
    cols = set(pd.read_csv(RESULTS / rel, nrows=0).columns)
    assert not cols & PER_ITEM_COLUMNS, rel


def _json_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _json_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _json_keys(v)


@needs_results
def test_json_files_are_aggregates():
    for rel in collect.FILES:
        if rel.endswith(".json"):
            keys = set(_json_keys(json.loads((RESULTS / rel).read_text())))
            assert not keys & PER_ITEM_COLUMNS, rel


@needs_results
def test_no_paths_or_file_names():
    for rel in _shipped():
        text = (RESULTS / rel).read_text(encoding="utf-8")
        m = LEAK.search(text)
        assert m is None, f"{rel}: {m.group(0)!r}"


@needs_results
@pytest.mark.parametrize("rel", sorted(EXPECTED_ROWS))
def test_only_study_codecs(rel):
    allowed = set(CODECS) | {"aligned", "original", "none"}
    assert collect.ALLOWED_CODECS == allowed
    df = pd.read_csv(RESULTS / rel, keep_default_na=False)
    assert "cheng" not in (RESULTS / rel).read_text().lower(), rel
    if collect.FILES[rel] == "copy":
        # byte copies whose codec-named columns hold other labels (e.g. the
        # preprocessing variants of ablation/preprocessing_ablation.csv)
        return
    for col in collect.CODEC_COLUMNS:
        if col in df:
            extra = set(df[col]) - allowed - {""}
            assert not extra, f"{rel}:{col} {sorted(extra)}"
    if "tag" in df:
        assert all(collect.TAG_PATTERN.fullmatch(t) for t in df.tag), rel


@needs_results
def test_codec_properties_has_no_placeholders():
    df = pd.read_csv(RESULTS / "codec_properties.csv")
    assert not set(collect.CODEC_PROPERTIES_DROP) & set(df.columns)
    assert not df.astype(str).apply(lambda c: c.str.contains("TODO")).any().any()


# ------------------------------------------------------------------ consistency
@needs_results
@pytest.mark.parametrize(
    "dataset,cells",
    [
        ("colorferet", {(m, 112, 1024) for m in ("a", "e", "l", "t")}),
        ("kk", {(m, r, 1024) for m in ("a", "e", "l", "t") for r in (112, 224)}),
    ],
)
def test_significance_cells_are_complete(dataset, cells):
    """Every merged cell holds all 55 pairs of the 10 codecs plus aligned."""
    df = pd.read_csv(RESULTS / "accuracy" / f"significance_{dataset}.csv")
    got = {(m[0], r, b) for m, r, b in zip(df.model, df.res, df.budget)}
    assert got == cells
    for _, cell in df.groupby(["model", "res", "budget"]):
        assert len(cell) == 55
        assert len(set(cell.codec_a) | set(cell.codec_b)) == 11


@needs_results
def test_posthoc_scalars_match_metrics():
    """posthoc_scalars.txt is reproduced from metrics.csv by face1kb.eval."""
    from face1kb.eval.significance import eer_matrix, posthoc
    from face1kb.report.style import CODEC_LABELS

    render = load("render_figures")
    head, ranks = render.read_posthoc_scalars(
        RESULTS / "accuracy" / "posthoc_scalars.txt"
    )
    r = posthoc(eer_matrix(pd.read_csv(RESULTS / "accuracy" / "metrics.csv")))
    assert (int(head["n"]), int(head["k"])) == (r["n"], r["k"])
    assert head["friedman_chi2"] == f"{r['friedman_chi2']:.1f}"
    assert head["friedman_p"] == f"{r['friedman_p']:.2e}"
    assert head["kendall_w"] == f"{r['kendall_w']:.3f}"
    assert ranks == {
        CODEC_LABELS[c]: round(float(v), 2) for c, v in r["mean_rank"].items()
    }


@needs_results
def test_contamination_summary_numbers():
    """The overlap numbers quoted in Section 7 of the paper."""
    df = pd.read_csv(RESULTS / "contamination" / "contamination_summary.csv")
    v = {(d, s): x for d, s, x in zip(df.dataset, df.statistic, df.value)}
    assert v["all", "n_identities"] == 1099
    assert v["all", "n_max_cosine_gt_0.8"] == 22
    assert v["colorferet", "n_max_cosine_gt_0.8"] == 0
    assert round(v["all", "max_cosine_max"], 3) == 0.915
    assert round(100 * v["all", "frac_hamming_le_6"]) == 34
    assert v["all", "n_hamming_eq_0"] == 3


@needs_results
def test_preprocessing_ablation_variants():
    """The clean crop-tightness and preprocessing grid of Tables 49 and 50."""
    df = pd.read_csv(RESULTS / "ablation" / "preprocessing_ablation.csv")
    variants = {s: set(g.variant) for s, g in df.groupby("study")}
    assert variants["crop"] == {"standard", "tight", "mid", "fill"}
    assert variants["preproc"] == {"std", *"A1 A2 A3 A4 B1 B2 C1 C2".split()}
    assert df.groupby(["study", "variant"]).size().eq(4).all()


# ------------------------------------------------------------------ collector units
def test_source_path_aliases(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    (a / "accuracy").mkdir(parents=True)
    (b / "fairness").mkdir(parents=True)
    (a / "accuracy" / "fairness_kk.csv").write_text("x\n")
    (b / "fairness" / "fairness_kk.csv").write_text("y\n")
    (b / "difficulty").mkdir()
    (b / "difficulty" / "difficulty_contrast.csv").write_text("z\n")
    rel = "fairness/fairness_kk.csv"
    assert collect.source_path(rel, [b, a]) == b / rel
    assert collect.source_path(rel, [a, b]) == a / "accuracy" / "fairness_kk.csv"
    got = collect.source_path("difficulty/difficulty_contrast.csv", [a, b])
    assert got == b / "difficulty" / "difficulty_contrast.csv"
    with pytest.raises(FileNotFoundError):
        collect.source_path("quality/fiq_kk.csv", [a, b])


def test_filter_csv_keeps_lines_verbatim():
    text = (
        "codec,res,eer\n"
        "jpeg,112,0.10000000000000001\n"
        "not_a_study_codec,112,0.2\n"
        "aligned,112,1e-05\n"
    )
    new, dropped = collect.filter_csv(text)
    assert dropped == 1
    assert new == "codec,res,eer\njpeg,112,0.10000000000000001\naligned,112,1e-05\n"


def test_filter_csv_drops_columns_and_keeps_crlf():
    text = 'codec,a,b\r\njpeg,"x, y",1\r\nother,z,2\r\n'
    new, dropped = collect.filter_csv(text, drop_columns=("b",))
    assert dropped == 1
    assert new == 'codec,a\r\njpeg,"x, y"\r\n'


def test_filter_csv_drops_stray_tags():
    text = "tag,codec,eer\naligned_112,,0.1\naligned_112.npy.tmp,,0.1\n"
    new, dropped = collect.filter_csv(text)
    assert dropped == 1 and ".tmp" not in new


def test_filter_csv_rejects_multiline_records():
    with pytest.raises(ValueError):
        collect.filter_csv('codec,note\njpeg,"a\nb"\n')


def test_rename_sources():
    text = "src\nreport/tables/posthoc_scalars.txt\n"
    assert collect.rename_sources(text) == "src\naccuracy/posthoc_scalars.txt\n"
    for new in collect.SOURCE_RENAMES.values():
        rec = next(csv.reader(io.StringIO(new)))
        assert rec == [new]


def test_contamination_summary_unit():
    ident = pd.DataFrame(
        {"eval_identity": ["a/x", "a/y", "b/z"], "max_cosine": [0.5, 0.85, 0.95]}
    )
    crops = pd.DataFrame(
        {"eval_crop": ["a/x/1.png", "b/z/2.png", "b/z/3.png"], "min_hamming": [0, 8, 6]}
    )
    df = collect.contamination_summary(ident, crops)
    v = {(d, c, s): x for d, c, s, x in df.itertuples(index=False)}
    assert v["all", "identity", "n_max_cosine_gt_0.8"] == 2
    assert v["all", "identity", "n_max_cosine_gt_0.9"] == 1
    assert v["b", "image", "n_hamming_le_6"] == 1
    assert v["all", "image", "n_hamming_eq_0"] == 1
    text = df.to_csv(index=False)
    assert "png" not in text and "/x" not in text
    assert "a,identity,n_identities,2\n" in text
