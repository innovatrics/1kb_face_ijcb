# SPDX-License-Identifier: MIT
r"""Assemble the shipped ``results/`` tree from an output root.

Copies the aggregate result files of the study from an output root (default
``FACE1KB_OUTPUT_ROOT``) into ``results/`` (default ``<repo>/results``), keeping the
output-root file layout, so that every table and figure generator can render from
either folder. Only the files listed in :data:`FILES` are copied; per-image and
per-identity files (difficulty per sample, contamination scans, JPEG-AI collapse
bitstreams), per-cell significance shards, backups and LaTeX/PNG renderings are not.

Rules applied while copying:

* CSV rows whose codec columns (``codec``, ``codec_a``, ``codec_b``,
  ``source_codec``, ``second_codec``, ``variant``) name a codec outside the study's
  codec set (:data:`face1kb.eval.embeddings.CODECS` plus the ``aligned``,
  ``original`` and ``none`` references) are dropped, and so are rows whose ``tag``
  is not a plain embedding tag (letters, digits, underscores; a stray
  ``aligned_112.npy.tmp`` duplicate row). All other lines are copied byte for byte,
  so the kept rows keep their exact float text.
* ``codec_properties.csv`` loses the ``encode_ms`` / ``decode_ms`` columns, which hold
  a "not measured" placeholder; the measured latencies are in
  ``quality/speed_*.csv``.
* ``accuracy/posthoc_scalars.txt`` is taken from ``--posthoc`` when given.
* ``report_summary/key_findings.{csv,json}``: the ``source`` fields are rewritten to
  the file names of this layout (see :data:`SOURCE_RENAMES`).
* ``fairness/*`` is also looked up under ``accuracy/`` of a source root (see
  :data:`SOURCE_ALIASES`), where output roots of the study run keep these files.
* ``contamination/contamination_summary.csv`` is derived (dataset-level counts and
  quantiles only) from the per-identity and per-crop overlap scans
  ``contamination/contamination.csv`` (``eval_identity`` = ``<dataset>/<subject>``,
  ``max_cosine``) and ``contamination/image_dup.csv`` (``eval_crop`` =
  ``<dataset>/<rel_path>``, ``min_hamming``); those two files are not copied.

Several ``--src`` roots may be given; each file is taken from the first root that
has it (e.g. a full run plus the output root of a separately run study).

Examples
--------
    python experiments/figures/collect_results.py --src outputs --dst results
    python experiments/figures/collect_results.py --src outputs --src studies_out
    python experiments/figures/collect_results.py --keep-shipped
    python experiments/figures/collect_results.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import re
import shutil
from pathlib import Path

import pandas as pd

from face1kb import config
from face1kb.eval.embeddings import CODECS

log = logging.getLogger("collect_results")

#: Codec names allowed in the codec columns of a shipped CSV.
ALLOWED_CODECS: frozenset[str] = frozenset(CODECS) | {"aligned", "original", "none"}
#: Columns that hold codec names.
CODEC_COLUMNS: tuple[str, ...] = (
    "codec",
    "codec_a",
    "codec_b",
    "source_codec",
    "second_codec",
    "variant",
)
#: Valid embedding source tags (a row whose ``tag`` has any other character, e.g. the
#: ``.npy.tmp`` name of an unfinished embedding file, is a stray duplicate).
TAG_PATTERN = re.compile(r"[A-Za-z0-9_]+")
#: Placeholder columns removed from ``codec_properties.csv``.
CODEC_PROPERTIES_DROP: tuple[str, ...] = ("encode_ms", "decode_ms")
#: ``source`` values of the findings register -> file names of this layout.
SOURCE_RENAMES: dict[str, str] = {
    "report/tables/posthoc_scalars.txt": "accuracy/posthoc_scalars.txt",
    "accuracy/fairness_kk.csv": "fairness/fairness_kk.csv",
    "accuracy/significance_colorferet__edgeface112_1024.csv + significance_kk.csv": (
        "accuracy/significance_colorferet.csv [edgeface_xs 112 px 1024 B]"
        " + significance_kk.csv"
    ),
}
#: Alternative locations (relative to a source root) tried for a shipped file when it
#: is not at its own path; the first existing candidate is used.
SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    f"fairness/{name}": (f"accuracy/{name}",)
    for name in (
        "fairness_colorferet.csv",
        "fairness_kk.csv",
        "fairness_disparity_colorferet.csv",
        "fairness_disparity_kk.csv",
        "fmr_fairness_colorferet.csv",
        "fmr_fairness_kk.csv",
        "disparity_ci_kk.csv",
    )
}
#: Hamming distance up to which a crop counts as a pHash near-match.
HAMMING_NEAR = 6
#: Identity-level cosine thresholds reported in the contamination summary.
COSINE_THRESHOLDS: tuple[float, ...] = (0.8, 0.9)

#: Shipped files (paths relative to the output root and to ``results/``) and the
#: rule that produces each: ``csv`` (codec-filtered copy), ``copy`` (byte copy),
#: ``codec_properties``, ``findings_csv``, ``findings_json``, ``posthoc`` and
#: ``contamination`` (derived summary).
FILES: dict[str, str] = {
    # verification accuracy, significance (Sections 5, 14)
    "accuracy/metrics.csv": "csv",
    "accuracy/preproc_through_codec_kk.csv": "csv",
    "accuracy/significance_colorferet.csv": "csv",
    "accuracy/significance_kk.csv": "csv",
    "accuracy/posthoc_scalars.txt": "posthoc",
    # subgroup fairness (Section 11)
    "fairness/fairness_colorferet.csv": "csv",
    "fairness/fairness_kk.csv": "csv",
    "fairness/fairness_disparity_colorferet.csv": "csv",
    "fairness/fairness_disparity_kk.csv": "csv",
    "fairness/fmr_fairness_colorferet.csv": "csv",
    "fairness/fmr_fairness_kk.csv": "csv",
    "fairness/disparity_ci_kk.csv": "csv",
    # custom-codec ablation (Section 7) and clean preprocessing / crop-tightness
    # ablation (Section 9; its ``variant`` column names preprocessing variants and
    # crop presets, not codecs, so the file is copied without codec filtering)
    "ablation/s7_ablation.csv": "csv",
    "ablation/s7_ablation.json": "copy",
    "ablation/preprocessing_ablation.csv": "copy",
    # adversarial sanitization (Section 13)
    "adversarial/sanitization.csv": "csv",
    # ISO/IEC 29794-5 annex study (Section 10)
    "annex/scores.csv": "csv",
    "annex/scores_pop2000.csv": "csv",
    "annex/ci.csv": "csv",
    # codec benchmark (Sections 2, 5, 6, 7)
    "codec_properties.csv": "codec_properties",
    "codec_comparison/file_size_summary.csv": "csv",
    "codec_comparison/jpegai_decoders.csv": "csv",
    "codec_comparison/jpegai_kk.csv": "csv",
    "codec_comparison/res112/comparison.json": "copy",
    "codec_comparison/res224/comparison.json": "copy",
    # train/test overlap (Section 7)
    "contamination/contamination_summary.csv": "contamination",
    # sample difficulty (Section 8)
    "difficulty/difficulty_correlations.csv": "csv",
    "difficulty/difficulty_contrast.csv": "csv",
    "difficulty/cross_codec_sharing.txt": "copy",
    # image quality, FIQ, identity tail, speed (Sections 6, 7)
    "quality/quality_summary.csv": "csv",
    "quality/quality_colorferet.csv": "csv",
    "quality/quality_kk.csv": "csv",
    "quality/fiq_colorferet.csv": "csv",
    "quality/fiq_kk.csv": "csv",
    "quality/idcos_tail_512.csv": "csv",
    "quality/codec_verify.csv": "csv",
    "quality/speed_benchmark.csv": "csv",
    "quality/speed_cpu_classical.csv": "csv",
    "quality/speed_cpu_jpegai.csv": "csv",
    "quality/speed_cpu_learned.csv": "csv",
    "quality/speed_gpu_jpegai.csv": "csv",
    "quality/speed_gpu_learned.csv": "csv",
    # recompression chains (Section 12)
    "recompression/recompression_colorferet_1024.csv": "csv",
    "recompression/recompression_colorferet_512.csv": "csv",
    "recompression/recompression_kk_1024.csv": "csv",
    "recompression/recompression_kk_512.csv": "csv",
    "recompression/recompression_colorferet_arcface_antelopev2.csv": "csv",
    # findings register and figure-derived scalars
    "report_summary/key_findings.csv": "findings_csv",
    "report_summary/key_findings.json": "findings_json",
    "report_summary/derived_stats.json": "copy",
    "report_summary/side_channel_leakage.json": "copy",
    # resolution-information decomposition (Section 9)
    "resolution_information/eer_by_resolution.csv": "csv",
    "resolution_information/embedding_decomposition.csv": "csv",
    "resolution_information/spectral_retention.csv": "csv",
}
#: Hand-written files of ``results/`` that are not produced by this script.
STATIC_FILES: tuple[str, ...] = ("README.md",)


def _line_terminator(text: str) -> str:
    """Return the line terminator of a CSV text (CRLF or LF)."""
    return "\r\n" if "\r\n" in text else "\n"


def _parse_line(line: str) -> list[str]:
    """Parse one CSV record (no embedded newlines)."""
    return next(csv.reader([line]))


def filter_csv(text: str, drop_columns: tuple[str, ...] = ()) -> tuple[str, int]:
    """Drop rows naming a codec outside :data:`ALLOWED_CODECS` or an invalid ``tag``.

    Also removes ``drop_columns``. Kept lines are copied verbatim unless columns are
    removed, in which case the records are re-serialised with the csv module (minimal
    quoting, same line terminator). Returns the new text and the number of dropped
    rows.
    """
    lines = text.splitlines(keepends=True)
    if len(list(csv.reader(io.StringIO(text)))) != len(lines):
        raise ValueError("CSV records span several lines; refusing to filter")
    header = _parse_line(lines[0])
    codec_ix = [i for i, c in enumerate(header) if c in CODEC_COLUMNS]
    tag_ix = header.index("tag") if "tag" in header else None
    keep_ix = [i for i, c in enumerate(header) if c not in drop_columns]
    missing = set(drop_columns) - set(header)
    if missing:
        raise ValueError(f"columns to drop are missing: {sorted(missing)}")
    eol = _line_terminator(text)
    out, dropped = [], 0

    def emit(line: str, rec: list[str]) -> None:
        if not drop_columns:
            out.append(line)
            return
        buf = io.StringIO()
        csv.writer(buf, lineterminator=eol).writerow([rec[i] for i in keep_ix])
        out.append(buf.getvalue())

    emit(lines[0], header)
    for line in lines[1:]:
        rec = _parse_line(line)
        bad_tag = tag_ix is not None and not TAG_PATTERN.fullmatch(rec[tag_ix])
        if bad_tag or any(rec[i] and rec[i] not in ALLOWED_CODECS for i in codec_ix):
            dropped += 1
            continue
        emit(line, rec)
    return "".join(out), dropped


def rename_sources(text: str) -> str:
    """Rewrite the ``source`` values of the findings register (:data:`SOURCE_RENAMES`).

    Raises ``ValueError`` if a new name contains a CSV delimiter or quote.
    """
    for old, new in SOURCE_RENAMES.items():
        if any(ch in new for ch in ',"'):
            raise ValueError(f"replacement needs CSV quoting: {new!r}")
        text = text.replace(old, new)
    return text


def contamination_summary(identity: pd.DataFrame, crops: pd.DataFrame) -> pd.DataFrame:
    """Dataset-level summary of the identity and image overlap scans.

    ``identity`` has ``eval_identity`` (``<dataset>/<subject>``) and ``max_cosine``;
    ``crops`` has ``eval_crop`` (``<dataset>/<rel_path>``) and ``min_hamming``.
    Returns a long table ``dataset, check, statistic, value`` with one block per
    dataset and one over both (``all``); no identity or image appears in it.
    """
    identity = identity.assign(dataset=identity.eval_identity.str.split("/").str[0])
    crops = crops.assign(dataset=crops.eval_crop.str.split("/").str[0])
    rows: list[tuple[str, str, str, float]] = []
    datasets = sorted(set(identity.dataset) | set(crops.dataset)) + ["all"]
    for ds in datasets:
        idf = identity if ds == "all" else identity[identity.dataset == ds]
        cdf = crops if ds == "all" else crops[crops.dataset == ds]
        if len(idf):
            cos = idf.max_cosine
            rows += [
                (ds, "identity", "n_identities", int(len(idf))),
                (ds, "identity", "max_cosine_max", float(cos.max())),
                (ds, "identity", "max_cosine_median", float(cos.median())),
            ]
            for t in COSINE_THRESHOLDS:
                n_above = int((cos > t).sum())
                rows.append((ds, "identity", f"n_max_cosine_gt_{t}", n_above))
        if len(cdf):
            ham = cdf.min_hamming
            near = int((ham <= HAMMING_NEAR).sum())
            rows += [
                (ds, "image", "n_crops", int(len(cdf))),
                (ds, "image", "min_hamming_median", float(ham.median())),
                (ds, "image", f"n_hamming_le_{HAMMING_NEAR}", near),
                (ds, "image", f"frac_hamming_le_{HAMMING_NEAR}", near / len(cdf)),
            ]
            for k, n in ham.value_counts().sort_index().items():
                rows.append((ds, "image", f"n_hamming_eq_{int(k)}", int(n)))
    df = pd.DataFrame(rows, columns=["dataset", "check", "statistic", "value"])
    # keep counts as integers in the CSV text (the column mixes counts and floats)
    df["value"] = pd.Series([v for *_, v in rows], dtype=object)
    return df


def source_path(rel: str, roots: list[Path]) -> Path:
    """First existing location of ``rel`` (or one of its aliases) in ``roots``.

    Raises ``FileNotFoundError`` naming the primary location in the first root.
    """
    for root in roots:
        for cand in (rel, *SOURCE_ALIASES.get(rel, ())):
            if (root / cand).is_file():
                return root / cand
    raise FileNotFoundError(2, "No such file", str(roots[0] / rel))


def build(rel: str, rule: str, src: list[Path], posthoc: Path | None) -> bytes:
    """Return the content of one shipped file, read from the roots ``src``."""
    if rule == "contamination":
        base = source_path("contamination/contamination.csv", src).parent
        df = contamination_summary(
            pd.read_csv(base / "contamination.csv"),
            pd.read_csv(base / "image_dup.csv"),
        )
        return df.to_csv(index=False).encode()
    if rule == "posthoc" and posthoc is not None:
        path = posthoc
    else:
        path = source_path(rel, src)
    raw = path.read_bytes()
    if rule in ("copy", "posthoc"):
        return raw
    text = raw.decode("utf-8")
    if rule in ("findings_csv", "findings_json"):
        return rename_sources(text).encode()
    drop = CODEC_PROPERTIES_DROP if rule == "codec_properties" else ()
    new, dropped = filter_csv(text, drop)
    if dropped:
        log.info(
            "%s: dropped %d rows (codec outside the study or stray tag)", rel, dropped
        )
    return new.encode()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--src",
        type=Path,
        action="append",
        default=None,
        help="output root, repeatable; first root with a file wins "
        "(default: OUTPUT_ROOT)",
    )
    ap.add_argument(
        "--dst", type=Path, default=None, help="results folder (default: results/)"
    )
    ap.add_argument(
        "--posthoc",
        type=Path,
        default=None,
        help="posthoc_scalars.txt (default: accuracy/posthoc_scalars.txt of --src)",
    )
    ap.add_argument(
        "--keep-shipped",
        action="store_true",
        help="keep the existing --dst copy of a file that no --src root has (files "
        "the public pipeline does not produce, e.g. codec_properties.csv, the "
        "findings register or comparison.json) instead of failing",
    )
    ap.add_argument("--dry-run", action="store_true", help="only report")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    src = [p.resolve() for p in (args.src or [config.OUTPUT_ROOT])]
    dst = (args.dst or config.RESULTS_ROOT).resolve()
    if dst in src:
        raise SystemExit("--src and --dst must differ")
    missing = []
    for rel, rule in FILES.items():
        try:
            data = build(rel, rule, src, args.posthoc)
        except FileNotFoundError as exc:
            if args.keep_shipped and (dst / rel).is_file():
                log.warning("no input for %s; kept the shipped copy", rel)
            else:
                missing.append(f"{rel} ({exc.filename})")
            continue
        out = dst / rel
        if args.dry_run:
            log.info("would write %s (%d bytes)", out, len(data))
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".part")
        tmp.write_bytes(data)
        shutil.move(tmp, out)
    known = set(FILES) | set(STATIC_FILES)
    for p in sorted(dst.rglob("*")) if dst.is_dir() else []:
        if p.is_file() and p.relative_to(dst).as_posix() not in known:
            log.warning("not a shipped result file: %s", p)
    if missing:
        for m in missing:
            log.error("missing input: %s", m)
        return 1
    log.info("%d files in %s", len(FILES), dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
