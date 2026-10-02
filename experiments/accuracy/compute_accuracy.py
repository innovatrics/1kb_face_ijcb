# SPDX-License-Identifier: MIT
r"""Verification accuracy of every embedding source: the ``metrics.csv`` grid.

For each dataset, model and embedding array (``config.embeddings_path``) this scores
all mated pairs of ``pairs.parquet`` plus one seeded sample of 5,000,000 non-mated
pairs (seed 0) by cosine similarity and reports EER, FNMR at FMR = 1e-2 / 1e-3 /
1e-4 with the realised FMR, and, for the clean cells (empty suffix) of ArcFace,
LVFace-L, TopoFR-R100, EdgeFace-XS and CVLface IR-101, 95 % subject-cluster
bootstrap intervals (200 replicates, impostors capped at 1,000,000). Every row also
carries ``eer_aligned`` (the clean aligned source of the same dataset, model and
resolution) and ``delta_eer``. The metric definitions are those of
:mod:`face1kb.eval.verification` (see ``docs/metrics.md``).

Outputs (under ``--out-dir``, default ``config.output_dir("accuracy")``):

* ``metrics_<dataset>.csv`` -- the rows of one dataset;
* ``metrics.csv`` -- all requested datasets, written with
  :func:`face1kb.eval.verification.write_metrics_csv`, which reproduces the text of
  the published file.

Both members of a pair come from the same array, so a compressed source is scored
compressed-vs-compressed. The published grid was scored with the CUDA scorer
(``--device cuda``, the default when a GPU is visible); scoring on the CPU gives the
same numbers up to one trial in a few cells. ``median_bytes`` is NaN (file sizes are
reported in ``codec_comparison/file_size_summary.csv``).

``--tags`` restricts scoring to the listed sources; the clean ``aligned_<res>``
source of each listed resolution is added (and written) so that ``eer_aligned`` and
``delta_eer`` stay defined. Rows left without a baseline are logged.

``--merge`` skips scoring and rebuilds ``metrics.csv`` from existing
``metrics_<dataset>.csv`` files (e.g. after one run per dataset on separate GPUs);
their values are used as written, derived columns are not recomputed.

Examples
--------
    python experiments/accuracy/compute_accuracy.py --models all
    python experiments/accuracy/compute_accuracy.py --datasets kk --models all \
        --device cuda:1 --no-combined
    python experiments/accuracy/compute_accuracy.py --merge
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from face1kb import config

log = logging.getLogger("compute_accuracy")


def _default_device() -> str | None:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return "cuda" if torch.cuda.is_available() else None


def with_baselines(tags: list[str]) -> list[str]:
    """``tags`` plus the clean ``aligned_<res>`` source of every listed resolution.

    ``eer_aligned`` and ``delta_eer`` are taken from that source, so a tag filter
    always keeps it. Tags that name no known codec are passed through unchanged.
    """
    from face1kb.eval.embeddings import parse_tag  # noqa: PLC0415

    out = list(tags)
    for tag in tags:
        info = parse_tag(tag)
        base = f"aligned_{info['res']}" if info else None
        if base and base not in out:
            out.append(base)
    return out


def _warn_missing_baseline(dataset: str, df) -> None:
    """Log the rows whose clean aligned source was not scored (``eer_aligned`` NaN)."""
    lost = df[df["eer_aligned"].isna() & df["eer"].notna()]
    if len(lost):
        log.warning(
            "[%s] %d rows have no clean aligned_<res> baseline (eer_aligned and "
            "delta_eer are NaN): %s",
            dataset,
            len(lost),
            sorted(set(zip(lost.model, lost.tag))),
        )


def score_dataset(dataset: str, models: list[str], args):
    """``metrics_frame`` of one dataset (all requested models and sources)."""
    from face1kb.eval import verification as ver  # noqa: PLC0415

    boot_models = [m.strip() for m in args.boot_models.split(",") if m.strip()]
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] or None
    if tags:
        tags = with_baselines(tags)
    rows = ver.accuracy_rows(
        dataset,
        models,
        sample_nonmated=args.sample_nonmated,
        bootstrap=args.bootstrap,
        boot_nonmated=args.boot_nonmated,
        boot_level=args.boot_level,
        boot_models=boot_models or None,
        seed=args.seed,
        device=args.device,
        cpu_fallback=not args.no_cpu_fallback,
        tags=tags,
    )
    df = ver.metrics_frame(rows)
    _warn_missing_baseline(dataset, df)
    return df


def merge(out_dir: Path, datasets: list[str]) -> int:
    """Concatenate ``metrics_<dataset>.csv`` into ``metrics.csv``."""
    import pandas as pd  # noqa: PLC0415

    parts = []
    for ds in datasets:
        f = out_dir / f"metrics_{ds}.csv"
        if not f.is_file():
            log.error("missing %s; cannot merge", f)
            return 1
        parts.append(pd.read_csv(f))
        log.info(
            "[%s] %d rows, %d models", ds, len(parts[-1]), parts[-1].model.nunique()
        )
    out = out_dir / "metrics.csv"
    pd.concat(parts, ignore_index=True).to_csv(out, index=False)
    log.info("wrote %s", out)
    return 0


def _guard_shrink(path: Path, frames, allow: bool) -> bool:
    """Refuse to replace a metrics.csv by one that covers fewer models."""
    import pandas as pd  # noqa: PLC0415

    if allow or not path.is_file():
        return True
    old = set(pd.read_csv(path, usecols=["model"]).model.unique())
    new = set().union(*(set(f.model.unique()) for f in frames))
    lost = old - new
    if lost:
        log.error(
            "refusing to overwrite %s: it covers %d models and this run only %d "
            "(missing %s); use --models all, another --out-dir, or --allow-shrink",
            path,
            len(old),
            len(new),
            sorted(lost),
        )
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface."""
    from face1kb.eval import verification as ver  # noqa: PLC0415

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument(
        "--models",
        default="all",
        help="'all' (every model with embeddings), 'anchor' or a comma list",
    )
    ap.add_argument(
        "--tags",
        default="",
        help="restrict to these source tags (comma list); the clean aligned_<res> "
        "source of every listed resolution is scored and written too, as the "
        "eer_aligned / delta_eer baseline",
    )
    ap.add_argument(
        "--device",
        default="auto",
        help="scoring device: 'auto' (cuda if available), 'cpu', 'cuda', 'cuda:N'",
    )
    ap.add_argument(
        "--no-cpu-fallback",
        action="store_true",
        help="fail instead of scoring on the CPU when the GPU runs out of memory",
    )
    ap.add_argument(
        "--sample-nonmated", type=int, default=ver.NONMATED_SAMPLE["accuracy"]
    )
    ap.add_argument("--bootstrap", type=int, default=ver.BOOTSTRAP_REPS)
    ap.add_argument("--boot-nonmated", type=int, default=ver.BOOTSTRAP_NONMATED_CAP)
    ap.add_argument("--boot-level", choices=("subject", "pair"), default="subject")
    ap.add_argument(
        "--boot-models",
        default=",".join(ver.BOOTSTRAP_MODELS),
        help="models that get CIs (clean cells only); empty: all",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None, help="default: outputs/accuracy")
    ap.add_argument(
        "--no-combined",
        action="store_true",
        help="write only metrics_<dataset>.csv (merge later with --merge)",
    )
    ap.add_argument(
        "--merge", action="store_true", help="only merge metrics_<dataset>.csv"
    )
    ap.add_argument(
        "--allow-shrink",
        action="store_true",
        help="allow replacing metrics.csv by one covering fewer models",
    )
    ap.add_argument("--dry-run", action="store_true", help="list the arrays only")
    return ap


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - CLI entry point
    """Score the requested datasets and write the metrics CSVs."""
    from face1kb.eval import verification as ver  # noqa: PLC0415
    from face1kb.eval.embeddings import list_sources, resolve_models  # noqa: PLC0415

    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else config.output_dir("accuracy")
    if args.merge:
        return merge(out_dir, datasets)
    args.device = _default_device() if args.device == "auto" else args.device
    if args.device == "cpu":
        args.device = None
    log.info("scoring on %s", args.device or "CPU (NumPy)")

    frames = []
    for ds in datasets:
        models = resolve_models(args.models, ds)
        if not models:
            log.warning("[%s] no embeddings for the requested models; skipped", ds)
            continue
        if args.dry_run:
            n = sum(len(list_sources(ds, m, aligned_codec="")) for m in models)
            log.info("[%s] %d models, %d arrays", ds, len(models), n)
            continue
        df = score_dataset(ds, models, args)
        out_dir.mkdir(parents=True, exist_ok=True)
        part = out_dir / f"metrics_{ds}.csv"
        df.to_csv(part, index=False)
        log.info("[%s] wrote %s (%d rows)", ds, part, len(df))
        frames.append(df)
    if args.dry_run or args.no_combined or not frames:
        return 0
    out = out_dir / "metrics.csv"
    if not _guard_shrink(out, frames, args.allow_shrink):
        return 1
    ver.write_metrics_csv(frames, out)
    log.info("wrote %s (%d rows)", out, sum(len(f) for f in frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
