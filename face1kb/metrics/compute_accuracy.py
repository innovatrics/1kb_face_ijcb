"""Verification accuracy of every model x codec combination.

For each recognition model the script computes the uncompressed baseline
(original vs. original embeddings) and the original-to-compressed condition
for every codec: EER, accuracy/FAR/FRR at the EER threshold and FRR at the
fixed FAR operating points 1:10 ... 1:100K. Results are written as a tidy
CSV (consumed by ``face1kb.metrics.statistical_tests``) and as a markdown
report::

    outputs/metrics/accuracy_<resolution>.csv
    outputs/metrics/accuracy_<resolution>.md

Usage
-----
    python -m face1kb.metrics.compute_accuracy --resolution 112
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from face1kb import config
from face1kb.data.pairs import load_names, pair_counts
from face1kb.metrics import verification


def all_model_names() -> list[str]:
    """File-name-safe identifiers of all evaluated recognition models."""
    models = list(config.DEEPFACE_MODELS) + list(config.PROPRIETARY_MODELS)
    return [m.replace("-", "_").replace(" ", "_") for m in models]


def embedding_path(resolution: int, model: str, source: str) -> Path:
    """NPY path of *model* embeddings for *source* at *resolution*."""
    suffix = "" if source == "original" else f"_{source}"
    directory = config.embeddings_dir(resolution)
    return directory / f"{model}{suffix}_embeddings.npy"


def evaluate_pairs(
    enroll: np.ndarray, probe: np.ndarray, identity_ids: np.ndarray
) -> dict:
    """Compute all verification metrics for one enroll/probe combination."""
    scores, labels = verification.pair_scores(enroll, probe, identity_ids)
    eer, eer_threshold = verification.compute_eer(scores, labels)
    far_eer, frr_eer = verification.far_frr_at_threshold(
        scores, labels, eer_threshold
    )
    row = {
        "eer": eer,
        "eer_threshold": eer_threshold,
        "accuracy_at_eer": verification.accuracy_at_threshold(
            scores, labels, eer_threshold
        ),
        "far_at_eer": far_eer,
        "frr_at_eer": frr_eer,
    }
    operating_points = verification.frr_at_far_levels(scores, labels)
    for level, label in zip(config.FAR_LEVELS, config.FAR_LEVEL_LABELS):
        threshold, far, frr = operating_points[level]
        row[f"threshold_{label}"] = threshold
        row[f"far_{label}"] = far
        row[f"frr_{label}"] = frr
    return row


def evaluate_model(
    resolution: int, model: str, identity_ids: np.ndarray
) -> list[dict]:
    """Evaluate the baseline and every codec condition of one model."""
    baseline_path = embedding_path(resolution, model, "original")
    if not baseline_path.exists():
        print(f"skipping {model}: {baseline_path.name} not found")
        return []

    original = np.load(baseline_path)
    rows = []

    print(f"[{model}] baseline")
    baseline = evaluate_pairs(original, original, identity_ids)
    baseline.update(
        {
            "resolution": resolution,
            "model": model,
            "source": "original",
            "embedding_dim": original.shape[1],
            "eer_decrease": 0.0,
        }
    )
    rows.append(baseline)

    for codec in config.CODECS:
        codec_path = embedding_path(resolution, model, codec)
        if not codec_path.exists():
            print(f"skipping {model}/{codec}: {codec_path.name} not found")
            continue
        print(f"[{model}] {codec}")
        compressed = np.load(codec_path)
        row = evaluate_pairs(original, compressed, identity_ids)
        row.update(
            {
                "resolution": resolution,
                "model": model,
                "source": codec,
                "embedding_dim": compressed.shape[1],
                "eer_decrease": row["eer"] - baseline["eer"],
            }
        )
        rows.append(row)

    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    """Write the tidy result table."""
    lead = ["resolution", "model", "source", "embedding_dim", "eer",
            "eer_decrease", "eer_threshold", "accuracy_at_eer",
            "far_at_eer", "frr_at_eer"]
    extra = [k for k in rows[0] if k not in lead]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=lead + extra)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} result rows to {path}")


def write_markdown(rows: list[dict], path: Path, resolution: int,
                   n_mated: int, n_nonmated: int) -> None:
    """Write a human-readable report."""
    with open(path, "w") as f:
        f.write(f"# Verification accuracy at {resolution}x{resolution} px\n\n")
        f.write(f"Pairs: {n_mated:,} mated / {n_nonmated:,} non-mated\n\n")

        f.write("## EER per model and codec\n\n")
        f.write("| Model | Source | EER | ΔEER | Acc@EER |\n")
        f.write("|-------|--------|-----|------|--------|\n")
        for row in rows:
            f.write(
                f"| {row['model']} | {row['source']} | {row['eer']:.6f} "
                f"| {row['eer_decrease']:+.6f} "
                f"| {row['accuracy_at_eer']:.6f} |\n"
            )

        f.write("\n## FRR at FAR operating points\n\n")
        header = " | ".join(config.FAR_LEVEL_LABELS)
        f.write(f"| Model | Source | {header} |\n")
        f.write("|-------|--------|" + "---|" * len(config.FAR_LEVELS) + "\n")
        for row in rows:
            frrs = " | ".join(
                f"{row[f'frr_{label}']:.6f}"
                for label in config.FAR_LEVEL_LABELS
            )
            f.write(f"| {row['model']} | {row['source']} | {frrs} |\n")
    print(f"Wrote report to {path}")


def main() -> None:
    """Evaluate all models at one resolution and write CSV + report."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        choices=config.RESOLUTIONS,
    )
    args = parser.parse_args()

    _, identity_ids = load_names()
    n_mated, n_nonmated = pair_counts(identity_ids)

    rows = []
    for model in all_model_names():
        rows.extend(evaluate_model(args.resolution, model, identity_ids))
    if not rows:
        raise SystemExit(
            "No embeddings found; run the embedding stages first."
        )

    out_dir = config.metrics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / f"accuracy_{args.resolution}.csv")
    write_markdown(
        rows,
        out_dir / f"accuracy_{args.resolution}.md",
        args.resolution,
        n_mated,
        n_nonmated,
    )


if __name__ == "__main__":
    main()
