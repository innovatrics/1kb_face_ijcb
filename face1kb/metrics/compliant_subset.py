"""Verification accuracy on the size-compliant image subsets (Sec. 3.4).

At 224x224 px only JPEG-AI and JPEG-FzT compress every image to 1024 B or
less (Fig. 3); the other codecs keep some over-budget bitstreams, so the
full-set comparison is not strictly like-for-like. This stage repeats the
verification evaluation of ``face1kb.metrics.compute_accuracy`` on image
subsets for which the compared codecs do meet the budget. It only
re-aggregates the stored embeddings - nothing is re-compressed or
re-embedded. A pair is kept iff both of its images belong to the subset, and
the uncompressed baseline (hence the EER increase) is recomputed on the same
subset. Three kinds of subset are available:

``intersection``
    Images that every included codec compresses to at most ``--budget``
    bytes - the paper's matched subset. JPEG2000 is excluded by default: at
    224x224 px it meets the budget for no image, so any intersection that
    contains it is empty.
``relaxed``
    The same intersection at the smallest budget (``--budget`` plus
    multiples of ``--relaxed-step``) that covers at least
    ``--relaxed-coverage`` of the images - the paper's sensitivity check.
``compliant_<codec>``
    Each included codec on its own compliant images, for reference; these
    subsets differ between codecs and are evaluated only on request
    (``--subsets intersection relaxed per_codec``).

On the intersection subset the codecs are compared with the tests of
``face1kb.metrics.statistical_tests`` (Friedman with Kendall's W, pairwise
Wilcoxon signed-rank with Holm correction, Cliff's delta). Each model is a
block and its EER on the subset is the dependent variable, as in the paper;
since all codecs of a model share the baseline, the EER increase would give
the same Friedman and Wilcoxon results but different Cliff's deltas.

Kendall's W is chi-squared / (n_models * (k_codecs - 1)), as in
``statistical_tests``; see "Differences from the paper" in README.md.

The paper evaluates this analysis at 224x224 px; at 112x112 px every codec
meets the budget for every image, so all subsets equal the full set. Writes::

    outputs/metrics/compliant_subset_<resolution>.csv
    outputs/metrics/compliant_subset_<resolution>.md

Usage
-----
    python -m face1kb.metrics.compliant_subset --resolution 224
"""

import argparse
import csv
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from face1kb import config
from face1kb.data.pairs import load_names, pair_counts
from face1kb.metrics.compute_accuracy import (
    all_model_names,
    embedding_path,
    evaluate_pairs,
)
from face1kb.metrics.statistical_tests import (
    cliffs_delta,
    describe_delta,
    holm_correction,
)

#: File-name-safe identifiers of the proprietary models (see all_model_names).
PROPRIETARY_IDS = tuple(m.replace("-", "_") for m in config.PROPRIETARY_MODELS)

#: Subset kinds that can be requested with ``--subsets``.
SUBSET_KINDS = ("intersection", "relaxed", "per_codec")

#: Subset kinds evaluated by default: the two analyses of the paper.
DEFAULT_SUBSETS = ("intersection", "relaxed")


@dataclass
class Subset:
    """An image subset and the codecs evaluated on it."""

    name: str
    budget: int
    mask: np.ndarray
    codecs: list[str]


def image_sizes(resolution: int, codec: str, names: list[str]) -> np.ndarray:
    """Bitstream size in bytes of every image, in canonical order.

    Images without a bitstream get ``inf`` so that they never meet a budget.
    """
    codec_dir = config.compressed_dir(resolution) / codec
    extension = config.CODEC_EXTENSIONS[codec]
    sizes = np.full(len(names), np.inf)
    for i, name in enumerate(names):
        try:
            sizes[i] = (codec_dir / Path(name).with_suffix(extension)).stat().st_size
        except FileNotFoundError:
            pass
    return sizes


def relaxed_budget(
    required: np.ndarray, budget: int, coverage: float, step: int, max_budget: int
) -> int | None:
    """Smallest budget ``budget + i * step`` that covers *coverage* of images.

    Parameters
    ----------
    required : numpy.ndarray
        Per-image budget needed by the intersection, i.e. the largest
        bitstream of the image over the included codecs.
    budget, step, max_budget : int
        Search grid in bytes.
    coverage : float
        Required fraction of images, e.g. 0.9.

    Returns
    -------
    int or None
        The relaxed budget, or None when no budget up to *max_budget* works.
    """
    target = int(len(required) * coverage)
    for candidate in range(budget, max_budget + 1, step):
        if np.count_nonzero(required <= candidate) >= target:
            return candidate
    return None


def build_subsets(
    sizes: dict[str, np.ndarray], identity_ids: np.ndarray, args: argparse.Namespace
) -> list[Subset]:
    """Construct the requested subsets from the per-codec bitstream sizes.

    Subsets without both mated and non-mated pairs are reported and dropped.
    """
    codecs = list(sizes)
    required = np.max(np.stack([sizes[c] for c in codecs]), axis=0)
    subsets = []
    if "intersection" in args.subsets:
        subsets.append(
            Subset("intersection", args.budget, required <= args.budget, codecs)
        )
    if "relaxed" in args.subsets:
        relaxed = relaxed_budget(
            required,
            args.budget,
            args.relaxed_coverage,
            args.relaxed_step,
            args.relaxed_max_budget,
        )
        if relaxed is None:
            print(
                f"no budget <= {args.relaxed_max_budget} B covers "
                f"{args.relaxed_coverage:.0%} of the images; skipping relaxed"
            )
        else:
            subsets.append(Subset("relaxed", relaxed, required <= relaxed, codecs))
    if "per_codec" in args.subsets:
        for codec in codecs:
            mask = sizes[codec] <= args.budget
            subsets.append(Subset(f"compliant_{codec}", args.budget, mask, [codec]))

    usable = []
    for subset in subsets:
        n_mated, n_nonmated = pair_counts(identity_ids[subset.mask])
        print(
            f"subset {subset.name}: {int(subset.mask.sum())} images "
            f"(<= {subset.budget} B), {n_mated} mated / {n_nonmated} non-mated"
        )
        if n_mated and n_nonmated:
            usable.append(subset)
        else:
            print(f"  skipping {subset.name}: needs mated and non-mated pairs")
    return usable


def evaluate_model(
    resolution: int,
    model: str,
    subsets: list[Subset],
    identity_ids: np.ndarray,
) -> list[dict]:
    """Evaluate one model on every subset: baseline plus each subset codec."""
    original = np.load(embedding_path(resolution, model, "original"))
    rows = []
    baselines = {}  # subsets with an identical mask share the baseline
    baseline_eer = {}
    for subset in subsets:
        key = subset.mask.tobytes()
        if key not in baselines:
            print(f"[{model}] baseline on {subset.name}")
            masked = original[subset.mask]
            baselines[key] = evaluate_pairs(masked, masked, identity_ids[subset.mask])
        baseline_eer[subset.name] = baselines[key]["eer"]
        rows.append({"subset": subset.name, "source": "original", **baselines[key]})

    for codec in dict.fromkeys(c for s in subsets for c in s.codecs):
        codec_path = embedding_path(resolution, model, codec)
        if not codec_path.exists():
            print(f"skipping {model}/{codec}: {codec_path.name} not found")
            continue
        compressed = np.load(codec_path)
        for subset in (s for s in subsets if codec in s.codecs):
            print(f"[{model}] {codec} on {subset.name}")
            row = evaluate_pairs(
                original[subset.mask],
                compressed[subset.mask],
                identity_ids[subset.mask],
            )
            rows.append({"subset": subset.name, "source": codec, **row})

    for row in rows:
        row["model"] = model
        row["baseline_eer"] = baseline_eer[row["subset"]]
        row["eer_decrease"] = row["eer"] - row["baseline_eer"]
    return rows


def codec_tests(eer: np.ndarray, codecs: list[str]) -> dict:
    """Friedman, Kendall's W, mean ranks and pairwise Wilcoxon-Holm tests.

    Parameters
    ----------
    eer : numpy.ndarray
        EER per model (rows) and codec (columns).
    codecs : list of str
        Codec names of the columns.

    Returns
    -------
    dict
        ``n_models``, ``chi2``, ``p``, ``kendalls_w``, ``mean_ranks``
        (codec -> mean rank, 1 = lowest EER) and ``pairwise`` (list of
        ``(codec_a, codec_b, p_raw, p_holm, cliffs_delta)``).
    """
    n_models, k = eer.shape
    chi2, p_friedman = stats.friedmanchisquare(*eer.T)
    pairs = list(itertools.combinations(range(k), 2))
    p_values = []
    for a, b in pairs:
        try:
            _, p = stats.wilcoxon(eer[:, a], eer[:, b])
        except ValueError:  # all differences zero
            p = 1.0
        p_values.append(float(p))
    return {
        "n_models": n_models,
        "chi2": float(chi2),
        "p": float(p_friedman),
        "kendalls_w": float(chi2 / (n_models * (k - 1))),
        "mean_ranks": dict(
            zip(codecs, stats.rankdata(eer, axis=1).mean(axis=0), strict=True)
        ),
        "pairwise": [
            (codecs[a], codecs[b], p, p_holm, cliffs_delta(eer[:, a], eer[:, b]))
            for (a, b), p, p_holm in zip(
                pairs, p_values, holm_correction(p_values), strict=True
            )
        ],
    }


def mean_metric(rows: list[dict], subset: str, codec: str, key: str) -> float:
    """Mean of *key* over the models evaluated with *codec* on *subset*."""
    values = [r[key] for r in rows if r["subset"] == subset and r["source"] == codec]
    return float(np.mean(values)) if values else float("nan")


def write_csv(rows: list[dict], path: Path) -> None:
    """Write the tidy result table (one row per subset x model x source)."""
    lead = [
        "resolution",
        "subset",
        "budget",
        "n_images",
        "n_mated",
        "n_nonmated",
        "model",
        "source",
        "eer",
        "baseline_eer",
        "eer_decrease",
    ]
    extra = [k for k in rows[0] if k not in lead]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=lead + extra)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} result rows to {path}")


def _subset_lines(
    subsets: list[Subset],
    all_sizes: dict[str, np.ndarray],
    codecs: list[str],
    budget: int,
) -> list[str]:
    """Report section: size compliance per codec and subset sizes."""
    n_total = len(next(iter(all_sizes.values())))
    lines = ["## Subsets", "", "| Subset | Budget [B] | Images | Share |"]
    lines.append("|---|---|---|---|")
    for codec, sizes in all_sizes.items():
        n = int(np.count_nonzero(sizes <= budget))
        note = "" if codec in codecs else " (excluded)"
        lines.append(
            f"| compliant_{codec}{note} | {budget} | {n:,} | {n / n_total:.1%} |"
        )
    for subset in subsets:
        if subset.name in ("intersection", "relaxed"):
            n = int(subset.mask.sum())
            lines.append(
                f"| {subset.name} | {subset.budget} | {n:,} | {n / n_total:.1%} |"
            )
    return lines + [""]


def _mean_lines(rows: list[dict], codecs: list[str], names: list[str]) -> list[str]:
    """Report section: mean EER and FRR at FAR 1:1000 over models."""
    lines = ["## Mean over models", ""]
    own = any(r["subset"].startswith("compliant_") for r in rows)
    columns = (["own subset"] if own else []) + names
    for key, label in (("eer", "EER"), ("frr_1:1000", "FRR@1:1000")):
        lines.append(f"| Codec | {' | '.join(f'{label}, {c}' for c in columns)} |")
        lines.append("|---" * (len(columns) + 1) + "|")
        for codec in codecs:
            subsets = ([f"compliant_{codec}"] if own else []) + names
            values = [mean_metric(rows, name, codec, key) for name in subsets]
            cells = ["--" if np.isnan(v) else f"{v:.4f}" for v in values]
            lines.append(f"| {codec} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def _frr_lines(rows: list[dict], codecs: list[str]) -> list[str]:
    """Report section: mean FRR per FAR level on the intersection subset."""
    lines = ["## Mean FRR at FAR operating points (intersection)", ""]
    lines.append(f"| Models | Codec | {' | '.join(config.FAR_LEVEL_LABELS)} |")
    lines.append("|---|---" + "|---" * len(config.FAR_LEVEL_LABELS) + "|")
    for group, members in (("all", None), ("proprietary", PROPRIETARY_IDS)):
        for codec in codecs:
            selected = [
                r
                for r in rows
                if r["subset"] == "intersection"
                and r["source"] == codec
                and (members is None or r["model"] in members)
            ]
            if not selected:
                continue
            frrs = [
                np.mean([r[f"frr_{label}"] for r in selected])
                for label in config.FAR_LEVEL_LABELS
            ]
            lines.append(
                f"| {group} ({len(selected)}) | {codec} | "
                + " | ".join(f"{v:.4f}" for v in frrs)
                + " |"
            )
    return lines + [""]


def _test_lines(rows: list[dict], codecs: list[str]) -> list[str]:
    """Report section: statistical tests on the intersection subset."""
    lines = ["## Statistical tests (intersection, EER per model)", ""]
    eer = {
        (r["model"], r["source"]): r["eer"]
        for r in rows
        if r["subset"] == "intersection"
    }
    models = sorted({model for model, _ in eer})
    complete = [m for m in models if all((m, c) in eer for c in codecs)]
    if len(complete) < 3 or len(codecs) < 3:
        return lines + [
            f"Not enough data (need >= 3 models and >= 3 codecs; found "
            f"{len(complete)} x {len(codecs)}).",
            "",
        ]
    tests = codec_tests(
        np.array([[eer[(m, c)] for c in codecs] for m in complete]), codecs
    )
    lines += [
        f"Blocks: {tests['n_models']} models; treatments: {len(codecs)} codecs.",
        "",
        f"- Friedman chi-squared = {tests['chi2']:.2f}",
        f"- p = {tests['p']:.2e}",
        f"- Kendall's W = {tests['kendalls_w']:.2f}",
        "",
        "| Codec | Mean rank (1 = lowest EER) |",
        "|---|---|",
    ]
    for codec, rank in sorted(tests["mean_ranks"].items(), key=lambda x: x[1]):
        lines.append(f"| {codec} | {rank:.2f} |")
    lines += [
        "",
        "| Codec A | Codec B | p (raw) | p (Holm) | Cliff's delta | Magnitude |",
        "|---|---|---|---|---|---|",
    ]
    for codec_a, codec_b, p, p_holm, delta in tests["pairwise"]:
        lines.append(
            f"| {codec_a} | {codec_b} | {p:.4f} | {p_holm:.4f} "
            f"| {delta:+.2f} | {describe_delta(delta)} |"
        )
    return lines + [""]


def write_markdown(
    rows: list[dict],
    subsets: list[Subset],
    all_sizes: dict[str, np.ndarray],
    codecs: list[str],
    budget: int,
    path: Path,
) -> None:
    """Write the human-readable report."""
    resolution = rows[0]["resolution"]
    excluded = [c for c in all_sizes if c not in codecs]
    names = [s.name for s in subsets if s.name in ("intersection", "relaxed")]
    lines = [
        f"# Size-compliant subsets at {resolution}x{resolution} px",
        "",
        f"Intersection over {', '.join(codecs)} (excluded: "
        f"{', '.join(excluded) or 'none'}). A pair is evaluated iff both of its "
        "images are in the subset; the uncompressed baseline is recomputed on "
        "every subset.",
        "",
    ]
    lines += _subset_lines(subsets, all_sizes, codecs, budget)
    lines += _mean_lines(rows, codecs, names)
    if "intersection" in names:
        lines += _frr_lines(rows, codecs)
        lines += _test_lines(rows, codecs)
    path.write_text("\n".join(lines))
    print(f"Wrote report to {path}")


def main() -> None:
    """Evaluate all models on the size-compliant subsets of one resolution."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        choices=config.RESOLUTIONS,
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=config.MAX_SIZE_BYTES,
        help="Per-image byte budget (default: %(default)s).",
    )
    parser.add_argument(
        "--exclude-codecs",
        nargs="*",
        default=["jpeg2000"],
        choices=config.CODECS,
        help="Codecs left out of the subsets (default: jpeg2000).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        choices=SUBSET_KINDS,
        help="Subset kinds to evaluate (default: intersection relaxed).",
    )
    parser.add_argument("--relaxed-coverage", type=float, default=0.9)
    parser.add_argument("--relaxed-step", type=int, default=64)
    parser.add_argument("--relaxed-max-budget", type=int, default=5120)
    args = parser.parse_args()

    names, identity_ids = load_names()
    all_sizes = {}
    for codec in config.CODECS:
        if not (config.compressed_dir(args.resolution) / codec).is_dir():
            print(f"skipping {codec}: no bitstreams found")
            continue
        all_sizes[codec] = image_sizes(args.resolution, codec, names)
        n_ok = int(np.count_nonzero(all_sizes[codec] <= args.budget))
        print(f"{codec}: {n_ok}/{len(names)} images <= {args.budget} B")
    sizes = {c: s for c, s in all_sizes.items() if c not in args.exclude_codecs}
    if not sizes:
        raise SystemExit("No bitstreams found; run the compression stage first.")

    subsets = build_subsets(sizes, identity_ids, args)
    if not subsets:
        raise SystemExit("No subset with both mated and non-mated pairs.")

    models = all_model_names()
    rows = []
    for model in models:
        if not embedding_path(args.resolution, model, "original").exists():
            print(f"skipping {model}: original embeddings not found")
            continue
        rows.extend(evaluate_model(args.resolution, model, subsets, identity_ids))
    if not rows:
        raise SystemExit("No embeddings found; run the embedding stages first.")

    info = {}
    for subset in subsets:
        n_mated, n_nonmated = pair_counts(identity_ids[subset.mask])
        info[subset.name] = {
            "resolution": args.resolution,
            "budget": subset.budget,
            "n_images": int(subset.mask.sum()),
            "n_mated": n_mated,
            "n_nonmated": n_nonmated,
        }
    for row in rows:
        row.update(info[row["subset"]])
    order = [s.name for s in subsets]
    sources = ["original", *config.CODECS]
    rows.sort(
        key=lambda r: (
            order.index(r["subset"]),
            models.index(r["model"]),
            sources.index(r["source"]),
        )
    )

    out_dir = config.metrics_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"compliant_subset_{args.resolution}"
    write_csv(rows, out_dir / f"{stem}.csv")
    write_markdown(
        rows, subsets, all_sizes, list(sizes), args.budget, out_dir / f"{stem}.md"
    )


if __name__ == "__main__":
    main()
