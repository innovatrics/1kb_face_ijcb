"""Statistical significance of the codec effect on recognition accuracy.

Reproduces the paper's statistical evaluation (Sec. 3.4). Each recognition
model acts as a subject (repeated-measures block), the codecs are the
treatments and the dependent variable is the EER increase over the
uncompressed baseline (``eer_decrease`` column of the accuracy CSVs):

- Friedman test (non-parametric repeated-measures ANOVA) with Kendall's W
  effect size,
- pairwise Wilcoxon signed-rank tests with Holm correction,
- Cliff's delta effect sizes for every codec pair.

Reads ``outputs/metrics/accuracy_<resolution>.csv`` and writes::

    outputs/metrics/statistical_tests_<resolution>.md

Usage
-----
    python -m face1kb.metrics.statistical_tests --resolution 112
"""

import argparse
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from face1kb import config


def load_effect_matrix(resolution: int) -> pd.DataFrame:
    """Return the model x codec matrix of EER increases over the baseline."""
    csv_path = config.metrics_dir() / f"accuracy_{resolution}.csv"
    if not csv_path.exists():
        raise SystemExit(
            f"{csv_path} not found; run face1kb.metrics.compute_accuracy "
            f"first."
        )
    frame = pd.read_csv(csv_path)
    frame = frame[frame["source"] != "original"]
    matrix = frame.pivot(index="model", columns="source",
                         values="eer_decrease")
    return matrix.dropna(axis=0)


def holm_correction(p_values: list[float]) -> list[float]:
    """Holm step-down correction for multiple comparisons."""
    order = np.argsort(p_values)
    m = len(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted_p = min(1.0, (m - rank) * p_values[idx])
        running_max = max(running_max, adjusted_p)
        adjusted[idx] = running_max
    return adjusted.tolist()


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta: P(x > y) - P(x < y) over all cross pairs."""
    greater = np.sum(x[:, None] > y[None, :])
    less = np.sum(x[:, None] < y[None, :])
    return float((greater - less) / (len(x) * len(y)))


def describe_delta(delta: float) -> str:
    """Common-language magnitude bands for Cliff's delta."""
    magnitude = abs(delta)
    if magnitude < 0.147:
        return "negligible"
    if magnitude < 0.33:
        return "small"
    if magnitude < 0.474:
        return "medium"
    return "large"


def main() -> None:
    """Run the statistical evaluation for one resolution."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--resolution",
        type=int,
        required=True,
        choices=config.RESOLUTIONS,
    )
    args = parser.parse_args()

    matrix = load_effect_matrix(args.resolution)
    codecs = [c for c in config.CODECS if c in matrix.columns]
    n_models = len(matrix)
    k = len(codecs)
    if n_models < 3 or k < 3:
        raise SystemExit(
            f"Need at least 3 models and 3 codecs "
            f"(found {n_models} x {k})."
        )

    chi2, p_friedman = stats.friedmanchisquare(
        *[matrix[c].to_numpy() for c in codecs]
    )
    kendalls_w = chi2 / (n_models * (k - 1))

    pairs = list(itertools.combinations(codecs, 2))
    p_values = []
    deltas = []
    for codec_a, codec_b in pairs:
        a = matrix[codec_a].to_numpy()
        b = matrix[codec_b].to_numpy()
        try:
            _, p = stats.wilcoxon(a, b)
        except ValueError:  # all differences zero
            p = 1.0
        p_values.append(float(p))
        deltas.append(cliffs_delta(a, b))
    p_adjusted = holm_correction(p_values)

    out_path = (
        config.metrics_dir() / f"statistical_tests_{args.resolution}.md"
    )
    with open(out_path, "w") as f:
        f.write(
            f"# Statistical evaluation at "
            f"{args.resolution}x{args.resolution} px\n\n"
        )
        f.write(
            f"Blocks: {n_models} recognition models; treatments: {k} "
            f"codecs; variable: EER increase over the uncompressed "
            f"baseline.\n\n"
        )
        f.write("## Friedman test\n\n")
        f.write(f"- chi-squared = {chi2:.2f}\n")
        f.write(f"- p = {p_friedman:.2e}\n")
        f.write(f"- Kendall's W = {kendalls_w:.2f}\n\n")

        f.write("## Pairwise Wilcoxon signed-rank (Holm-corrected)\n\n")
        f.write(
            "| Codec A | Codec B | p (raw) | p (Holm) | Cliff's delta "
            "| Magnitude |\n"
        )
        f.write("|---|---|---|---|---|---|\n")
        for (codec_a, codec_b), p, p_adj, delta in zip(
            pairs, p_values, p_adjusted, deltas
        ):
            f.write(
                f"| {codec_a} | {codec_b} | {p:.4f} | {p_adj:.4f} "
                f"| {delta:+.2f} | {describe_delta(delta)} |\n"
            )

        f.write("\n## Mean EER increase per codec\n\n")
        f.write("| Codec | Mean ΔEER | Median ΔEER |\n|---|---|---|\n")
        for codec in codecs:
            values = matrix[codec]
            f.write(
                f"| {codec} | {values.mean():+.4f} "
                f"| {values.median():+.4f} |\n"
            )

    print(f"Friedman chi2={chi2:.2f}, p={p_friedman:.2e}, W={kendalls_w:.2f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
