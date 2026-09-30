# SPDX-License-Identifier: MIT
r"""Render the summary figures of the paper from the aggregate result files.

CPU only; reads the aggregate CSVs of an input root (default ``FACE1KB_OUTPUT_ROOT``;
``--from-results`` reads the shipped ``results/``) and writes PNGs (170 dpi) to
``--out-dir`` (default ``OUTPUT_ROOT/figures``) plus the figure-derived scalars to
``--stats-out`` (default ``OUTPUT_ROOT/report_summary/derived_stats.json``).

Figures (paper figure in brackets) and inputs, relative to the input root:

* ``quality_vs_identity.png`` [Fig. 22] -- PSNR against verification EER per codec,
  Color FERET, ArcFace, 112 px, both budgets, with the Spearman rho in the panel
  titles. ``accuracy/metrics.csv`` (compressed rows, no crop-variant suffix) and
  ``quality/quality_summary.csv`` (cells with at least 100 crops).
* ``fairness_disparity.png`` / ``fairness_disparity_512.png`` [Figs. 40, 41] --
  max-min subgroup EER disparity per codec at 112 px: AI-Solutions-KK Monk skin tone
  over the uniform MST5/6/7/9/10 basis for the four anchor matchers
  (``fairness/fairness_kk.csv``), and Color FERET ethnicity and pose under
  EdgeFace-XS (``fairness/fairness_disparity_colorferet.csv``).
* ``recompression_heatmap.png`` [Fig. 42] -- chain EER of the six classical codecs
  and Ours-ACCURATE, both datasets and budgets
  (``recompression/recompression_<dataset>_<budget>.csv``).
* ``sanitization_residual.png`` [Fig. 43] -- HFC-attack residual identity-cosine gap
  against the attack strength, mean over the four anchors
  (``adversarial/sanitization.csv``).
* ``significance_matrix.png`` [Fig. 44] -- pairwise McNemar chi^2 with the
  Benjamini-Hochberg decisions of one cell (Color FERET, EdgeFace-XS, 112 px,
  1024 B), taken from ``accuracy/significance_colorferet.csv``.
* ``codec_mean_rank.png`` [Fig. 45] -- mean EER rank of each codec across the
  matcher roster, with the Friedman and Kendall statistics
  (``accuracy/posthoc_scalars.txt``).

``derived_stats.json`` holds the Spearman rho / p / n of the quality-vs-identity
panels and, per anchor, the KK skin-tone disparities of the aligned reference,
WebP, Ours-ACCURATE and JPEG 2000 at 1024 B with the JPEG 2000 amplification factor.

Examples
--------
    python experiments/figures/render_figures.py --from-results
    python experiments/figures/render_figures.py --only significance_matrix
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from face1kb import config
from face1kb.eval.embeddings import ANCHOR_MODELS
from face1kb.eval.fairness import MONK_UNIFORM_BASIS, disparity_pp, subgroup_spread_pp
from face1kb.report import style
from face1kb.report.style import (
    ANCHOR_LABELS,
    AQUA,
    BASE,
    BLUE,
    CODEC_LABELS,
    GRID,
    INK,
    INK2,
    MUTED,
    NEURAL_STD,
    OURS,
    RED,
    VIOLET,
    save_figure,
    style_axes,
)

style.use_agg()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

log = logging.getLogger("render_figures")

#: Quality cells with fewer crops are pilot runs and are left out of the scatter.
MIN_QUALITY_N = 100
#: The cell of the significance matrix: dataset, matcher, resolution, budget.
SIGNIFICANCE_CELL = ("colorferet", "edgeface_xs", 112, 1024)
#: Codec order of the fairness figure (JPEG-AI is inserted where present).
FAIRNESS_CODEC_ORDER: tuple[str, ...] = (
    "aligned",
    "webp",
    "avif",
    "heif",
    "jpeg_xl",
    "jpeg",
    "jpeg_fzt",
    "ours_accurate",
    "ours_fast",
    "jpeg2000",
)
#: Codecs of the recompression heatmap (rows = first codec, columns = second codec).
RECOMPRESSION_CODECS: tuple[str, ...] = (
    "jpeg",
    "jpeg2000",
    "webp",
    "jpeg_xl",
    "avif",
    "heif",
    "ours_accurate",
)
#: Highlighted codecs of the sanitization figure: color and label.
SANITIZATION_HIGHLIGHT: dict[str, tuple[str, str]] = {
    "jpeg2000": (BLUE, "JPEG 2000"),
    "webp": (AQUA, "WebP"),
    "ours_accurate": (RED, "Ours-ACCURATE"),
    "ours_fast": (VIOLET, "Ours-FAST"),
}
DATASET_TITLES = {"colorferet": "ColorFERET", "kk": "AI-Solutions-KK"}

#: Hand-placed label offsets (dx, dy, ha) in points of the quality-vs-identity
#: scatter, tuned to the published data; other points get (0, 6, "center").
QUALITY_LABEL_OFFSETS: dict[tuple[int, str], tuple[int, int, str]] = {
    (1024, "jpeg_fzt"): (0, -12, "center"),
    (1024, "ours_fast"): (0, 7, "center"),
    (1024, "ours_accurate"): (0, -12, "center"),
    (1024, "jpeg_xl"): (-4, 7, "right"),
    (1024, "heif"): (-8, 3, "right"),
    (1024, "neural_bmshj2018"): (-9, -3, "right"),
    (1024, "avif"): (0, 9, "center"),
    (1024, "jpeg_ai"): (0, -13, "center"),
    (1024, "webp"): (9, -3, "left"),
    (1024, "neural_mbt2018_mean"): (-8, 2, "right"),
    (512, "jpeg"): (8, 0, "left"),
    (512, "jpeg2000"): (9, -9, "left"),
    (512, "heif"): (8, 2, "left"),
    (512, "jpeg_fzt"): (-8, -2, "right"),
    (512, "avif"): (8, 0, "left"),
    (512, "jpeg_xl"): (-8, -3, "right"),
    (512, "ours_accurate"): (-8, -9, "right"),
    (512, "ours_fast"): (-8, 3, "right"),
    (512, "webp"): (-8, -2, "right"),
    (512, "neural_bmshj2018"): (7, 2, "left"),
    (512, "neural_mbt2018_mean"): (7, 3, "left"),
    (512, "jpeg_ai"): (5, -10, "left"),
}

FIGURES: tuple[str, ...] = (
    "quality_vs_identity",
    "fairness_disparity",
    "fairness_disparity_512",
    "recompression_heatmap",
    "sanitization_residual",
    "significance_matrix",
    "codec_mean_rank",
)


def _family_color(codec: str) -> str:
    """Color of a codec family: Ours red, learned standard / baselines aqua."""
    return RED if codec in OURS else AQUA if codec in NEURAL_STD else BLUE


# ---------------------------------------------------------------- quality vs identity
def fig_quality_vs_identity(root: Path, out: Path) -> dict:
    """PSNR vs EER scatter (Color FERET, ArcFace, 112 px); returns Spearman stats."""
    met = pd.read_csv(root / "accuracy" / "metrics.csv")
    qua = pd.read_csv(root / "quality" / "quality_summary.csv")
    met = met[
        (met.dataset == "colorferet")
        & (met.model == "arcface_antelopev2")
        & (met.res == 112)
        & (met.kind == "compressed")
        & (met.suffix.isna())
    ]
    qua = qua[
        (qua.dataset == "colorferet") & (qua.res == 112) & (qua.n >= MIN_QUALITY_N)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.3), sharey=True)
    stats = {}
    for ax, budget in zip(axes, (1024, 512)):
        m = met[met.budget == budget][["codec", "eer"]].dropna()
        q = qua[qua.budget == budget][["codec", "psnr"]]
        df = m.merge(q, on="codec")
        df["eer_pct"] = df.eer * 100.0
        rho, p = spearmanr(df.psnr, df.eer_pct)
        stats[budget] = {
            "spearman_rho": round(float(rho), 3),
            "p": float(p),
            "n": len(df),
        }
        for _, r in df.iterrows():
            y = max(r.eer_pct, 2e-3)
            ax.scatter(
                r.psnr,
                y,
                s=42,
                color=_family_color(r.codec),
                zorder=3,
                edgecolor="white",
                linewidth=0.8,
            )
            dx, dy, ha = QUALITY_LABEL_OFFSETS.get((budget, r.codec), (0, 6, "center"))
            label = CODEC_LABELS[r.codec]
            if r.eer_pct == 0.0:
                label += " (0.00)"
            ax.annotate(
                label,
                (r.psnr, y),
                textcoords="offset points",
                xytext=(dx, dy),
                ha=ha,
                fontsize=6.8,
                color=INK2,
            )
        ax.set_yscale("log")
        ax.set_xlabel("PSNR (dB)  → better", fontsize=8.5, color=INK2)
        ax.set_title(
            f"{budget} B    (Spearman ρ = {rho:+.2f})",
            fontsize=9.5,
            color=INK,
            fontweight="bold",
        )
        style_axes(ax)
    axes[0].set_ylabel("verification EER (%, log)  ↓ better", fontsize=8.5, color=INK2)
    families = (
        (BLUE, "classical"),
        (AQUA, "learned standard/baseline"),
        (RED, "Ours"),
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", color=col, markersize=7, label=name)
        for col, name in families
    ]
    # upper right of the 1024 B panel is the only region with no codec in it
    axes[0].legend(handles=handles, fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle(
        "Pixel fidelity does not predict identity — ColorFERET, ArcFace, 112 px",
        fontsize=10.5,
        color=INK,
        fontweight="bold",
        y=1.02,
    )
    save_figure(fig, out / "quality_vs_identity.png")
    return stats


# ---------------------------------------------------------------- fairness
def _present(frame: pd.DataFrame, codecs) -> list[str]:
    """``codecs`` in their order, restricted to those with rows in ``frame``."""
    have = set(frame.codec)
    return [c for c in codecs if c in have]


def fig_fairness_disparity(root: Path, out: Path, budget: int = 1024) -> dict:
    """Max-min subgroup EER disparity per codec (dot plot), KK and Color FERET.

    The KK panel uses the uniform Monk basis MST5/6/7/9/10 (MST8 left out), because
    the classical codecs lose the MST8 cell under compression while the Ours
    variants keep it, so the disparity CSV would compare spreads over different
    subgroup sets. Returns the JPEG 2000 amplification per anchor (1024 B figure).
    """
    fair = pd.read_csv(root / "fairness" / "fairness_kk.csv")
    cf = pd.read_csv(root / "fairness" / "fairness_disparity_colorferet.csv")

    per = fair[
        (fair.res == 112)
        & (fair.attribute == "skin_tone")
        & (fair.subgroup.isin(list(MONK_UNIFORM_BASIS)))
        & ((fair.budget == budget) | (fair.codec == "aligned"))
    ]
    kk_codecs = _present(per, FAIRNESS_CODEC_ORDER)
    kk = {
        (a, c): subgroup_spread_pp(fair, model=a, codec=c, budget=budget)
        for a in ANCHOR_MODELS
        for c in kk_codecs
    }
    anchor_colors = dict(zip(ANCHOR_MODELS, (BLUE, AQUA, VIOLET, RED)))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.9, 3.6), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    ypos = {c: len(kk_codecs) - 1 - i for i, c in enumerate(kk_codecs)}
    for anchor in ANCHOR_MODELS:
        ax1.scatter(
            [kk[anchor, c] for c in kk_codecs],
            [ypos[c] for c in kk_codecs],
            s=34,
            color=anchor_colors[anchor],
            label=ANCHOR_LABELS[anchor],
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )
    for c in kk_codecs:
        vals = [kk[a, c] for a in ANCHOR_MODELS if not np.isnan(kk[a, c])]
        ax1.plot(
            [min(vals), max(vals)],
            [ypos[c]] * 2,
            color=GRID,
            linewidth=1.4,
            zorder=1,
        )
    ax1.set_yticks(list(ypos.values()))
    ax1.set_yticklabels([CODEC_LABELS[c] for c in kk_codecs], fontsize=8)
    ax1.set_xlabel("Monk skin-tone EER disparity Δ (pp)", fontsize=8.5, color=INK2)
    ax1.set_title(
        f"AI-Solutions-KK · 112 px / {budget} B · 4 anchors",
        fontsize=9.5,
        color=INK,
        fontweight="bold",
    )
    ax1.legend(
        fontsize=7,
        frameon=False,
        loc="upper right",
        handletextpad=0.1,
        borderaxespad=0.2,
    )
    style_axes(ax1, grid_axis="x")

    cf_model = "edgeface_xs"
    cf = cf[
        (cf.res == 112)
        & (cf.model == cf_model)
        & ((cf.budget == budget) | (cf.codec == "aligned"))
    ]
    cf_codecs = _present(cf[cf.attribute == "skin_tone"], FAIRNESS_CODEC_ORDER)
    if "jpeg_ai" in set(cf.codec) and "jpeg_ai" not in cf_codecs:
        cf_codecs.insert(4, "jpeg_ai")
    ypos2 = {c: len(cf_codecs) - 1 - i for i, c in enumerate(cf_codecs)}
    for att, (label, color) in {
        "skin_tone": ("ethnicity", BLUE),
        "pose": ("pose", AQUA),
    }.items():
        xs = [
            disparity_pp(cf, model=cf_model, codec=c, budget=budget, attribute=att)[0]
            for c in cf_codecs
        ]
        ax2.scatter(
            xs,
            [ypos2[c] for c in cf_codecs],
            s=34,
            color=color,
            label=label,
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )
    ax2.set_yticks(list(ypos2.values()))
    ax2.set_yticklabels([CODEC_LABELS[c] for c in cf_codecs], fontsize=8)
    ax2.set_xlabel("EER disparity Δ (pp)", fontsize=8.5, color=INK2)
    ax2.set_title(
        f"ColorFERET · 112 px / {budget} B · EdgeFace-XS",
        fontsize=9.5,
        color=INK,
        fontweight="bold",
    )
    ax2.legend(
        fontsize=7,
        frameon=False,
        loc="upper right",
        handletextpad=0.1,
        borderaxespad=0.2,
    )
    style_axes(ax2, grid_axis="x")

    fig.suptitle(
        "Compression widens the between-group EER gap — "
        "JPEG 2000 by an order of magnitude",
        fontsize=10.5,
        color=INK,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    name = (
        "fairness_disparity.png"
        if budget == 1024
        else f"fairness_disparity_{budget}.png"
    )
    save_figure(fig, out / name)

    amp: dict = {"basis": "MST5/6/7/9/10 (MST8 excluded; uniform across codecs)"}
    for a in ANCHOR_MODELS:
        # NumPy rounding, as in the published derived_stats.json
        base, j2k = np.float64(kk[a, "aligned"]), np.float64(kk[a, "jpeg2000"])
        amp[a] = {
            "base_pp": float(round(base, 2)),
            "webp_pp": float(round(np.float64(kk[a, "webp"]), 2)),
            "ours_accurate_pp": float(round(np.float64(kk[a, "ours_accurate"]), 2)),
            "jpeg2000_pp": float(round(j2k, 2)),
            "jpeg2000_amplification": float(round(j2k / base, 1)),
        }
    return amp


# ---------------------------------------------------------------- recompression
def fig_recompression_heatmap(root: Path, out: Path) -> None:
    """Chain-EER heatmaps (6 classical codecs + Ours-ACCURATE), datasets x budgets."""
    order = list(RECOMPRESSION_CODECS)
    short = {**CODEC_LABELS, "ours_accurate": "Ours-ACC"}
    labels = [short[c] for c in order]
    nc = len(order)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.7))
    fig.subplots_adjust(wspace=0.42, hspace=0.40)
    norm = LogNorm(vmin=0.5, vmax=50.0)
    cmap = style.seq_blues()
    cells = [(ds, b) for ds in ("colorferet", "kk") for b in (1024, 512)]
    for ax, (ds, budget) in zip(axes.flat, cells):
        df = pd.read_csv(root / "recompression" / f"recompression_{ds}_{budget}.csv")
        mat = (
            df.pivot(
                index="source_codec", columns="second_codec", values="eer"
            ).reindex(index=order, columns=order)
            * 100.0
        )
        im = ax.imshow(mat.values, cmap=cmap, norm=norm)
        for i in range(nc):
            for j in range(nc):
                v = mat.values[i, j]
                if np.isnan(v):
                    continue
                ax.text(
                    j,
                    i,
                    f"{v:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6.8,
                    color="white" if v > 8 else INK,
                    fontweight="bold" if v >= 7.5 else "normal",
                )
        ax.set_xticks(range(nc), labels, fontsize=6.6, rotation=35, ha="right")
        ax.set_yticks(range(nc), labels, fontsize=6.6)
        ax.tick_params(colors=MUTED, length=0)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(INK2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(
            f"{DATASET_TITLES[ds]} · {budget} B",
            fontsize=9.5,
            color=INK,
            fontweight="bold",
        )
    axes[0, 0].set_ylabel("source (first) codec", fontsize=8.5, color=INK2)
    axes[1, 0].set_ylabel("source (first) codec", fontsize=8.5, color=INK2)
    axes[1, 0].set_xlabel("second codec", fontsize=8.5, color=INK2)
    axes[1, 1].set_xlabel("second codec", fontsize=8.5, color=INK2)
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("chain EER (%, log scale)", fontsize=8.5, color=INK2)
    cbar.ax.tick_params(labelsize=7.5, colors=MUTED)
    fig.suptitle(
        "Recompression chain EER at 112 px — dark cells are the chains to forbid",
        fontsize=10.5,
        color=INK,
        fontweight="bold",
        y=0.98,
    )
    save_figure(fig, out / "recompression_heatmap.png")


# ---------------------------------------------------------------- sanitization
def fig_sanitization_residual(root: Path, out: Path) -> None:
    """HFC residual vs attack strength per codec, datasets x budgets."""
    sa = pd.read_csv(root / "adversarial" / "sanitization.csv")
    sa = sa[(sa.attack == "hfc") & (sa.model.isin(list(ANCHOR_MODELS)))]
    agg = (
        sa.groupby(["dataset", "codec", "eps", "budget"]).residual.mean().reset_index()
    )
    hl = SANITIZATION_HIGHLIGHT
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 5.6), sharex=True, sharey=True)
    for row, ds in enumerate(("colorferet", "kk")):
        for col, budget in enumerate((1024, 512)):
            ax = axes[row, col]
            cell = agg[(agg.dataset == ds) & (agg.budget == budget)]
            # labels of the highlighted codecs are offset by the rank of the curve's
            # end point in this panel (alternating left/right), so they never collide
            ends = sorted(
                (
                    (
                        codec,
                        cell[cell.codec == codec].sort_values("eps").residual.iloc[-1],
                    )
                    for codec in hl
                    if not cell[cell.codec == codec].empty
                ),
                key=lambda t: -t[1],
            )
            label_pos = {
                codec: (5 + 27 * (i % 2), 10 - 13 * i)
                for i, (codec, _) in enumerate(ends)
            }
            for codec in sorted(cell.codec.unique()):
                cur = cell[cell.codec == codec].sort_values("eps")
                if codec in hl:
                    color, lw, z = hl[codec][0], 2.0, 3
                else:
                    color, lw, z = BASE, 1.2, 2
                ax.plot(
                    cur.eps,
                    cur.residual,
                    color=color,
                    linewidth=lw,
                    zorder=z,
                    marker="o",
                    markersize=4 if codec in hl else 2.5,
                )
                if codec in hl:
                    ax.annotate(
                        CODEC_LABELS[codec],
                        (cur.eps.iloc[-1], cur.residual.iloc[-1]),
                        textcoords="offset points",
                        xytext=label_pos[codec],
                        fontsize=6.6,
                        color=color,
                    )
            ax.axhline(0, color=BASE, linewidth=0.8, zorder=1)
            ax.set_title(
                f"{DATASET_TITLES[ds]} · {budget} B",
                fontsize=9.5,
                color=INK,
                fontweight="bold",
            )
            ax.set_xticks([0.03, 0.06, 0.10])
            style_axes(ax, grid_axis="y")
            ax.set_xlim(0.025, 0.125)
    for ax in axes[1]:
        ax.set_xlabel("attack strength ε (ℓ∞)", fontsize=8.5, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("residual id-cosine gap ↓", fontsize=8.5, color=INK2)
    fig.suptitle(
        "HFC-attack residual after compression — tighter budgets sanitize harder;\n"
        "JPEG 2000 sanitizes best, the learned codecs least (grey: remaining codecs)",
        fontsize=10,
        color=INK,
        fontweight="bold",
        y=1.0,
    )
    fig.tight_layout()
    save_figure(fig, out / "sanitization_residual.png")

    check = agg[
        (agg.dataset == "colorferet")
        & (agg.codec == "jpeg2000")
        & (agg.eps == 0.06)
        & (agg.budget == 1024)
    ].residual
    if len(check):
        log.info(
            "Color FERET JPEG 2000 eps=0.06 / 1024 B mean residual = %.3f",
            check.iloc[0],
        )


# ---------------------------------------------------------------- significance
def fig_significance_matrix(root: Path, out: Path) -> None:
    """Pairwise McNemar matrix of one cell (:data:`SIGNIFICANCE_CELL`)."""
    ds, model, res, budget = SIGNIFICANCE_CELL
    sig = pd.read_csv(root / "accuracy" / f"significance_{ds}.csv")
    sig = sig[(sig.model == model) & (sig.res == res) & (sig.budget == budget)]
    if sig.empty:
        raise SystemExit(f"no significance rows for {SIGNIFICANCE_CELL}")
    acc = {}
    for _, r in sig.iterrows():
        acc[r.codec_a] = r.acc_a
        acc[r.codec_b] = r.acc_b
    conds = sorted(set(sig.codec_a) | set(sig.codec_b))
    conds = sorted(conds, key=lambda c: -acc[c])
    n = len(conds)
    chi = np.full((n, n), np.nan)
    signif = np.zeros((n, n), dtype=bool)
    for _, r in sig.iterrows():
        i, j = conds.index(r.codec_a), conds.index(r.codec_b)
        i, j = min(i, j), max(i, j)
        chi[j, i] = r.mcnemar_chi2
        signif[j, i] = bool(r.significant)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    disp = np.log10(1 + np.nan_to_num(chi, nan=0.0))
    masked = np.ma.masked_where(np.isnan(chi), disp)
    im = ax.imshow(masked, cmap=style.seq_blues(), vmin=0, vmax=5.2)
    for i in range(n):
        for j in range(i):
            v = chi[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:,.0f}" if v >= 100 else f"{v:.1f}" if v >= 1 else f"{v:.2f}"
            color = "white" if disp[i, j] > 3.1 else INK
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.6, color=color)
            if not signif[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=RED,
                        linewidth=1.8,
                    )
                )
                ax.text(
                    j,
                    i + 0.32,
                    "n.s.",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=RED,
                    fontweight="bold",
                )
    labels = [CODEC_LABELS.get(c, c) for c in conds]
    ax.set_xticks(range(n), labels, fontsize=7.5, rotation=35, ha="right")
    ax.set_yticks(range(n), labels, fontsize=7.5)
    ax.tick_params(colors=MUTED, length=0)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, n - 1.5)
    ax.set_ylim(n - 0.5, 0.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(
        "McNemar $\\chi^2$ (log$_{10}$(1+$\\chi^2$))", fontsize=8, color=INK2
    )
    cbar.ax.tick_params(labelsize=7.5, colors=MUTED)
    ax.set_title(
        "Pairwise McNemar $\\chi^2$ — ColorFERET, EdgeFace-XS, 112 px / 1024 B\n"
        "(ordered by accuracy; red boxes = not significant after BH at q=0.05)",
        fontsize=9.5,
        color=INK,
        fontweight="bold",
    )
    save_figure(fig, out / "significance_matrix.png")


# ---------------------------------------------------------------- mean rank
def read_posthoc_scalars(path: Path) -> tuple[dict[str, str], dict[str, float]]:
    """Parse ``posthoc_scalars.txt``: header statistics and the codec mean ranks.

    The header line is ``n=<matchers> k=<codecs> friedman_chi2=... friedman_p=...
    kendall_w=...``; the second line is ``mean_rank: <label>=<rank>, ...``. The
    header values are returned as the strings written in the file.
    """
    text = path.read_text()
    head = re.search(
        r"n=(\d+) k=(\d+) friedman_chi2=([\d.]+)"
        r" friedman_p=([\deE.+-]+) kendall_w=([\d.]+)",
        text,
    )
    if head is None or "mean_rank:" not in text:
        raise ValueError(f"unexpected posthoc scalars format in {path}")
    keys = ("n", "k", "friedman_chi2", "friedman_p", "kendall_w")
    ranks = dict(re.findall(r"([\w ()-]+)=(\d+\.\d+)", text.split("mean_rank:")[1]))
    return dict(zip(keys, head.groups())), {
        k.strip(): float(v) for k, v in ranks.items()
    }


def fig_codec_mean_rank(root: Path, out: Path) -> None:
    """Cross-matcher mean-rank dot plot from the post-hoc scalars."""
    head, ranks = read_posthoc_scalars(root / "accuracy" / "posthoc_scalars.txt")
    items = sorted(ranks.items(), key=lambda kv: kv[1])

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, (name, rank) in enumerate(items):
        y = len(items) - 1 - i
        color = RED if name.startswith("Ours") else BLUE
        ax.plot([1, rank], [y, y], color=GRID, linewidth=1.2, zorder=1)
        ax.scatter(
            [rank], [y], s=46, color=color, zorder=3, edgecolor="white", linewidth=0.8
        )
        ax.annotate(
            f"{rank:.1f}",
            (rank, y),
            textcoords="offset points",
            xytext=(8, -2.5),
            fontsize=7.5,
            color=INK2,
        )
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([name for name, _ in reversed(items)], fontsize=8.5)
    n, k = head["n"], head["k"]
    ax.set_xlabel(
        f"mean EER rank across {n} matchers (1 = best)", fontsize=8.5, color=INK2
    )
    ax.set_xlim(0.5, int(k) + 0.3)
    style_axes(ax, grid_axis="x")
    ax.set_title(
        f"Codec mean rank over the matcher roster — 112 px / 1024 B\n"
        f"Friedman $\\chi^2$={head['friedman_chi2']} "
        f"(p≈{float(head['friedman_p']):.0e}); Kendall's W={head['kendall_w']} "
        f"({n} matchers × {k} codecs)",
        fontsize=9.5,
        color=INK,
        fontweight="bold",
    )
    save_figure(fig, out / "codec_mean_rank.png")


# ---------------------------------------------------------------- main
def render(root: Path, out: Path, only: set[str] | None = None) -> dict:
    """Render the figures in ``only`` (default: all) and return the derived stats.

    The derived stats contain ``quality_vs_identity`` and ``fairness_amplification``
    when the corresponding figures were rendered.
    """
    want = set(FIGURES) if only is None else only
    derived: dict = {}
    if "quality_vs_identity" in want:
        derived["quality_vs_identity"] = fig_quality_vs_identity(root, out)
    if "fairness_disparity" in want:
        derived["fairness_amplification"] = fig_fairness_disparity(root, out, 1024)
    if "fairness_disparity_512" in want:
        fig_fairness_disparity(root, out, 512)
    if "recompression_heatmap" in want:
        fig_recompression_heatmap(root, out)
    if "sanitization_residual" in want:
        fig_sanitization_residual(root, out)
    if "significance_matrix" in want:
        fig_significance_matrix(root, out)
    if "codec_mean_rank" in want:
        fig_codec_mean_rank(root, out)
    return derived


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument(
        "--input-root", type=Path, default=None, help="default: OUTPUT_ROOT"
    )
    src.add_argument(
        "--from-results",
        action="store_true",
        help="read the shipped results/ folder",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None, help="default: OUTPUT_ROOT/figures"
    )
    ap.add_argument(
        "--stats-out",
        type=Path,
        default=None,
        help="default: OUTPUT_ROOT/report_summary/derived_stats.json",
    )
    ap.add_argument(
        "--only",
        default="",
        help=f"comma-separated subset of: {','.join(FIGURES)}",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    root = (
        config.RESULTS_ROOT
        if args.from_results
        else (args.input_root or config.OUTPUT_ROOT)
    )
    out = args.out_dir or config.output_dir("figures")
    only = {s for s in args.only.split(",") if s} or None
    if only and only - set(FIGURES):
        ap.error(f"unknown figures: {sorted(only - set(FIGURES))}")
    derived = render(Path(root), Path(out), only)
    log.info("figures written to %s", out)
    if only is None:
        stats = args.stats_out or config.output_dir(
            "report_summary", "derived_stats.json"
        )
        stats.parent.mkdir(parents=True, exist_ok=True)
        with open(stats, "w") as fh:
            json.dump(derived, fh, indent=2)
            fh.write("\n")
        log.info("wrote %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
