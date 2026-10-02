# SPDX-License-Identifier: MIT
"""Matplotlib conventions of the paper figures.

The figures of the paper use matplotlib's default style (``rcParams`` untouched) with
the non-interactive ``Agg`` backend, plus, for the summary figures, a small validated
categorical palette and a "recessive chrome" axis style (hairline grid, muted spines
and ticks). This module collects those constants so every figure generator uses the
same values:

* :func:`use_agg` selects the ``Agg`` backend;
* :data:`BLUE`, :data:`AQUA`, :data:`YELLOW`, :data:`VIOLET`, :data:`RED` -- the
  categorical palette (classical codecs blue, learned standard / neural baselines
  aqua, the face1kb codecs red; four FR anchors blue / aqua / violet / red);
* :data:`INK`, :data:`INK2`, :data:`MUTED`, :data:`GRID`, :data:`BASE` -- text, axis
  and grid greys;
* :func:`seq_blues` -- the sequential colormap of the heatmaps (recompression chain
  EER, significance matrix);
* :func:`style_axes` and :func:`save_figure` -- axis styling and the ``savefig``
  defaults of the summary figures (170 dpi, tight bounding box, white background).

Other figure families keep matplotlib defaults with the per-figure DPI given in
:data:`FIGURE_DPI` and the colormaps in :data:`HEATMAP_CMAPS`.
"""

from __future__ import annotations

from pathlib import Path

#: Categorical palette (light mode).
BLUE, AQUA, YELLOW, VIOLET, RED = (
    "#2a78d6",
    "#1baf7a",
    "#eda100",
    "#4a3aa7",
    "#e34948",
)
#: Text, secondary text, muted ticks, grid lines and axis spines.
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
#: Stops of the sequential blue colormap.
SEQ_BLUES_STOPS: tuple[str, ...] = (
    "#f3f8fe",
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
)
#: Colors of the two datasets in bar charts (face image quality figure).
DATASET_COLORS: dict[str, str] = {"colorferet": "#3b6ea5", "kk": "#d98c3f"}
#: Colormaps of the heatmap figures.
HEATMAP_CMAPS: dict[str, str] = {
    "fnmr": "RdYlGn_r",  # FNMR heatmaps (lower is better)
    "fit_rate": "RdYlGn",  # budget compliance (higher is better)
    "many_lines": "tab20",  # line plots with many codecs
}
#: Typical ``savefig`` resolution per figure family (generators that reproduce a
#: specific paper figure keep that figure's own value).
FIGURE_DPI: dict[str, int] = {
    "summary": 170,  # interpretation figures (quality vs identity, fairness, ...)
    "rate_eer": 130,
    "frr_far": 120,
    "file_size": 120,
    "speed": 130,
    "h4_curve": 130,
    "fiq": 140,
    "visual_grid": 130,
}

#: Display names of the FR anchor matchers.
ANCHOR_LABELS: dict[str, str] = {
    "arcface_antelopev2": "ArcFace",
    "lvface_l": "LVFace-L",
    "topofr_r100": "TopoFR-R100",
    "edgeface_xs": "EdgeFace-XS",
}
#: Display names of the codecs in figures.
CODEC_LABELS: dict[str, str] = {
    "jpeg": "JPEG",
    "jpeg2000": "JPEG 2000",
    "webp": "WebP",
    "jpeg_xl": "JPEG XL",
    "avif": "AVIF",
    "heif": "HEIF",
    "jpeg_fzt": "JPEG-FzT",
    "jpeg_ai": "JPEG-AI",
    "neural_bmshj2018": "bmshj2018",
    "neural_mbt2018_mean": "mbt2018",
    "ours_fast": "Ours-FAST",
    "ours_accurate": "Ours-ACCURATE",
    "aligned": "aligned (ref)",
}
#: Codec groups (color coding of the summary figures).
CLASSICAL: tuple[str, ...] = (
    "jpeg",
    "jpeg2000",
    "webp",
    "jpeg_xl",
    "avif",
    "heif",
    "jpeg_fzt",
)
NEURAL_STD: tuple[str, ...] = ("jpeg_ai", "neural_bmshj2018", "neural_mbt2018_mean")
OURS: tuple[str, ...] = ("ours_fast", "ours_accurate")


def use_agg() -> None:
    """Select the non-interactive ``Agg`` backend (call before importing pyplot)."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")


def codec_color(codec: str) -> str:
    """Group color of a codec: face1kb red, learned standard aqua, classical blue."""
    if codec in OURS:
        return RED
    if codec in NEURAL_STD:
        return AQUA
    return BLUE


def seq_blues():
    """Return the sequential blue colormap of the heatmaps."""
    from matplotlib.colors import LinearSegmentedColormap  # noqa: PLC0415

    return LinearSegmentedColormap.from_list("seq_blues", list(SEQ_BLUES_STOPS))


def style_axes(ax, grid_axis: str | None = "both") -> None:
    """Apply the recessive-chrome style: hairline grid, muted spines and ticks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, width=0.8)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def save_figure(
    fig,
    path: str | Path,
    dpi: int = FIGURE_DPI["summary"],
    tight: bool = True,
    close: bool = True,
) -> Path:
    """Save ``fig`` with the summary-figure defaults (white background).

    Parent folders are created. Returns the path.
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(path, dpi=dpi, facecolor="white", **kw)
    if close:
        plt.close(fig)
    return path
