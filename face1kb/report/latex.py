# SPDX-License-Identifier: MIT
r"""Shared LaTeX table helpers: best/worst shading with the paper's emphasis rules.

Convention of the paper tables
------------------------------
The best cell of a ranked column is filled ``\cellcolor{green!25}`` and the worst
``\cellcolor{red!22}``. Because the two fills are close in luminance, a redundant,
non-colour marker is added inside the shaded cell: **best is bold**
(``\textbf{...}``) and **worst is underlined** (``\underline{...}``), so the ranking
survives greyscale printing and colour-blind reading. The marker wraps the whole cell
content after the ``\cellcolor`` command, e.g.::

    \cellcolor{green!25}\textbf{0.947}      \cellcolor{red!22}\underline{37.82}

Use :func:`shade_best` / :func:`shade_worst` when writing a cell, or run
:func:`add_emphasis` over a finished table that carries colour only; both produce the
same text. :func:`add_emphasis` is idempotent.

Ranking rules of :func:`shade_table` (used for tables assembled from CSVs):

* ranking is per numeric column and per *block* -- blocks are separated by any rule
  or ``\multicolumn`` line, and the header before the first ``\midrule`` is skipped;
* a cell's value is its leading number (``"1.4 [0.8,6.0]"`` -> 1.4); ``--``, ``---``
  and ``n/a`` cells are not ranked;
* reference rows (e.g. the uncompressed ``aligned`` / ``base`` row) are excluded
  from the ranking;
* a column whose values all tie is left unshaded; every cell equal to the best (or
  worst) value is shaded.

Exceptions in the published tables (arXiv v1)
---------------------------------------------
* The frr_far and model_effect tables (``frr_far_<ds>_<budget>[_224]``,
  ``frr_far_summary``, ``model_effect_<ds>_1024[_224]``; :data:`NO_EMPHASIS_TABLES`)
  are shaded with colour only, without the bold/underline marker: their generator was
  re-run after the emphasis pass. Pass ``emphasis=False`` (or use
  :func:`emphasis_for`) to reproduce them byte for byte.
* ``fairness_disparity_512`` is published without any shading
  (:data:`UNSHADED_TABLES`), although its caption announces best/worst shading.
* The hand-maintained tables ``best_config``, ``fmr_fairness_cf`` and
  ``disparity_ci_kk`` were shaded with :func:`shade_table` rules (lower is better;
  the ``base``/``aligned`` rows are references) and then emphasised.
* ``jpegai_speed`` is hand-shaded per column, with mixed directions that
  :func:`shade_table` (one direction per table) cannot reproduce: the time columns
  (CPU and GPU decode, GPU encode, ms) rank lower-is-better and the quality columns
  (PSNR, SSIM, identity cosine) higher-is-better. Only :data:`UNSHADE_COLUMNS` is
  applied to it: its byte-count column (column index 8) carries no shading, because
  the values span 1108-1109 B, a spread that cannot support a ranking.
  :func:`finalize_table` therefore reproduces the published table from the
  hand-shaded one.
"""

from __future__ import annotations

import re

#: Fill colour of the best cell.
BEST_FILL = "green!25"
#: Fill colour of the worst cell.
WORST_FILL = "red!22"
#: (fill, wrapping macro): bold for best, underline for worst.
EMPHASIS_RULES: tuple[tuple[str, str], ...] = (
    (BEST_FILL, "textbf"),
    (WORST_FILL, "underline"),
)

#: Tables published with colour-only shading (no bold/underline) in arXiv v1.
NO_EMPHASIS_TABLES: frozenset[str] = frozenset(
    [
        f"frr_far_{ds}_{b}{r}"
        for ds in ("colorferet", "kk")
        for b in (1024, 512)
        for r in ("", "_224")
    ]
    + ["frr_far_summary"]
    + [
        f"model_effect_{ds}_1024{r}"
        for ds in ("colorferet", "kk")
        for r in ("", "_224")
    ]
)
#: Tables published without any shading in arXiv v1.
UNSHADED_TABLES: frozenset[str] = frozenset(["fairness_disparity_512"])
#: Table stem -> 0-based column indices that carry no shading (column 0 = row label).
UNSHADE_COLUMNS: dict[str, tuple[int, ...]] = {"jpegai_speed": (8,)}

CELLCOLOR = re.compile(r"\\cellcolor\{[a-z]+!\d+\}")
# The redundant marker wrapped around a whole shaded cell.
EMPHASIS = re.compile(r"^\\(?:textbf|underline)\{(.*)\}$")
# Leading number of a cell: "1.4 [0.8,6.0]", "0.57 @224", "\textbf{0.57 @224}", "12.54".
NUMBER = re.compile(r"-?\d+\.?\d*")


# ------------------------------------------------------------------- cell writers
def shade_best(body: str, emphasis: bool = True) -> str:
    r"""Return ``\cellcolor{green!25}\textbf{body}`` (no marker w/o ``emphasis``)."""
    return f"\\cellcolor{{{BEST_FILL}}}" + (f"\\textbf{{{body}}}" if emphasis else body)


def shade_worst(body: str, emphasis: bool = True) -> str:
    r"""Return ``\cellcolor{red!22}\underline{body}`` (no marker w/o ``emphasis``)."""
    return f"\\cellcolor{{{WORST_FILL}}}" + (
        f"\\underline{{{body}}}" if emphasis else body
    )


def emphasis_for(stem: str) -> bool:
    """Whether the published table ``stem`` carries the bold/underline marker."""
    return stem not in NO_EMPHASIS_TABLES


# ------------------------------------------------------------------- emphasis pass
def add_emphasis(text: str) -> tuple[str, int]:
    r"""Wrap every shaded cell that lacks it in the bold / underline marker.

    A shaded cell runs from its ``\cellcolor`` to the next ``&`` or row end; the
    content is taken lazily so that a trailing ``$^\dagger$`` or ``{\tiny[..]}``
    stays inside the marker. Cells already wrapped are left alone (idempotent).

    Returns
    -------
    tuple[str, int]
        The new text and the number of cells that were wrapped.
    """
    n = 0
    for fill, macro in EMPHASIS_RULES:
        pat = re.compile(
            r"(\\cellcolor\{" + re.escape(fill) + r"\})((?:(?!\\cellcolor)[^&\\])*"
            r"(?:\\(?!cellcolor)[a-zA-Z]+(?:\{[^{}]*\})*(?:(?!\\cellcolor)[^&\\])*)*)"
        )

        def sub(m: re.Match, macro: str = macro) -> str:
            nonlocal n
            body = m.group(2)
            stripped = body.strip()
            if not stripped or stripped.startswith(f"\\{macro}{{"):
                return m.group(0)
            lead = body[: len(body) - len(body.lstrip())]
            trail = body[len(body.rstrip()) :]
            n += 1
            return f"{m.group(1)}{lead}\\{macro}{{{stripped}}}{trail}"

        text = pat.sub(sub, text)
    return text, n


def strip_emphasis(text: str) -> str:
    r"""Remove the bold / underline marker from shaded cells (keep the colour).

    The inverse of :func:`add_emphasis` for cells whose content has balanced braces;
    useful to compare against the colour-only tables of :data:`NO_EMPHASIS_TABLES`.
    """
    for fill, macro in EMPHASIS_RULES:
        prefix = f"\\cellcolor{{{fill}}}"
        out, i = [], 0
        while True:
            j = text.find(prefix, i)
            if j < 0:
                out.append(text[i:])
                break
            k = j + len(prefix)
            out.append(text[i:k])
            m = re.match(r"(\s*)\\" + macro + r"\{", text[k:])
            if not m:
                i = k
                continue
            start = k + m.end()
            depth, p = 1, start
            while p < len(text) and depth:
                depth += {"{": 1, "}": -1}.get(text[p], 0)
                p += 1
            out.append(m.group(1) + text[start : p - 1])
            i = p
        text = "".join(out)
    return text


# ------------------------------------------------------------------- ranking
def _strip(cell: str) -> str:
    return CELLCOLOR.sub("", cell).strip()


def cell_value(cell: str) -> float | None:
    """Leading numeric value of a cell, or ``None`` if it carries no number."""
    txt = _strip(cell)
    if not txt or txt in {"--", "---", "n/a"}:
        return None
    m = NUMBER.search(txt.replace("\\textbf{", "").replace("}", ""))
    return float(m.group()) if m else None


def _is_row(line: str) -> bool:
    """Return True for a data row: has cells and is no rule, header or block title."""
    if "&" not in line or "\\\\" not in line:
        return False
    bare = line.strip()
    if bare.startswith(
        ("\\toprule", "\\midrule", "\\bottomrule", "\\cmidrule", "\\addlinespace", "%")
    ):
        return False
    # A \multicolumn line is a spanning header or an in-table block title, never data.
    return "\\multicolumn" not in bare


def _blocks(lines: list[str]) -> list[list[int]]:
    r"""Group row indices into independently ranked blocks.

    Blocks break at any structural line (a rule or a ``\multicolumn`` title); the
    region before the first ``\midrule`` (the header) is skipped.
    """
    try:
        start = (
            next(i for i, ln in enumerate(lines) if ln.strip().startswith("\\midrule"))
            + 1
        )
    except StopIteration:
        start = 0
    out, cur = [], []
    for i in range(start, len(lines)):
        if _is_row(lines[i]):
            cur.append(i)
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def shade_table(
    text: str,
    lower_better: bool,
    refs: tuple[str, ...] = (),
    emphasis: bool = True,
) -> str:
    r"""Shade best / worst per numeric column and block of a LaTeX tabular.

    Existing ``\cellcolor`` commands are removed first, so the function is
    idempotent.

    Parameters
    ----------
    text
        The tabular source.
    lower_better
        Whether lower values are better (EER, disparity) or higher (id-cos, PSNR).
    refs
        Row-label substrings of reference rows excluded from the ranking.
    emphasis
        Also add the bold / underline marker (:func:`add_emphasis`).
    """
    lines = text.split("\n")
    for block in _blocks(lines):
        split = {i: lines[i].rsplit("\\\\", 1)[0].split("&") for i in block}
        ncol = max(len(v) for v in split.values())
        ranked = [i for i in block if not any(r in split[i][0] for r in refs)]
        for j in range(1, ncol):
            vals = {
                i: cell_value(split[i][j])
                for i in ranked
                if j < len(split[i]) and cell_value(split[i][j]) is not None
            }
            if len(set(vals.values())) < 2:
                continue
            best = (min if lower_better else max)(vals.values())
            worst = (max if lower_better else min)(vals.values())
            for i, v in vals.items():
                body = _strip(split[i][j])
                if v == best:
                    split[i][j] = f" \\cellcolor{{{BEST_FILL}}}{body} "
                elif v == worst:
                    split[i][j] = f" \\cellcolor{{{WORST_FILL}}}{body} "
        for i in block:  # strip stale colours from cells that were not shaded
            for j in range(len(split[i])):
                if "\\cellcolor" not in split[i][j]:
                    split[i][j] = f" {_strip(split[i][j])} " if j else split[i][j]
            lines[i] = "&".join(split[i]).rstrip() + " \\\\"
    out = "\n".join(lines)
    return add_emphasis(out)[0] if emphasis else out


def _plain(cell: str) -> str:
    """Cell content with the colour and its redundant marker both removed."""
    txt = _strip(cell)
    m = EMPHASIS.match(txt)
    return m.group(1) if m else txt


def unshade_columns(text: str, cols: tuple[int, ...]) -> str:
    """Return ``text`` with colour and marker dropped from the given columns."""
    lines = text.split("\n")
    for block in _blocks(lines):
        for i in block:
            cells = lines[i].rsplit("\\\\", 1)[0].split("&")
            if not any(j < len(cells) for j in cols):
                continue
            for j in cols:
                if j < len(cells):
                    cells[j] = f" {_plain(cells[j])} "
            lines[i] = "&".join(cells).rstrip() + " \\\\"
    return "\n".join(lines)


def finalize_table(stem: str, text: str) -> str:
    """Apply the published post-processing of table ``stem`` to generator output.

    Adds the bold / underline marker unless ``stem`` is in
    :data:`NO_EMPHASIS_TABLES`, and removes shading from the columns listed in
    :data:`UNSHADE_COLUMNS`. Tables in :data:`UNSHADED_TABLES` are returned as is.
    """
    if stem in UNSHADED_TABLES:
        return text
    if emphasis_for(stem):
        text = add_emphasis(text)[0]
    if stem in UNSHADE_COLUMNS:
        text = unshade_columns(text, UNSHADE_COLUMNS[stem])
    return text
