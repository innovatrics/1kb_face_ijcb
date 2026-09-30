# SPDX-License-Identifier: MIT
r"""Codec-properties table of the related-work section (``tab:codec-properties``).

One row per benchmarked codec: owner / reference implementation, year of the
standard, licensing, hardware-decode support and the share of Color FERET crops whose
stored stream fits 1024 B at 112 / 224 px.

Sources:

* ``--properties`` (default: the shipped ``results/codec_properties.csv``): the
  codec provenance record (owner, year, standard, licence, implementation, hardware
  support, ...). The table prints the paper's condensed wording of those fields
  (:data:`ROWS`); the script checks that every baseline of the table has a record
  there. Rows of codecs outside the benchmark are ignored.
* ``<input-root>/codec_comparison/file_size_summary.csv`` (written by
  ``generate_file_size_boxplots.py``; default input root ``OUTPUT_ROOT``,
  ``--from-results`` for the shipped ``results/``): the fit column,
  ``round(pct_holding)`` of the Color FERET 1024 B cells at 112 and 224 px. A codec
  run only on a subset of the crops (fewer than half of the largest cell count; the
  CompressAI baselines ran on 300 crops) reads ``n/a``.

Output: ``OUTPUT_ROOT/tables/codec_properties.tex``. With ``--from-results`` the
published table is reproduced.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

from face1kb import config

log = logging.getLogger("codec_properties_table")

#: (codec, label, owner / reference implementation, standard year, licensing,
#: hardware decode) in table order: the paper's condensed wording.
ROWS = (
    (
        "jpeg",
        "JPEG",
        "JPEG cmte.; libjpeg-turbo",
        "1992",
        "Royalty-free (expired)",
        "Universal (ISP/GPU)",
    ),
    (
        "jpeg2000",
        "JPEG\\,2000",
        "JPEG cmte.; OpenJPEG, Kakadu",
        "2000",
        "Part\\,1 royalty-free",
        "Rare (specialized)",
    ),
    (
        "webp",
        "WebP",
        "Google; libwebp",
        "2010",
        "Royalty-free (BSD-3)",
        "None (SW; fast)",
    ),
    (
        "jpeg_xl",
        "JPEG\\,XL",
        "JPEG cmte.; libjxl",
        "2022",
        "Royalty-free (BSD-3)",
        "None (SW; 2026)",
    ),
    (
        "avif",
        "AVIF",
        "AOMedia; libavif/dav1d",
        "2019",
        "Royalty-free (AOM)",
        "Growing (recent SoCs)",
    ),
    (
        "heif",
        "HEIF",
        "MPEG+Nokia; libheif/x265",
        "2017",
        "HEVC patent-encumbered",
        "Ubiquitous (HEVC)",
    ),
    (
        "jpeg_fzt",
        "JPEG-FzT",
        "Innovatrics / U.\\ Ostrava",
        "2021",
        "Research; JPEG base RF",
        "JPEG HW + SW pass",
    ),
    (
        "jpeg_ai",
        "JPEG-AI",
        "ITU-T/ISO WG1; ref.\\ SW",
        "2025",
        "Royalty-free goal (BSD)",
        "None yet (NPU/GPU)",
    ),
    (
        "neural_bmshj2018",
        "bmshj2018",
        "Ball\\'e et al.; CompressAI",
        "2018",
        "BSD-3, patent non-grant",
        "None (NN)",
    ),
    (
        "neural_mbt2018_mean",
        "mbt2018",
        "Minnen et al.; CompressAI",
        "2018",
        "BSD-3, patent non-grant",
        "None (NN)",
    ),
)
#: The face1kb codecs (second block of the table).
OURS_ROWS = (
    (
        "ours_fast",
        "Ours-FAST",
        "This work (learned)",
        "2026",
        "MIT (source to be released)",
        "None (GPU/NPU)",
    ),
    (
        "ours_accurate",
        "Ours-ACCURATE",
        "This work (learned, side-stream)",
        "2026",
        "MIT (source to be released)",
        "None (GPU/NPU)",
    ),
)
#: Padded widths of the first five columns in the two blocks of the published table.
WIDTHS = ((12, 37, 5, 30, 25), (15, 34, 5, 30, 25))
#: Width of the fit column.
FIT_WIDTH = 11
FIT_DATASET, FIT_BUDGET, FIT_RES = "colorferet", 1024, (112, 224)


def read_properties(path: Path) -> dict[str, dict]:
    """Codec -> record of ``codec_properties.csv``."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r["codec"]: r for r in csv.DictReader(f)}


def fit_values(summary_csv: Path) -> dict[tuple[str, int], float | None]:
    """``(codec, res) -> pct_holding`` of the fit cells (None: subset-only codec)."""
    import pandas as pd  # noqa: PLC0415

    df = pd.read_csv(summary_csv)
    out: dict[tuple[str, int], float | None] = {}
    for res in FIT_RES:
        sub = df[
            (df.dataset == FIT_DATASET) & (df.budget == FIT_BUDGET) & (df.res == res)
        ]
        if sub.empty:
            continue
        full = int(sub.n.max())
        for r in sub.itertuples(index=False):
            out[(r.codec, res)] = float(r.pct_holding) if r.n >= 0.5 * full else None
    return out


def fit_cell(codec: str, fits: dict) -> str:
    r"""Return ``"<112>\,/\,<224>"`` in whole percent, or ``n/a``."""
    vals = [fits.get((codec, r)) for r in FIT_RES]
    if any(v is None for v in vals):
        return "n/a"
    return "\\,/\\,".join(f"{round(v):.0f}" for v in vals)


def render(fits: dict) -> str:
    """Return the LaTeX tabular."""
    lines = [
        "\\begin{tabular}{@{}llllll@{}}",
        "\\toprule",
        "Codec & Owner / reference impl. & Std.\\ year & Licensing & HW decode & "
        "Fit $\\le$1\\,kB (112/224) \\\\",
        "\\midrule",
    ]
    for block, (rows, widths) in enumerate(zip((ROWS, OURS_ROWS), WIDTHS)):
        if block:
            lines.append("\\midrule")
        for codec, *text in rows:
            cells = [t.ljust(w) for t, w in zip(text, widths)]
            fit = fit_cell(codec, fits).ljust(FIT_WIDTH)
            lines.append("& ".join(cells) + "& " + fit + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--from-results", action="store_true", help="read results/")
    ap.add_argument("--input-root", default=None, help="root of the size summary")
    ap.add_argument("--properties", default=None, help="codec_properties.csv")
    ap.add_argument("--output-root", default=None, help="override FACE1KB_OUTPUT_ROOT")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    in_root = Path(
        args.input_root
        or (config.RESULTS_ROOT if args.from_results else config.OUTPUT_ROOT)
    )
    props = read_properties(
        Path(args.properties or config.results_dir("codec_properties.csv"))
    )
    missing = [c for c, *_ in ROWS if c not in props]
    if missing:
        raise SystemExit(f"codec_properties.csv has no record for {missing}")
    fits = fit_values(in_root / "codec_comparison" / "file_size_summary.csv")
    out_root = Path(args.output_root) if args.output_root else config.OUTPUT_ROOT
    out = out_root / "tables" / "codec_properties.tex"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(fits))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
