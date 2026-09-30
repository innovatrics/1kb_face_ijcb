# SPDX-License-Identifier: MIT
r"""ISO/IEC 29794-5 annex study: Stage-B flag sweep and confirmation cell lists.

Both lists are chosen on the Color FERET 2000-crop scores (``scores_pop2000.csv`` of
``score.py``), with the EER of the ``all`` population under the symmetric protocol
averaged over ``arcface_antelopev2`` and ``cvlface_vit_b``:

* default: for each classical codec, the two best Stage-A grid cells (resolution,
  colour, manipulation at default flags), each combined with every flag set of that
  codec (:data:`FLAG_SETS`), 62 cells -> ``stage_b_cells.json``. Only grid cells of
  the six classical codecs are considered; the JPEG-AI arm, which is recorded as grid
  cells too, has no flag sets;
* ``--confirm``: the best cell per classical codec over the grid and Stage B, 6
  cells -> ``confirm_cells.json``, re-run by ``sweep.py --stage confirm`` at the full
  population of both datasets with all five matchers.

Input: ``OUTPUT_ROOT/annex/scores_pop2000.csv`` (``--scores``). Output: the JSON cell
list in ``OUTPUT_ROOT/annex/`` (``--out``). CPU, seconds.

Examples
--------
    python experiments/annex/stageb.py
    python experiments/annex/stageb.py --confirm
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from face1kb import config
from face1kb.data.cli_utils import setup_logging

log = logging.getLogger("annex.stageb")

TOP_K = 2
MODELS = ("arcface_antelopev2", "cvlface_vit_b")
BUDGET = 1024

#: Per-codec flag sets; each becomes one cell on top of a Stage-A winner.
FLAG_SETS: dict[str, list[dict]] = {
    "jpeg": [{"smooth": s} for s in (10, 30, 50)]
    + [{"progressive": True}, {"arithmetic": True}, {"smooth": 30, "arithmetic": True}],
    "jpeg2000": [{"num_resolutions": n} for n in (1, 3, 5, 7)],
    "jpeg_xl": [{"effort": e} for e in (1, 5, 10)],
    "avif": [
        {"subsampling": s, "speed": sp}
        for s in ("4:4:4", "4:2:0", "4:0:0")
        for sp in (1, 6)
    ],
    "heif": [
        {"chroma_downsampling": c} for c in ("nearest-neighbor", "average", "sharp-yuv")
    ],
    "webp": [{"method": m, "sns": s} for m in (0, 3, 6) for s in (0, 40, 80)],
}


def winners(df: pd.DataFrame) -> list[dict]:
    """Best cell per codec across the grid and Stage B (the confirmation run)."""
    x = df[df.arm.isin(["grid", "stageB"]) & df.codec.isin(FLAG_SETS)]
    k = (
        x.groupby(["codec", "res", "color", "manip", "flags"])
        .all_sym_eer.mean()
        .reset_index()
    )
    out = []
    for c, g in k.groupby("codec"):
        w = g.nsmallest(1, "all_sym_eer").iloc[0]
        out.append(
            {
                "codec": c,
                "res": int(w.res),
                "color": w.color,
                "manip": w.manip,
                "flags": json.loads(w["flags"]),
                "budget": BUDGET,
                "arm": "winner",
            }
        )
    return out


def stage_b(df: pd.DataFrame) -> list[dict]:
    """Flag-sweep cells around the two best grid cells of each classical codec."""
    x = df[(df.arm == "grid") & df.codec.isin(FLAG_SETS)]
    key = x.groupby(["codec", "res", "color", "manip"]).all_sym_eer.mean().reset_index()
    cells = []
    for codec, g in key.groupby("codec"):
        for _, w in g.nsmallest(TOP_K, "all_sym_eer").iterrows():
            for flags in FLAG_SETS[codec]:
                cells.append(
                    {
                        "codec": codec,
                        "res": int(w.res),
                        "color": w.color,
                        "manip": w.manip,
                        "flags": flags,
                        "budget": BUDGET,
                        "arm": "stageB",
                    }
                )
    return cells


def main(argv: list[str] | None = None) -> int:
    """Write the Stage-B or the confirmation cell list."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--scores", default=None, help="default OUTPUT_ROOT/annex/...")
    ap.add_argument("--out", default=None, help="output JSON")
    args = ap.parse_args(argv)
    setup_logging()
    scores = (
        Path(args.scores)
        if args.scores
        else config.output_dir("annex", "scores_pop2000.csv")
    )
    df = pd.read_csv(scores)
    df = df[(df.dataset == "colorferet") & df.model.isin(MODELS)]
    cells = winners(df) if args.confirm else stage_b(df)
    name = "confirm_cells.json" if args.confirm else "stage_b_cells.json"
    out = Path(args.out) if args.out else config.output_dir("annex", name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cells))
    log.info("%d cells -> %s", len(cells), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
