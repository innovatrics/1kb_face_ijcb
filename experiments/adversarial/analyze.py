# SPDX-License-Identifier: MIT
r"""Sanitization metric of every (dataset, matcher, attack, eps, codec, budget) cell.

Reads the 112 px embedding arrays of the study (``WORK_ROOT/<dataset>/embeddings``)
and writes one row per cell with the columns of
:func:`face1kb.adversarial.sanitization_record`::

    dataset, model, attack, eps, codec, budget,
    n, cos_adv, cos_clean_comp, cos_adv_comp, residual, sanitization

The four arrays of a cell are the embeddings tagged

* ``aligned_112`` (clean, uncompressed: the reference),
* ``aligned_112_adv_<attack>_<tag>`` (adversarial, uncompressed),
* ``<codec>_112_<budget>`` (clean after the codec),
* ``<codec>_112_adv_<attack>_<tag>_<budget>`` (adversarial after the codec),

with ``<tag> = eps_tag(eps)`` (``006`` for 0.06). A cell whose arrays do not exist is
written with ``n = 0`` and NaN metrics, so the output always holds the full grid of
the requested datasets, matchers, attacks, ``eps``, codecs and budgets. A
(dataset, matcher) pair without clean embeddings is skipped.

The defaults are the paper's scope: both datasets, the four anchor matchers, the
three attacks at ``eps`` 0.03/0.06/0.10, all twelve codecs and both budgets. Output:
``OUTPUT_ROOT/adversarial/sanitization.csv`` (``--out`` to change). CPU only; the
cost is reading the embedding arrays (a few minutes on a local disk).

Examples
--------
    python experiments/adversarial/analyze.py
    python experiments/adversarial/analyze.py --datasets colorferet --attacks hfc \
        --models arcface_antelopev2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from face1kb import adversarial as adv
from face1kb import config
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval.embeddings import ANCHOR_MODELS, load_embeddings

log = logging.getLogger("adversarial.analyze")

#: Codecs of the study, in the row order of the result file.
CODECS: tuple[str, ...] = (
    "jpeg",
    "jpeg2000",
    "webp",
    "jpeg_xl",
    "avif",
    "heif",
    "jpeg_fzt",
    "ours_accurate",
    "ours_fast",
    "jpeg_ai",
    "neural_bmshj2018",
    "neural_mbt2018_mean",
)
#: Identifying columns of a result row (followed by the metric columns).
KEY_COLUMNS: tuple[str, ...] = ("dataset", "model", "attack", "eps", "codec", "budget")
RES = adv.ATTACK_RES


class _Arrays:
    """Embedding arrays of one (dataset, model), loaded once and kept."""

    def __init__(self, dataset: str, model: str):
        self.dataset, self.model = dataset, model
        self._cache: dict[str, np.ndarray | None] = {}

    def __call__(self, tag: str, keep: bool = True) -> np.ndarray | None:
        if tag in self._cache:
            return self._cache[tag]
        arr = load_embeddings(self.dataset, self.model, tag)
        if keep:
            self._cache[tag] = arr
        return arr


def metric_rows(
    dataset: str,
    model: str,
    attacks: list[str],
    eps_tags: list[str],
    codecs: list[str],
    budgets: list[int],
) -> list[dict]:
    """Rows of one (dataset, model); empty if its clean embeddings are missing."""
    arrays = _Arrays(dataset, model)
    clean = arrays(config.embedding_tag(RES))
    if clean is None:
        log.info("%s/%s: no clean %d px embeddings, skipped", dataset, model, RES)
        return []
    rows = []
    for attack in attacks:
        for tag in eps_tags:
            sfx = f"_adv_{attack}_{tag}"
            adv_emb = arrays(config.embedding_tag(RES, sfx), keep=False)
            for codec in codecs:
                for budget in budgets:
                    cc = arrays(config.embedding_tag(RES, "", codec, budget))
                    ca = arrays(config.embedding_tag(RES, sfx, codec, budget), False)
                    rec = adv.sanitization_record(clean, adv_emb, cc, ca)
                    rows.append(
                        {
                            "dataset": dataset,
                            "model": model,
                            "attack": attack,
                            "eps": int(tag) / 100,
                            "codec": codec,
                            "budget": budget,
                            **rec,
                        }
                    )
    return rows


def main(argv: list[str] | None = None) -> int:
    """Compute the sanitization metric from the stored embeddings."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--models", default=",".join(ANCHOR_MODELS))
    ap.add_argument("--attacks", default="hfc,clip,liae")
    ap.add_argument("--eps", default=",".join(str(e) for e in adv.DEFAULT_EPS))
    ap.add_argument("--codecs", default=",".join(CODECS))
    ap.add_argument("--budgets", default=",".join(str(b) for b in config.BUDGETS))
    ap.add_argument("--out", default=None, help="output CSV")
    args = ap.parse_args(argv)
    setup_logging()

    attacks = parse_list(args.attacks)
    unknown = sorted(set(attacks) - set(adv.ATTACKS))
    if unknown:
        raise SystemExit(
            f"unknown attack(s) {unknown}; choose from {sorted(adv.ATTACKS)}"
        )
    eps_tags = [adv.eps_tag(e) for e in parse_list(args.eps, float)]
    codecs = parse_list(args.codecs)
    budgets = parse_list(args.budgets, int)
    rows = []
    for ds in parse_list(args.datasets):
        for model in parse_list(args.models):
            rows += metric_rows(ds, model, attacks, eps_tags, codecs, budgets)
    cols = [*KEY_COLUMNS, *adv.METRIC_COLUMNS]
    df = pd.DataFrame(rows, columns=cols)
    out = (
        Path(args.out)
        if args.out
        else config.output_dir("adversarial") / "sanitization.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    have = df.dropna(subset=["residual"])
    log.info("wrote %s: %d cells, %d with data", out, len(df), len(have))
    if not have.empty:
        summary = have.pivot_table(
            index="codec", columns=["attack", "budget"], values="residual"
        )
        log.info("mean residual per codec:\n%s", summary.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
