# SPDX-License-Identifier: MIT
"""Clean-crop ablation: crop tightness and preprocessing operators (Color FERET).

Scores the uncompressed 112 px crop variants from their embeddings
(``embeddings/<model>/aligned_112<suffix>.npy``) on one fixed verification trial
set -- all mated pairs plus 2,000,000 seeded (seed 0) non-mated pairs of
``pairs.parquet`` -- and writes one aggregate CSV::

    OUTPUT_ROOT/ablation/preprocessing_ablation.csv
    columns: study, variant, suffix, model, eer, id_cos_vs_std, n_pos, n_neg

* ``study = crop``: the crop-tightness presets ``standard`` (the benchmark crop),
  ``tight``, ``mid`` and ``fill`` (suffixes ``""``, ``_tight``, ``_mid``, ``_fill``;
  made by ``experiments/prepare/make_crop_variants.py``).
* ``study = preproc``: ``std`` (no preprocessing) and the operators ``A1``-``A4``,
  ``B1``, ``B2``, ``C1``, ``C2`` (suffix ``_<op>``; made by
  ``experiments/prepare/preprocess.py``).

``id_cos_vs_std`` is the mean cosine between the variant's and the standard crop's
embedding of the same image (float64, rows with a non-finite value in either array
dropped; 1.0 for the standard crop). Missing embedding arrays give NaN rows.

The embeddings are computed with the benchmark embedding stage for the suffixed
crop sets (see docs/reproduce_studies.md). ``make_tables.py`` renders the paper
tables from this CSV.

Examples
--------
    python experiments/preprocessing/ablation.py
    python experiments/preprocessing/ablation.py --models arcface_antelopev2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from face1kb import config
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval import verification
from face1kb.eval.embeddings import ANCHOR_MODELS, EMB_DIM, load_embeddings

log = logging.getLogger("preprocessing_ablation")

DATASET = "colorferet"
RES = 112
#: Non-mated trials of the ablation (seed 0).
SAMPLE_NONMATED = 2_000_000
#: (study, variant, suffix) in table order.
VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("crop", "standard", ""),
    ("crop", "tight", "_tight"),
    ("crop", "mid", "_mid"),
    ("crop", "fill", "_fill"),
    ("preproc", "std", ""),
    ("preproc", "A1", "_A1"),
    ("preproc", "A2", "_A2"),
    ("preproc", "A3", "_A3"),
    ("preproc", "A4", "_A4"),
    ("preproc", "B1", "_B1"),
    ("preproc", "B2", "_B2"),
    ("preproc", "C1", "_C1"),
    ("preproc", "C2", "_C2"),
)
OUT_NAME = "preprocessing_ablation.csv"


def id_cos(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Mean cosine between the rows of two embedding arrays of the same images."""
    if a is None or b is None:
        return float("nan")
    n = min(len(a), len(b))
    a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
    ok = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    if not ok.any():
        return float("nan")
    a, b = a[ok], b[ok]
    a /= np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
    b /= np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
    return float((a * b).sum(1).mean())


def _load(model: str, suffix: str) -> np.ndarray | None:
    emb = load_embeddings(DATASET, model, config.embedding_tag(RES, suffix))
    if emb is None or emb.ndim != 2 or emb.shape[1] != EMB_DIM:
        return None
    return np.asarray(emb)


def ablation_rows(models, pairs) -> list[dict]:
    """One row per (variant, model); every embedding array is loaded once."""
    (p1, p2), (n1, n2) = pairs
    rows, cache, metrics = [], {}, {}
    for model in models:
        std = _load(model, "")
        for study, variant, suffix in VARIANTS:
            if suffix not in cache:
                cache[suffix] = std if suffix == "" else _load(model, suffix)
            emb = cache[suffix]
            rec = {
                "study": study,
                "variant": variant,
                "suffix": suffix,
                "model": model,
                "eer": float("nan"),
                "id_cos_vs_std": 1.0 if suffix == "" else id_cos(std, emb),
                "n_pos": 0,
                "n_neg": 0,
            }
            if emb is not None:
                if suffix not in metrics:
                    metrics[suffix] = verification.verification_metrics(
                        verification.cosine_scores(emb, p1, p2),
                        verification.cosine_scores(emb, n1, n2),
                    )
                met = metrics[suffix]
                rec.update(
                    eer=float(met["eer"]), n_pos=met["n_pos"], n_neg=met["n_neg"]
                )
            else:
                log.warning("missing %s %s", model, config.embedding_tag(RES, suffix))
            rows.append(rec)
            log.info(
                "  %-18s %-8s %-8s eer=%.4f%% id_cos=%.4f",
                model,
                study,
                variant,
                rec["eer"] * 100,
                rec["id_cos_vs_std"],
            )
        cache.clear()
        metrics.clear()
    return rows


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default=",".join(ANCHOR_MODELS))
    ap.add_argument("--sample-nonmated", type=int, default=SAMPLE_NONMATED)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="default: OUTPUT_ROOT/ablation/...")
    args = ap.parse_args(argv)
    setup_logging()
    out = Path(args.out) if args.out else config.output_dir("ablation") / OUT_NAME
    pairs = verification.pair_indices(DATASET, args.sample_nonmated, args.seed)
    rows = ablation_rows(parse_list(args.models), pairs)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
