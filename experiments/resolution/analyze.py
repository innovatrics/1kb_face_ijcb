# SPDX-License-Identifier: MIT
r"""Resolution-information analysis: what a crop resolution removes, and its EER.

Three measurements per dataset, written to ``OUTPUT_ROOT/resolution_information/``:

``embedding_decomposition.csv`` (dataset, model, tag, kind, res, codec, budget, ...)
    Per matcher and source, the median over images of the embedding cosine
    ``resize_cos = cos(clean_112, clean_R)`` for the clean crops, and, for compressed
    sources included with ``--codecs``, ``compress_cos = cos(clean_R, comp_R)`` and
    ``net_cos = cos(clean_112, comp_R)``. Rows with a missing embedding drop out.
``spectral_retention.csv`` (dataset, res, nyquist_cyc, frac_energy_retained, ...)
    The radial power spectrum of ``--spectral-sample`` seeded (seed 0) clean 224 px
    crops (grey, mean removed), averaged; the fraction of AC energy up to each
    resolution's Nyquist frequency (R/2 cycles per width).
``eer_by_resolution.csv`` (dataset, model, tag, kind, res, codec, budget, eer)
    EER of each source over all mated pairs plus ``--sample-nonmated`` seeded
    non-mated pairs (seed 0).

The sources are the clean tags ``aligned_<R>`` for R in 64, 96, 112, 168, 224, plus,
with ``--codecs``, the compressed tags ``<codec>_<R>_<budget>`` of the benchmark grid
(no crop-variant suffixes). Rows are in tag-name order per model. ``render.py`` draws
the paper table and figure from these CSVs.

Examples
--------
    python experiments/resolution/analyze.py
    python experiments/resolution/analyze.py --datasets kk \
        --models arcface_antelopev2,topofr_r100,edgeface_xs,lvface_l
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from face1kb import config, fr
from face1kb.data.cli_utils import parse_list, setup_logging
from face1kb.eval import verification
from face1kb.eval.embeddings import CODECS, load_embeddings, parse_tag

log = logging.getLogger("resolution_information")

RESOLUTIONS = config.RESOLUTIONS
#: Fixed reference: the clean crop at the matchers' native input size.
REF_RES = 112
#: Reference resolution of the spectral measurement.
SPECTRAL_REF = 224


def source_tags(codecs: list[str], budgets=config.BUDGETS) -> list[str]:
    """Tags analysed: clean ``aligned_<R>`` and optional ``<codec>_<R>_<B>``."""
    tags = [config.embedding_tag(r) for r in RESOLUTIONS]
    tags += [
        config.embedding_tag(r, "", c, b)
        for c in codecs
        for r in RESOLUTIONS
        for b in budgets
    ]
    return sorted(tags)


def _unit(e: np.ndarray) -> np.ndarray:
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-12)


def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-image cosine of two id-aligned arrays (NaN where a row is missing)."""
    return (_unit(a) * _unit(b)).sum(axis=1)


def median_finite(x: np.ndarray) -> float:
    """Median of the non-NaN values (NaN when there are none)."""
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else float("nan")


def _meta(tag: str) -> dict:
    m = parse_tag(tag, aligned_codec="", codecs=CODECS)
    return {k: m[k] for k in ("kind", "res", "codec", "budget")}


def embedding_decomposition(dataset: str, models: list[str], tags: list[str]):
    """Median resize / compress / net cosines per (model, tag)."""
    rows = []
    for model in models:
        ref = load_embeddings(dataset, model, config.embedding_tag(REF_RES))
        clean = {
            r: load_embeddings(dataset, model, config.embedding_tag(r))
            for r in RESOLUTIONS
        }
        for tag in tags:
            meta = _meta(tag)
            r = meta["res"]
            e = (
                clean[r]
                if meta["kind"] == "aligned"
                else load_embeddings(dataset, model, tag)
            )
            if e is None:
                continue
            rec = {"dataset": dataset, "model": model, "tag": tag, **meta}
            if meta["kind"] == "aligned" and ref is not None:
                rec["resize_cos"] = median_finite(rowwise_cos(ref, e))
            if meta["kind"] == "compressed":
                if clean[r] is not None:
                    rec["compress_cos"] = median_finite(rowwise_cos(clean[r], e))
                if ref is not None:
                    rec["net_cos"] = median_finite(rowwise_cos(ref, e))
            rows.append(rec)
    return rows


def radial_spectrum(gray: np.ndarray) -> np.ndarray:
    """Radially averaged power spectrum of a mean-removed grey image."""
    g = gray.astype(np.float64)
    g -= g.mean()
    psd = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = g.shape
    yy, xx = np.indices((h, w))
    rad = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    return np.bincount(rad.ravel(), psd.ravel()) / np.maximum(
        np.bincount(rad.ravel()), 1
    )


def retention_rows(dataset: str, acc: np.ndarray, used: int, ref_res: int):
    """AC-energy fraction retained at each resolution's Nyquist frequency."""
    nyq = ref_res // 2
    csum = np.cumsum(acc[1 : nyq + 1])
    total = csum[-1]
    rows = []
    for r in RESOLUTIONS:
        fr_ = min(r // 2, nyq)
        retained = float(csum[fr_ - 1] / total)
        rows.append(
            {
                "dataset": dataset,
                "res": r,
                "nyquist_cyc": fr_,
                "frac_energy_retained": retained,
                "frac_energy_removed_vs_ref": 1.0 - retained,
                "ref_res": ref_res,
                "n_images": used,
            }
        )
    return rows


def spectral_retention(dataset: str, n_sample: int, ref_res: int = SPECTRAL_REF):
    """Spectral retention of ``n_sample`` seeded clean crops of ``dataset``."""
    import cv2  # noqa: PLC0415

    rels = config.read_index(dataset)["rel_path"].to_numpy()
    rng = np.random.default_rng(0)
    take = rng.choice(len(rels), size=min(n_sample, len(rels)), replace=False)
    crop_dir = config.aligned_dir(dataset, ref_res)
    acc, used = None, 0
    for i in take:
        g = cv2.imread(str(crop_dir / rels[i]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        radial = radial_spectrum(g)
        acc = radial if acc is None else acc + radial[: len(acc)]
        used += 1
    if used == 0:
        return []
    return retention_rows(dataset, acc / used, used, ref_res)


def eer_by_resolution(dataset: str, models: list[str], tags: list[str], nonmated: int):
    """EER per (model, source) over the shared trial set."""
    (p1, p2), (n1, n2) = verification.pair_indices(dataset, nonmated, 0)
    rows = []
    for model in models:
        for tag in tags:
            emb = load_embeddings(dataset, model, tag)
            if emb is None:
                continue
            eer = verification.eer(
                verification.cosine_scores(emb, p1, p2),
                verification.cosine_scores(emb, n1, n2),
            )
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "tag": tag,
                    **_meta(tag),
                    "eer": eer,
                }
            )
            log.info("  %s %-20s %-24s eer=%.4f%%", dataset, model, tag, eer * 100)
    return rows


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument(
        "--models",
        default="roster",
        help="comma list, or 'roster' (the public matchers with embeddings on disk, "
        "in name order)",
    )
    ap.add_argument(
        "--codecs",
        default="",
        help="compressed sources to add ('all' = every benchmark codec; default none)",
    )
    ap.add_argument("--sample-nonmated", type=int, default=3_000_000)
    ap.add_argument("--spectral-sample", type=int, default=400)
    ap.add_argument("--no-eer", action="store_true", help="skip the EER pass")
    ap.add_argument(
        "--out-dir", default=None, help="default: OUTPUT_ROOT/resolution_information"
    )
    args = ap.parse_args(argv)
    setup_logging()
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else config.output_dir("resolution_information")
    )
    codecs = list(CODECS) if args.codecs == "all" else parse_list(args.codecs)
    tags = source_tags(codecs)

    dec_rows, eer_rows, spec_rows = [], [], []
    for ds in parse_list(args.datasets):
        if args.models == "roster":
            models = sorted(
                m for m in fr.ROSTER if config.embeddings_dir(ds, m).is_dir()
            )
        else:
            models = parse_list(args.models)
        log.info("[%s] models: %s", ds, models)
        dec_rows += embedding_decomposition(ds, models, tags)
        spec_rows += spectral_retention(ds, args.spectral_sample)
        if not args.no_eer:
            eer_rows += eer_by_resolution(ds, models, tags, args.sample_nonmated)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dec_rows).to_csv(out_dir / "embedding_decomposition.csv", index=False)
    pd.DataFrame(spec_rows).to_csv(out_dir / "spectral_retention.csv", index=False)
    if not args.no_eer:
        pd.DataFrame(eer_rows).to_csv(out_dir / "eer_by_resolution.csv", index=False)
    log.info("wrote the CSVs to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
