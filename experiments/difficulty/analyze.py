# SPDX-License-Identifier: MIT
"""Reduce the per-image difficulty CSV to the aggregates of the paper section.

Reads ``difficulty/sample_difficulty.csv`` (from ``compute.py``; per-image, local) and
writes to ``OUTPUT_ROOT/difficulty/`` the aggregates that ``make_tables.py`` renders:

``difficulty_correlations.csv`` (predictor, key, <ds>, <ds>_p, <ds>_n per dataset)
    Image-level Spearman correlation between difficulty and each codec-independent
    descriptor (both datasets) and each dataset attribute (Color FERET |yaw|,
    glasses, beard, mustache; AI-Solutions-KK Monk skin-tone index, age).
    Difficulty is ``1 - id_cos`` averaged over the codecs of an image at the primary
    budget (512 B), so each image is one point. Predictors with fewer than 10 images
    or a single distinct value give NaN.
``cross_codec_sharing.txt``
    Per dataset, the mean / min / max over codec pairs of the Spearman correlation
    of per-image difficulty (pairs with at least 20 common images).
``difficulty_contrast.csv`` (key, easiest, hardest, k, n, count_easiest, count_hardest)
    Color FERET: mean descriptor / attribute value of the k = n // 10 easiest and
    hardest images (image-level deciles of mean difficulty), and the glasses counts.

Examples
--------
    python experiments/difficulty/analyze.py
    python experiments/difficulty/analyze.py --csv my_sample_difficulty.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from face1kb import config
from face1kb.data.cli_utils import setup_logging

log = logging.getLogger("difficulty_analyze")

#: The discriminating budget of the analysis.
PRIMARY_BUDGET = 512
DATASETS = ("colorferet", "kk")
#: (key, table label) of the codec-independent descriptors.
DESCRIPTORS: tuple[tuple[str, str], ...] = (
    ("hf_share", "High-freq.\\ DCT share"),
    ("lap_var", "Laplacian var.\\ (detail)"),
    ("grad_mag", "Gradient magnitude"),
    ("edge_density", "Edge density"),
    ("entropy", "Grey-level entropy"),
    ("contrast", "Contrast (luma std)"),
    ("colorfulness", "Colorfulness"),
    ("brightness", "Brightness (luma mean)"),
)
#: Dataset attributes tested per dataset.
CF_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("abs_yaw", "$|$yaw$|$ (non-frontal)"),
    ("glasses", "Glasses"),
    ("beard", "Beard"),
    ("mustache", "Mustache"),
)
KK_ATTRIBUTES: tuple[tuple[str, str], ...] = (
    ("mst_index", "Monk skin-tone index"),
    ("age", "Age"),
)
#: Traits of the easiest-vs-hardest contrast (Color FERET).
CONTRAST_KEYS: tuple[str, ...] = (
    "hf_share",
    "lap_var",
    "abs_yaw",
    "brightness",
    "contrast",
)


def image_spearman(df: pd.DataFrame, predictor: str) -> tuple[float, float, int]:
    """Spearman ``(rho, p, n)`` of per-image mean difficulty vs. ``predictor``."""
    from scipy.stats import spearmanr  # noqa: PLC0415

    d = df.copy()
    d["difficulty"] = 1.0 - pd.to_numeric(d["id_cos"], errors="coerce")
    d[predictor] = pd.to_numeric(d[predictor], errors="coerce")
    g = d.groupby(["subject", "image"])
    di = g["difficulty"].mean()
    xv = g[predictor].first()
    m = di.notna() & xv.notna()
    if int(m.sum()) < 10 or xv[m].nunique() < 2:
        return float("nan"), float("nan"), int(m.sum())
    r, p = spearmanr(di[m], xv[m])
    return float(r), float(p), int(m.sum())


def correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the predictor table (``difficulty_correlations.csv``)."""
    d = df[df.budget == PRIMARY_BUDGET]
    rows = []
    for key, lab in DESCRIPTORS:
        rec = {"predictor": lab, "key": key}
        for ds in DATASETS:
            r, p, n = image_spearman(d[d.dataset == ds], key)
            rec[ds], rec[f"{ds}_p"], rec[f"{ds}_n"] = r, p, n
        rows.append(rec)
    for ds, attrs in (("colorferet", CF_ATTRIBUTES), ("kk", KK_ATTRIBUTES)):
        for key, lab in attrs:
            r, p, n = image_spearman(d[d.dataset == ds], key)
            rows.append(
                {"predictor": lab, "key": key, ds: r, f"{ds}_p": p, f"{ds}_n": n}
            )
    return pd.DataFrame(rows)


def cross_codec_sharing(df: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    """Mean / min / max pairwise Spearman of per-image difficulty across codecs."""
    from scipy.stats import spearmanr  # noqa: PLC0415

    out = {}
    d = df[df.budget == PRIMARY_BUDGET]
    for ds in DATASETS:
        sub = d[d.dataset == ds].copy()
        sub["difficulty"] = 1.0 - pd.to_numeric(sub["id_cos"], errors="coerce")
        piv = sub.pivot_table(
            index=["subject", "image"],
            columns="codec",
            values="difficulty",
            aggfunc="first",
        )
        codecs = list(piv.columns)
        rhos = []
        for i in range(len(codecs)):
            for j in range(i + 1, len(codecs)):
                a, b = piv[codecs[i]], piv[codecs[j]]
                m = a.notna() & b.notna()
                if m.sum() >= 20:
                    r, _ = spearmanr(a[m], b[m])
                    if np.isfinite(r):
                        rhos.append(r)
        nan = float("nan")
        out[ds] = (
            float(np.mean(rhos)) if rhos else nan,
            float(np.min(rhos)) if rhos else nan,
            float(np.max(rhos)) if rhos else nan,
        )
    return out


def sharing_text(sharing: dict[str, tuple[float, float, float]]) -> str:
    """``cross_codec_sharing.txt`` content."""
    return (
        "\n".join(
            f"{k}: mean_pairwise_rho={v[0]:.2f} range[{v[1]:.2f},{v[2]:.2f}]"
            for k, v in sharing.items()
        )
        + "\n"
    )


def contrast(df: pd.DataFrame) -> pd.DataFrame:
    """Easiest vs. hardest image-decile means on Color FERET, plus glasses counts."""
    d = df[df.budget == PRIMARY_BUDGET].copy()
    d["difficulty"] = 1.0 - pd.to_numeric(d["id_cos"], errors="coerce")
    sub = d[d.dataset == "colorferet"]
    g = sub.groupby(["subject", "image"])
    di = g["difficulty"].mean().rename("difficulty")
    props = g[[*CONTRAST_KEYS[:3], "glasses", *CONTRAST_KEYS[3:]]].first()
    im = props.join(di).dropna(subset=["difficulty"]).sort_values("difficulty")
    k = max(1, len(im) // 10)
    easy, hard = im.head(k), im.tail(k)
    rows = [
        {
            "key": key,
            "easiest": float(pd.to_numeric(easy[key], errors="coerce").mean()),
            "hardest": float(pd.to_numeric(hard[key], errors="coerce").mean()),
            "k": k,
            "n": len(im),
        }
        for key in CONTRAST_KEYS
    ]
    rows.append(
        {
            "key": "glasses",
            "easiest": float("nan"),
            "hardest": float("nan"),
            "k": k,
            "n": len(im),
            "count_easiest": int(pd.to_numeric(easy["glasses"]).sum()),
            "count_hardest": int(pd.to_numeric(hard["glasses"]).sum()),
        }
    )
    out = pd.DataFrame(rows)
    for col in ("count_easiest", "count_hardest"):
        out[col] = out[col].astype("Int64")
    return out


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", default=None, help="default: OUTPUT_ROOT/difficulty/...")
    ap.add_argument("--out-dir", default=None, help="default: OUTPUT_ROOT/difficulty")
    args = ap.parse_args(argv)
    setup_logging()
    out_dir = Path(args.out_dir) if args.out_dir else config.output_dir("difficulty")
    csv = Path(args.csv) if args.csv else out_dir / "sample_difficulty.csv"
    df = pd.read_csv(csv)
    log.info(
        "loaded %d rows, codecs=%s, budgets=%s",
        len(df),
        sorted(df.codec.unique()),
        sorted(df.budget.unique()),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    correlations(df).to_csv(out_dir / "difficulty_correlations.csv", index=False)
    sharing = cross_codec_sharing(df)
    (out_dir / "cross_codec_sharing.txt").write_text(sharing_text(sharing))
    for ds, v in sharing.items():
        log.info("cross-codec sharing %s: mean rho=%.2f [%.2f, %.2f]", ds, *v)
    contrast(df).to_csv(out_dir / "difficulty_contrast.csv", index=False)
    log.info("wrote the aggregates to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
