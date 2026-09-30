# SPDX-License-Identifier: MIT
"""Evaluation library: verification accuracy, significance, fairness, image quality.

Modules
-------
embeddings
    Source tags, model selection, embedding loading and ``embeddings/manifest.csv``.
verification
    Pair sampling, cosine scoring, EER, FNMR@FMR, subject-cluster bootstrap CIs and
    the ``metrics.csv`` rows.
significance
    Paired McNemar and DeLong tests per cell, BH-FDR, Friedman / Kendall's W /
    Holm-Wilcoxon / Cliff's delta over matchers.
fairness
    Per-subgroup EER and FMR, disparity measures and their cluster-bootstrap CIs.
quality
    PSNR, SSIM, MS-SSIM, LPIPS and DISTS of decoded crops (needs torch, pyiqa, piq).

Importing the package does not import torch; see ``docs/metrics.md`` for the metric
definitions.
"""
