# Evaluation metrics

This page defines the evaluation metrics of the study. They are computed by the
library package `face1kb.eval`; the experiment scripts under `experiments/` are thin
command-line wrappers around it. The definitions below are the ones the code
implements. Where the paper text says something different, the code is what produced
the published numbers (see [Known differences from the paper text](#known-differences-from-the-paper-text)).

| Module | Contents |
|---|---|
| `face1kb.eval.embeddings` | source tags, model selection, embedding arrays, `embeddings/manifest.csv` |
| `face1kb.eval.verification` | pair sampling, cosine scores, EER, FNMR@FMR, bootstrap CIs, `metrics.csv` rows |
| `face1kb.eval.significance` | McNemar, DeLong, BH-FDR per cell, Friedman / Kendall's W / Holm-Wilcoxon / Cliff's delta |
| `face1kb.eval.fairness` | subgroup attributes, per-subgroup EER and FMR, disparity, disparity CIs |
| `face1kb.eval.quality` | PSNR, SSIM, MS-SSIM, LPIPS, DISTS |

Everything except `quality` runs on the CPU with NumPy, pandas and SciPy. Pair scoring
can optionally run on a CUDA GPU through torch; the published accuracy grid and
subgroup EERs were scored that way (see *Reproducibility*). `quality` needs torch,
`pyiqa` and `piq` (the `eval` extra), and normally a GPU.

## Inputs

All paths come from `face1kb.config`:

- `DATA_ROOT/<dataset>/index.csv` has one row per aligned crop (`id, rel_path, subject[, pose]`).
  Its row order is the canonical image order.
- `DATA_ROOT/<dataset>/pairs.parquet` has the columns `idx1, idx2, label`, where the
  indices are rows of `index.csv` and `label` is 1 for mated pairs and 0 for non-mated
  pairs. Color FERET has 64.2 M pairs and AI-Solutions-KK has 153.7 M.
- `WORK_ROOT/<dataset>/embeddings/<model>/<tag>.npy` holds one float32 array of shape
  `(N, 512)` per source. Row `i` is image `i` of the index. A NaN row marks a crop that
  could not be embedded. Embeddings are stored raw and L2-normalised only when they are
  scored. The accuracy, significance and fairness drivers check that every array
  has one row per index entry (per attribute entry for the fairness drivers) and
  raise `ValueError` otherwise (`embeddings.check_rows`).
- Source tags have two forms:
  - `aligned_<res><suffix>` for the uncompressed aligned crops;
  - `<codec>_<res><suffix>_<budget>` for a compressed cell.

  The core grid has an empty suffix. The suffixes select crop variants: `_tight`,
  `_mid` and `_fill`; `_A1` to `_C2`; and `_adv_<attack>_<eps>`.
  `embeddings.parse_tag` splits a tag. It resolves the longest codec name first, so
  `jpeg_xl_...` is not read as `jpeg` with a suffix.
- `embeddings/manifest.csv` lists every array. Its columns are `dataset, model,
  source_tag, kind, res, codec, budget, n_images, n_ok, n_missing, dim`.
  `embeddings.scan_manifest` rebuilds it from the arrays on disk, and `read_manifest` /
  `write_manifest` read and merge it.

Scoring is **symmetric: compressed-vs-compressed.** Both members of every pair are
taken from the same embedding array. For a compressed source, the probe and the
reference have both gone through the codec. The `aligned` rows are the
original-vs-original reference. The recompression, preprocessing and resolution
studies and the adversarial EER rows score the same way. Only the ISO/IEC 29794-5
annex study additionally reports an asymmetric protocol (compressed reference
against an uncompressed probe); it uses its own estimators and is documented with
that study. Per-image identity similarities (a reconstruction against its own
original, `verification.paired_cosines`) are not verification scores.

## Verification accuracy (`metrics.csv`)

**Trials.** All mated pairs are scored. Non-mated pairs are sampled once per dataset,
with `np.random.default_rng(seed).choice(nonmated_positions, size, replace=False)`
and `seed = 0` (`verification.sample_pairs`). The same trials are used for every
codec, budget and matcher of a study. The sample size depends on the study
(`verification.NONMATED_SAMPLE`):

| Study | Non-mated pairs |
|---|---|
| accuracy grid (`metrics.csv`) | 5,000,000 |
| paired significance tests | 2,000,000 |
| subgroup EER (fairness) | 5,000,000 |
| differential FMR (fairness) | 300,000 |
| disparity CIs (fairness) | 400,000 |

The sample is drawn from positions in `pairs.parquet`, so the numbers can be reproduced
only with an identical index row order and an identical pair table. The index is sorted
by `(subject, stem)`, and the pair builder is deterministic.

**Score.** The score of a pair is the cosine similarity
`s = <e1/(|e1|+1e-12), e2/(|e2|+1e-12)>`, computed in float32 (`cosine_scores`). If
either member of a pair is a NaN row, its score is NaN. NaN scores are dropped before
any metric is computed.

**EER.** Let `P` be the genuine scores and `N` the impostor scores. Take 4000
thresholds `t` from `np.linspace(min(P ∪ N), max(P ∪ N), 4000)`, evaluated in
float32 (`verification.eer_grid`, see *Reproducibility*). At each threshold,
FNMR(t) = #{p < t}/|P| and FMR(t) = #{n ≥ t}/|N|. At the grid point `j` that minimises
|FNMR − FMR|, EER = (FNMR(t_j) + FMR(t_j)) / 2 (`verification.eer`). The significance
tests use `t_j` itself as each source's decision threshold (`eer_threshold`).

**FNMR @ FMR = f** (f ∈ {1e-2, 1e-3, 1e-4}). Let `m = round(f·|N|)` and `t` be the
`m`-th largest impostor score, i.e. `sorted(N)[|N| − m]`. Then FNMR = #{p < t}/|P| and
`fmr_realised_f = m/|N|`. If `m < 10`, too few impostors lie above the operating point
for it to be estimated, and both values are NaN. For example, the 3,375-impostor
subsets of the CompressAI and adversarial pilot cells have NaN at 1e-4.

**Confidence intervals.** The CIs are 95 % percentile intervals
(`np.nanpercentile(.., [2.5, 97.5])`) from a subject-cluster bootstrap
(`verification.bootstrap_ci`, `level="subject"`):

1. Cap the impostors at 1,000,000 with a seeded subsample.
2. Factorise the subject labels (the `subject` column of the index) to codes in sorted
   order.
3. In each of 200 replicates, draw `K` subjects with replacement and count the draws
   of each subject. A mated pair is repeated as often as its shared subject was drawn,
   and an impostor pair as often as the subject of its `idx1` image was drawn.
4. Recompute EER and each FNMR@FMR on the weighted scores.

The cap, then the replicates, use one generator, `default_rng(0)`. The point estimate
always uses all impostors. The grid computes CIs only for the clean cells (empty
suffix) of `arcface_antelopev2`, `lvface_l`, `topofr_r100`, `edgeface_xs` and
`cvlface_ir101`.

**Reference and delta.** `eer_aligned` is the EER of the clean aligned source with the
same `(dataset, model, res)`, and `delta_eer = eer − eer_aligned`
(`attach_delta_eer`).

**Columns** (`verification.METRICS_COLUMNS`):

- `dataset, model, tag, kind, res, suffix, codec, budget`. `codec` is empty for
  aligned rows, and `budget` is 0.
- `n_pos, n_neg, eer, fnmr_<f>, fmr_realised_<f>`
- `eer_lo, eer_hi, fnmr_<f>_lo, fnmr_<f>_hi`
- `median_bytes, eer_aligned, delta_eer`

`median_bytes` is not filled by the scorer: it is NaN in the published grid. File
sizes are in `codec_comparison/file_size_summary.csv`.

```python
from face1kb.eval import embeddings, verification as V

models = embeddings.resolve_models("anchor", "colorferet")
rows = V.accuracy_rows("colorferet", models, device="cuda")  # 5 M impostors, CIs
df = V.metrics_frame(rows)                                   # adds eer_aligned / delta_eer
# metrics.csv: one finished frame per dataset, in order (see "CSV text" below)
V.write_metrics_csv([df], "metrics.csv")
```

`accuracy_rows(..., device="cuda")` scores on the GPU. If it runs out of memory, it
halves the chunk and retries, down to 25 k pairs. If even that does not fit (for
example on a GPU shared with other jobs), the default `cpu_fallback=True` logs a
warning and scores that source with the NumPy scorer, whose numbers can differ from
the published ones by one trial (see below); pass `cpu_fallback=False` to raise
`RuntimeError` instead. All `*_rows` drivers take this argument. **The published
grid was scored on CUDA.** The NumPy
scorer (`device=None`) computes the same formula, but its float32 sums differ from
the CUDA ones in the last bit. As a result a few cells differ by one trial, for
example the EER of Color FERET / LVFace-L / Ours-FAST 112 px 512 B: 1.221550 % on
CUDA (the published `metrics.csv`; 1.22 in the paper tables), 1.221540 % on the CPU
(see *Reproducibility*).

## Paired significance tests (`significance_<dataset>.csv`)

Each cell `(dataset, model, res, budget)` holds the aligned reference and every codec
that has embeddings. The test procedure (`significance.significance_rows`):

1. Score every source on one shared trial set: all mated pairs plus 2,000,000 sampled
   non-mated pairs.
2. Drop any source with fewer than 50 % valid (non-NaN) trials.
3. Restrict the remaining sources to the trials that are valid for all of them, so
   every comparison is exactly paired (`finalize_cell`).
4. Make each source decide at its own EER threshold on that trial set. A mated trial
   is correct when `s ≥ t`, a non-mated trial when `s < t`.
5. For every unordered pair of sources, in sorted names:
   - **McNemar** (`mcnemar`). `b` counts trials where A is right and B wrong; `c`
     counts the reverse. The statistic is `χ² = (|b − c| − 1)² / (b + c)`, with a χ²(1)
     p-value. The exact two-sided binomial p is computed but is not in the CSV.
   - **DeLong** (`delong_paired`). This is the paired test of the ROC-AUC difference
     (mated = positive), using the structural components of Sun & Xu (2014); the
     p-value is two-sided normal. When the variance is zero, p = 1 for equal AUCs and 0
     otherwise.
     The mid-ranks are computed vectorised (`midrank`). The result is identical to the
     rank-by-rank loop (half-integer ranks `0.5·(i+j−1)+1` per tie block).
6. **BH-FDR** (`bh_fdr`, `correct_family`). Adjust the McNemar p-values with
   Benjamini-Hochberg within **one cell** (`CELL_FAMILY`: 55 tests for 11 sources).
   `significant = p_adj < 0.05`.
   - `correct_family(df, ("dataset", "model"))` pools all cells of a matcher instead.
     On `significance_kk.csv` that would shift `p_adj` by up to 0.0099 without flipping
     any decision.

`significance_frame(rows)` returns the published columns: `dataset, model, res,
budget, codec_a, codec_b, n_trials, acc_a, acc_b, b, c, mcnemar_chi2, mcnemar_p, auc_a,
auc_b, auc_diff, auc_test, auc_p, p_adj, significant`. Because the family is one cell,
cells computed in separate runs can simply be concatenated. When no requested cell has
two usable sources (for example a resolution or budget filter that matches nothing),
`significance_rows` returns an empty list and `significance_frame` an empty frame
with these columns; `verification.metrics_frame([])` behaves the same way.

One cell of 11 sources and 55 tests takes 5-15 minutes on one CPU core, pair loading
included. The vectorised mid-ranks take 0.7 s per 2.1 M scores; the
rank-by-rank loop takes 5-6 s. `auc_bootstrap_paired` is a bootstrap alternative to
DeLong; the published tables do not use it.

### Rank statistics over matchers (`posthoc_scalars.txt`, `posthoc_stats.tex`)

`significance.eer_matrix(metrics)` builds the matcher × codec EER matrix of Color FERET
at 112 px and 1024 B. It keeps only the matchers that have every codec; there are 14
matchers × 12 codecs. `significance.posthoc(matrix)` returns:

- the Friedman χ² and its p-value;
- Kendall's `W = χ² / (n (k − 1))`;
- the mean rank of each codec (1 = lowest EER, ties averaged);
- for the headline pairs, the Wilcoxon signed-rank p-value with Holm step-down
  correction (a failing Wilcoxon counts as p = 1), and Cliff's delta
  `(#(a_i > b_j) − #(a_i < b_j)) / n²` over all matcher combinations.

`eer_matrix(..., exclude_prefix="edgeface")` drops the 4 EdgeFace matchers (10
remain). A codec whose rows are all NaN in the cell (an empty cell) is left out of the
matrix; if no matcher has an EER for every remaining codec, `eer_matrix` raises
`ValueError`.

## Fairness (`fairness_*.csv`, `fmr_fairness_*.csv`, `disparity_ci_kk.csv`)

**Attributes** (`fairness.load_attributes`) give one value per index row.

- Color FERET: the NIST ground-truth labels (`labels.csv`), joined on the image stem.
  - `pose`: the coarse class of the textual pose (`pose_class`). The first rule that
    matches wins: profile (`profile`, `75 degree`), quarter (`quarter`, `45 degree`),
    half, frontal (`frontal`, `15 degree`).
  - `skin_tone`: the `race` field.
  - `gender`.
  - `age`: the age at capture, binned as ≤25, 26-35, 36-50, 51-65, >65.
- AI-Solutions-KK: the estimated attributes (`attributes.csv`), propagated per
  identity. KK has no pose attribute.
  - `skin_tone`: the first non-empty Monk Skin Tone label of the identity.
  - `gender`: the identity's mode.
  - `age`: the band of the identity's median age.

**Subgroup EER** (`subgroup_rows`, `fairness_rows`). A pair belongs to subgroup `g` if
both of its images carry `g`.

- Per attribute there is an `__overall__` row. It uses all pairs whose two ends carry
  the same non-null value, the population that the subgroups partition.
- Per subgroup there is one row if it has at least 50 mated pairs and at least one
  impostor pair.
- The EER is the one defined above.
- The **disparity** of an attribute (`disparity_row`) is `eer_max − eer_min` (and
  `eer_std`) over the subgroups with at least 1,000 impostor pairs. Thinner subgroups
  are dropped from the spread; on Color FERET this removes three ethnicity cohorts.
- The 512 B and 1024 B disparity tables in percentage points are
  `subgroup_spread_pp` for the uniform KK Monk basis MST5/6/7/9/10 and `disparity_pp`
  for Color FERET.

**Differential FMR** (`fmr_rows`, `fmr_fairness_rows`). This analysis uses 300 k
non-mated pairs at 112 px for 512 B, 1024 B and aligned.

- Take the attribute's within-subgroup impostor pool.
- Set a global threshold `τ` at its `1 − target` quantile (`np.quantile`, linear
  interpolation, evaluated in float32 by `verification.linear_quantile`), with a
  target of 1e-2 or 1e-3.
- For every subgroup with at least 50 impostor pairs, report
  FMR(g) = P(s ≥ τ | both ends in g).
- The `__overall__` row carries the pooled FMR and `fmr_disparity_pp = (max − min)·100`.
  It is written only when at least two subgroups qualify.
- **NaN scores.** If any impostor pair of the pool touches a crop without an
  embedding, `τ` is NaN and every FMR is 0 (`nan_policy="propagate"`, the default,
  which is what the published CSVs contain). This affects the Color FERET JPEG-AI
  cells (4 crops without embeddings), the CompressAI subsets and the
  AI-Solutions-KK JPEG XL 112 px / 512 B cell (1 crop). `nan_policy="omit"` drops
  those pairs instead. It leaves every other codec unchanged and gives, for example,
  a Color FERET JPEG-AI ethnicity ΔFMR of 2.62 / 1.05 / 2.10 / 1.62 pp (ArcFace /
  LVFace-L / TopoFR-R100 / EdgeFace-XS, 112 px, 1024 B, target 1e-2).

**Disparity CIs** (`disparity_ci_rows`). These cover KK, 112 px and 1024 B, with 400 k
non-mated pairs and 500 replicates at seed 0.

- The statistic is the max − min EER in percentage points over MST5/6/7/9/10
  (`uniform_disparity`). A subgroup needs 20 mated pairs and one impostor pair;
  otherwise the replicate value is NaN and is dropped.
- A replicate draws identities with replacement (`cluster_bootstrap_masks`). A pair is
  kept if the identities of both of its ends were drawn at least once.
- In this analysis a mated pair belongs to `g` through its identity, and a non-mated
  pair through the skin tone of its **first** image (`idx1`) only. The impostors of a
  subgroup therefore include cross-tone pairs, so the point values differ from the
  within-subgroup disparities of `fairness_kk.csv`. For example, ArcFace aligned gives
  0.35 pp here and 0.37 pp there.
- The amplification ratio is `Δ_codec / median(Δ_aligned bootstrap)`. Its CI is the
  percentile interval of `Δ_codec,b / max(Δ_aligned,b, 1e-6)` over replicates.

## Per-image identity similarity (`idcos_tail_512.csv`)

`verification.paired_cosines(ref, rec)` is the cosine between the embedding of each
reconstruction and that of its own aligned original (rows where either is NaN are
dropped). The identity-tail table reports, per codec at 112 px and 512 B under
ArcFace, the number of crops `n`, the 5th percentile
(`verification.linear_quantile(cos, 0.05)`) and the median (`np.median`). All 10
published rows are reproduced exactly this way.

## Image quality (`quality_<dataset>.csv`)

Every decoded crop is compared with its lossless aligned PNG. Both are RGB uint8,
scaled to [0, 1] (`quality.make_metrics`):

| Column | Implementation | Better |
|---|---|---|
| `psnr` | `pyiqa.create_metric("psnr")`: RGB, no border crop | higher |
| `ssim` | `pyiqa.create_metric("ssim")`: on the Y channel (`test_y_channel=True`), no downsampling | higher |
| `ms_ssim` | `piq.multi_scale_ssim(kernel_size=7, scale_weights=(0.0448, 0.2856, 0.3001, 0.2363), data_range=1)`: 4 scales, weights renormalised by piq, per RGB channel | higher |
| `lpips` | `pyiqa.create_metric("lpips")`: AlexNet, v0.1 | lower |
| `dists` | `pyiqa.create_metric("dists")` | lower |

- **MS-SSIM configuration.** MS-SSIM uses 4 scales and a 7 px kernel. piq needs
  `(kernel − 1)·2^(scales − 1) + 1` pixels, which is 161 px for the standard
  5-scale / 11 px configuration and 49 px for this one. One configuration therefore
  covers every crop size from 64 to 224 px.
- **Weights.** pyiqa downloads the LPIPS and DISTS weights on first use, into the torch
  hub cache.
- **Batching and aggregation.** Images are scored in batches of 64 (`score_images`).
  `aggregate_quality` reduces the per-image records to the **median** of each metric
  per `(codec, res, budget)` and adds the image count `n`. The per-image records are
  local intermediates and are not published.
- **GPU cost.** Scoring the five metrics (batch 64, tensor conversion included,
  decoding excluded) takes about 0.8 ms per crop at 64 px, 1.7 ms at 112 px and
  7.5 ms at 224 px on one RTX 2080 Ti, i.e. under 2 minutes for an 11 k-crop cell;
  decoding dominates the cost of a cell (JPEG-AI and the learned codecs in
  particular).

Decoding of the codecs is not part of this module; see the baseline codecs' decode
dispatcher.

**mbt2018-mean rows.** The published CompressAI mbt2018-mean quality medians (Color
FERET, 112 px, 512 B and 1024 B; the mbt2018 row of the paper's quality-matrix table)
were scored on crops decoded once on CUDA without checking the decoded latent. For 7
of the 300 crops at 512 B and 9 of the 300 at 1024 B, that decode lost
synchronisation with the bitstream, and the scored image is partly or entirely noise
(per-image PSNR as low as 8.4 dB). The public decoder checks the latent and
reconstructs these crops correctly, so a rerun gives slightly better medians than the
published ones:

| mbt2018-mean, 112 px | PSNR | SSIM | MS-SSIM | LPIPS | DISTS |
|---|---|---|---|---|---|
| 1024 B, published | 33.85 | 0.897 | 0.983 | 0.118 | 0.162 |
| 1024 B, verified decode | 33.89 | 0.899 | 0.983 | 0.114 | 0.161 |
| 512 B, published | 30.99 | 0.845 | 0.970 | 0.203 | 0.207 |
| 512 B, verified decode | 30.99 | 0.846 | 0.970 | 0.203 | 0.207 |

The metric code is not the cause: on the same decoded pixels, `face1kb.eval.quality`
gives the values of the study's scoring code. An unverified decode (`verify=False`)
reproduces the published records only by chance, because the desynchronisation does
not repeat from run to run. The shipped `quality_colorferet.csv` keeps the published
values. The bmshj2018 cells have no such crops. Their reruns differ from the
published medians only at about 1e-4, because the learned decoders are not bit-exact
from run to run on CUDA.

## Library usage

The experiment scripts wrap these calls; the snippets show the library directly.

```python
import pandas as pd
import torch
from face1kb.eval import fairness as F, quality as Q, significance as S
from face1kb.eval import verification as V

# paired tests of one cell (NumPy scorer, 2 M impostors, BH per cell)
sig = S.significance_frame(
    S.significance_rows("colorferet", ["edgeface_xs"], res=[112], budgets=[1024])
)

# subgroup EER + disparity, differential FMR, disparity CIs (CUDA scorer)
dev = torch.device("cuda")
fair, disp = F.fairness_rows("colorferet", res=[112], budgets=[1024], device=dev)
fmr = F.fmr_fairness_rows("colorferet", device=dev)
dci = F.disparity_ci_rows("kk", device=dev)

# rank statistics from a metrics.csv frame
ph = S.posthoc(S.eer_matrix(metrics_df))

# image quality of decoded crops: items = [(record, decoded_rgb, reference_rgb), ...]
records = Q.score_images(items, Q.make_metrics("cuda"))
summary = Q.aggregate_quality(pd.DataFrame(records), "colorferet")
```

Pair tables are read once per call; pass `pairs=V.pair_indices(dataset, n, 0)` to
reuse them across calls with the same sample size. Measured on one RTX 2080 Ti and
one CPU core: scoring a source on the 5 M-impostor trial set takes about 1 s on the
GPU and 20-40 s with NumPy; loading a pair table takes 10-50 s (64 M / 154 M rows);
the 200-replicate subject bootstrap of one cell takes about 7 s on Color FERET and
16 s on AI-Solutions-KK; one 500-replicate disparity-CI row takes about 30 s.

## Reproducibility

- **Float32 evaluation, independent of the NumPy version.** Scores are float32. The
  published numbers were computed under NumPy 2.4 (2.4.6). Its promotion rules
  (NEP 50) evaluate:
  - `np.linspace` over float32 endpoints in float32. Every NumPy 2 release does
    this. NumPy 1.x uses float64; for typical score ranges about half of the 4000
    grid points then differ, which can move an EER by one trial.
  - `np.quantile` of a float32 array with a Python-float `q` from a float64 virtual
    index `(n − 1)·q`, with the interpolation in float32. This holds from NumPy 2.4
    on. NumPy 1.x interpolates in float64, and NumPy 2.0-2.3 cast `q` to float32 and
    so form the virtual index in float32. Either can move an FMR threshold (`τ` of
    the differential-FMR analysis) or a 5th percentile by one float32 step.

  `verification.eer_grid` and `verification.linear_quantile` perform this
  arithmetic explicitly, so the published values come out under any NumPy version.
  In 300,000 randomised cases, `eer_grid` is identical to `np.linspace` and
  `linear_quantile` to `np.quantile` of NumPy 2.4. `eer_grid` also matches
  `np.linspace` of NumPy 2.0, 2.2 and 2.3 (2,000 cases each). Both helpers give the
  same values under NumPy 1.26 and 2.0-2.3, where `np.quantile` itself differs.
- **Scorer.**
  - The CUDA scorer (`device="cuda"`, torch float32, 500 k-pair chunks) produced the
    published `metrics.csv` and all fairness CSVs.
  - The NumPy scorer produced the significance CSVs.
  - The two scorers differ in the last float32 bit for most pairs. At the resolution
    of the 4000-point grid, this shifts a metric by one trial only when a score lies
    at a grid threshold. It also reorders nearly tied scores, which moves the DeLong
    AUCs by up to about 1e-10 and the DeLong p-values by up to 2e-6 (relative 3e-5);
    the McNemar counts, p-values and decisions do not change.
  - To regenerate `metrics.csv` and the fairness tables, use `device="cuda"` with
    `cpu_fallback=False`, so that a CUDA out-of-memory condition cannot silently
    switch a source to the NumPy scorer. For significance, use the default NumPy
    scorer.
  - CUDA results were validated on an RTX 2080 Ti (torch 2.10, CUDA 12.8).
- **CSV text.** The published `metrics.csv` and `significance_*.csv` were assembled
  from per-dataset or per-cell files that were read back with pandas' default float
  parser and written again. That parser is not exactly round-trip, so some printed
  values differ from the freshly computed ones in the last digit (e.g. an FNMR of
  9/95839 is printed as `9.390749068750715e-05`, not `...717e-05`). After the same
  single read/write, recomputed rows are string-identical to the published files.
  `verification.write_metrics_csv(frames, path)` and
  `significance.write_significance_csv(frames, path)` apply exactly this: pass the
  finished per-dataset `metrics_frame` (or per-shard `significance_frame`) outputs,
  and they are written, read back, concatenated and written once. Derived columns
  (`eer_aligned`, `delta_eer`, `p_adj`, `significant`) must be computed on the
  in-memory values *before* this round trip: re-running `metrics_frame` /
  `attach_delta_eer` or `correct_family` on parsed values changes their printed text
  (e.g. `delta_eer` in about two thirds of the rows).
  The fairness CSVs were written directly and match without it. Of the quality
  CSVs, `quality_kk.csv` is the direct `aggregate_quality` output and
  `quality_colorferet.csv` carries one such read/write. `quality_summary.csv` was
  built by re-reading the per-dataset CSVs from disk and concatenating them, at a
  time when both had been through one read/write, so both of its parts carry two.
  The values are exact in every case; the parser changes at most the last printed
  digit (below 4e-15).
- **Tests.** `pytest tests/eval` runs the CPU tests on synthetic data. The
  data-marked test `test_recompute_published_cell` recomputes one Color FERET cell
  from real data. It runs only with `FACE1KB_SLOW=1` (it reads the 64 M-pair table)
  and needs `FACE1KB_DATA_ROOT` / `FACE1KB_WORK_ROOT` with the Color FERET index,
  `pairs.parquet` and the ArcFace `aligned_112` embeddings, plus
  `results/accuracy/metrics.csv` (or `FACE1KB_OUTPUT_ROOT/accuracy/metrics.csv`).
- **Row order and subject labels.** The pair sample depends on the row order of the
  pair table. The bootstrap depends on the sort order of the subject labels (for
  Color FERET, integer ids and zero-padded directory names sort the same).
- **Validation.** The library was checked against the published aggregates, reading
  the study's embeddings, pair tables and labels:
  - `metrics.csv`: 49 rows over both datasets, 9 matchers, 11 codecs, 5
    resolutions, 3 budgets (512, 768, 1024 B), crop-variant, annex and adversarial
    suffixes, the NaN operating-point path and the empty-cell path, with the
    200-replicate bootstrap CIs of 5 clean cells and `eer_aligned` / `delta_eer`.
    With the CUDA scorer all 981 recomputed values are string-identical to the
    published file after the read/write described above. With the NumPy scorer 971 of them are; the
    EERs of 4 cells move by one impostor trial (1e-7), together with the
    `eer_aligned` / `delta_eer` values that depend on them;
  - significance: two full cells of 55 rows each (Color FERET / EdgeFace-XS 112 px
    and AI-Solutions-KK / LVFace-L 224 px, 1024 B), string-identical after the
    read/write; the vectorised mid-ranks are identical to the rank-by-rank loop on
    the 2.1 M-trial score vectors of the Color FERET cell and on random data with
    ties;
  - fairness: 440 subgroup-EER rows and 85 disparity rows (Color FERET / EdgeFace-XS
    and AI-Solutions-KK / ArcFace, 112 px, aligned and 1024 B), 900 differential-FMR
    rows and all 40 disparity-CI rows, bit-identical with the CUDA scorer;
  - image quality: the study's per-image records of 384 crops over 6 cells (both
    datasets, 64-224 px, WebP, AVIF, HEIF, JPEG XL, JPEG-AI, and Ours-ACCURATE
    decoded with the public codec API). `face1kb.eval.quality` and the scoring code
    that produced the study's records give identical values on the same GPU.
    Against the study's records, PSNR, SSIM, MS-SSIM and DISTS match exactly except
    two PSNR values one float32 step apart (2e-6 dB), and LPIPS matches to within
    6e-8; the study's scoring code shows the same differences when rerun, so they
    come from the GPU that produced the records. Re-aggregating the study's records
    reproduces all 108 + 94 rows of `quality_colorferet.csv` / `quality_kk.csv`.
    The mbt2018-mean cells are the exception for a rerun from the bitstreams,
    because their decodes differ (see *mbt2018-mean rows* above);
  - `embeddings/manifest.csv`: 11 rows rebuilt from the arrays are identical;
  - `idcos_tail_512.csv`: all 10 rows exact;
  - the Friedman / Kendall's W / Holm-Wilcoxon table (χ² = 130.6, W = 0.848; 0.893
    without the EdgeFace family).

## Known differences from the paper text

- **Pairing scenario (§4.2).** The paper says results are compressed-vs-original
  unless stated otherwise. The code scores every accuracy, significance and fairness
  cell compressed-vs-compressed, as described above.
- **Quality aggregation (§4.3).** The paper says the quality pipeline "averages over
  the cell". The published quality tables are per-cell **medians**.
- **mbt2018-mean image quality (quality-matrix table).** The paper row for mbt2018 at
  112 px and 1024 B (PSNR 33.85, SSIM 0.897, MS-SSIM 0.983, LPIPS 0.118, DISTS 0.162)
  includes 9 of 300 crops whose decode lost synchronisation with the bitstream. With
  the verified public decoder the row is 33.89 / 0.899 / 0.983 / 0.114 / 0.161 (see
  *mbt2018-mean rows*).
- **`median_bytes` (§4.4).** The paper says this column records the median encoded
  size. It is NaN in `metrics.csv`; sizes are in `file_size_summary.csv`.
- **Disparity CIs (§11.4).** The paper attributes the gap between the CI table and the
  5 M-impostor disparity table to the 400 k impostor sample. The impostor subgroups of
  the CI analysis are also defined by the first image only (see above).
- **McNemar threshold (§4.5).** The paper says McNemar compares the decisions "at a
  fixed operating threshold". In the code each source decides at its own EER threshold
  on the shared trial set.
- **Cliff's delta (§4.5).** The paper describes it as the probability that codec `a`
  has the lower EER "on a randomly chosen matcher, minus the reverse". The code
  computes the unpaired two-sample form over all matcher combinations, so −1 means
  that every EER of `a` is below every EER of `b` (a stronger condition than `a`
  winning on every matcher).
- **Disparity-CI values (§11.4, `tab:disparity-ci`).** The aligned, Ours-ACCURATE and
  Ours-FAST rows of the paper table match the published `disparity_ci_kk.csv`, which
  the code reproduces. The WebP and JPEG 2000 rows of the table do not match that CSV
  (ratio [95 % CI], ArcFace / LVFace-L / TopoFR-R100 / EdgeFace-XS):

  | Codec | paper table | `disparity_ci_kk.csv` |
  |---|---|---|
  | WebP | 1.4 [0.8,6.0] / 1.3 [0.7,4.1] / 0.8 [0.4,4.8] / 1.5 [0.5,4.4] | 1.6 [1.1,2.9] / 1.1 [0.7,2.3] / 1.1 [0.8,2.5] / 1.1 [0.9,1.6] |
  | JPEG 2000 | 12.7 [4.5,46.6] / 7.6 [4.2,49.7] / 5.6 [3.3,38.8] / 3.4 [1.5,13.8] | 10.7 [4.5,29.4] / 8.7 [3.9,31.3] / 12.6 [4.9,33.9] / 4.3 [2.1,9.5] |

  With the CSV values, the JPEG 2000 lower bounds are 2.1-4.9× (paper text:
  1.5-4.5×), and the ArcFace WebP interval excludes 1× (paper text: "WebP's CI
  includes 1×"). The qualitative reading (JPEG 2000 amplifies the gap, the learned
  codec does not) is unchanged.
- **Differential-FMR roster (§11.6, `tab:fmr-fairness-cf`).** The caption says codecs
  are omitted because "every subgroup FMR quantises to zero" at the target. In the
  code, JPEG-AI and the CompressAI rows are all zero because their impostor pools
  contain NaN scores (crops without embeddings), which makes the threshold NaN (see
  *NaN scores* above); with those pairs omitted JPEG-AI lies inside the band of the
  other codecs.
- **Independent-matcher significance check (§14.2).** The paper states that under
  CVLFace-IR101 (Color FERET, 112 px, 1024 B) Ours-ACCURATE vs AVIF is not
  significant (`p_adj = 0.34`). No published significance file contains that
  matcher. Running the published protocol for that cell (2 M impostors, NumPy
  scorer, BH per cell) gives `χ² = 51.8`, `p_adj = 6.7e-13` (significant);
  the only non-significant pair of the 55 is JPEG-AI vs WebP (`p_adj = 0.88`). The
  other part of the claim holds: `|ΔAUC| ≤ 1.4e-6` for Ours-ACCURATE vs JPEG-AI,
  AVIF and WebP.
