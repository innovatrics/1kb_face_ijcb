# Shipped results and summary figures (`results/`, `experiments/figures/`)

`results/` holds the aggregate result files behind the tables and figures of the paper
(arXiv 2608.22866). They are the published numbers: every table generator of the other
experiment areas, and the summary-figure renderer described below, can read them
instead of a fresh run, so the paper's tables and figures can be rebuilt on a CPU
without the datasets.

The folder has the same file layout as `FACE1KB_OUTPUT_ROOT` (`face1kb.config.OUTPUT_ROOT`,
default `<repo>/outputs`), which is where a full run of the pipeline writes the same
files. `face1kb.config.RESULTS_ROOT` / `config.results_dir(...)` point at it. Treat it as
read-only reference data; a new run writes to `OUTPUT_ROOT`.

Everything in `results/` is an aggregate over a cell (dataset x matcher x codec x
resolution x budget, or a subgroup of one). It holds no images, no per-image or
per-identity values, no file names or paths, no embeddings and no bitstreams. The
datasets themselves are not distributed (see [datasets.md](datasets.md)).

## Commands

```bash
# the seven summary figures (Figures 22, 40-45) and derived_stats.json, from results/
python experiments/figures/render_figures.py --from-results
# the same from a fresh run of the other areas (reads FACE1KB_OUTPUT_ROOT)
python experiments/figures/render_figures.py
# a subset
python experiments/figures/render_figures.py --from-results --only significance_matrix

# rebuild results/ from a finished run (maintainers; review the diff before committing)
python experiments/figures/collect_results.py --src "$FACE1KB_OUTPUT_ROOT" --dst results

# figures (FROM=results|outputs); COLLECT=1 renders from outputs, then collects
experiments/figures/run.sh
```

`render_figures.py` writes PNGs (170 dpi) to `--out-dir` (default
`OUTPUT_ROOT/figures`) and the figure-derived scalars to `--stats-out` (default
`OUTPUT_ROOT/report_summary/derived_stats.json`). It never writes into `results/`.
`--only` renders a subset and then skips `derived_stats.json`.

`collect_results.py` takes each file from the first `--src` root that has it (the
option is repeatable), and looks for the `fairness/` files also under `accuracy/`.
Some shipped files are not produced by the public pipeline (`codec_properties.csv`,
`codec_comparison/res*/comparison.json`, `contamination/contamination_summary.csv`,
the findings register); without `--keep-shipped` a missing input is an error, with it
the shipped copy is kept. `run.sh` with `COLLECT=1` first renders the figures and
`derived_stats.json` from `FACE1KB_OUTPUT_ROOT` (it is one of the collected files) and
then runs `collect_results.py --keep-shipped`.

Cost: CPU only, about 10-15 s for all seven figures and under 1 GB of RAM;
`collect_results.py` takes a few seconds. Both need pandas and SciPy (core
dependencies); rendering needs matplotlib (the `eval` extra).

With the paper environment (matplotlib 3.11.0, `requirements/paper.txt`) the seven PNGs
rendered from `results/` are byte-identical to the figures of the arXiv source, and
`derived_stats.json` is byte-identical to the shipped copy. Other matplotlib or font
versions render the same content, but the pixels can differ.

## Summary figures

| File | Paper | Input (relative to the input root) |
|---|---|---|
| `quality_vs_identity.png` | Figure 22 (Section 6) | `accuracy/metrics.csv` (Color FERET, ArcFace, 112 px, compressed rows without a crop-variant suffix) and `quality/quality_summary.csv` (cells with at least 100 crops) |
| `fairness_disparity.png`, `fairness_disparity_512.png` | Figures 40, 41 (Section 11) | `fairness/fairness_kk.csv` (Monk skin tone, uniform MST5/6/7/9/10 basis, four anchors) and `fairness/fairness_disparity_colorferet.csv` (ethnicity and pose, EdgeFace-XS), 112 px, 1024 B / 512 B |
| `recompression_heatmap.png` | Figure 42 (Section 12) | `recompression/recompression_{colorferet,kk}_{1024,512}.csv` |
| `sanitization_residual.png` | Figure 43 (Section 13) | `adversarial/sanitization.csv` (HFC attack, mean over the four anchors) |
| `significance_matrix.png` | Figure 44 (Section 14) | `accuracy/significance_colorferet.csv`, the cell EdgeFace-XS / 112 px / 1024 B |
| `codec_mean_rank.png` | Figure 45 (Section 14) | `accuracy/posthoc_scalars.txt` |

`derived_stats.json` holds the Spearman rho / p / n of the two quality-vs-identity
panels (quoted in Section 6) and, per anchor, the KK skin-tone disparity (pp) of the
aligned reference, WebP, Ours-ACCURATE and JPEG 2000 at 112 px / 1024 B on the uniform
basis, with the JPEG 2000 amplification factor (quoted in Section 11).

The fairness disparities are read with `face1kb.eval.fairness.subgroup_spread_pp`
(KK) and `disparity_pp` (Color FERET). The KK panel uses the uniform basis because
the classical codecs lose the MST8 cell under compression while the Ours variants keep
it; the per-attribute rows of `fairness_disparity_kk.csv` would compare spreads over
different subgroup sets. The KK panel shows the aligned reference, the seven classical
codecs and the two Ours variants (not JPEG-AI, although `fairness_kk.csv` has its rows:
0.72 pp for ArcFace at 1024 B); the Color FERET panel adds JPEG-AI. The JPEG 2000
amplification factors of `derived_stats.json` are 13.7x (ArcFace), 11.2x (LVFace-L),
20.0x (TopoFR-R100) and 4.2x (EdgeFace-XS); the Section 11 footnote quotes the range
11.2-20.0x of the first three.

## File reference

Section, table and figure numbers are those of the arXiv paper. "[static]" marks a
paper table that is maintained by hand from the file; all other tables are rendered
by the generator of the owning experiment area.

### `accuracy/`

| File | Content | Paper |
|---|---|---|
| `metrics.csv` | Verification grid: one row per (dataset, matcher, source tag) over the 14 public matchers, all codecs, resolutions 64-224 px and budgets 512/768/960/1024 B, plus the aligned references and the crop-tightness, preprocessing and adversarial variants (`suffix`) | Tables 9-29, 42, 78; Figures 9-19, 22; [static] Table 24 |
| `preproc_through_codec_kk.csv` | EER and identity cosine of preprocessing operators followed by a codec (KK, 224 px / 1024 B, 800 crops) | Tables 47, 48 |
| `significance_colorferet.csv`, `significance_kk.csv` | Pairwise McNemar and DeLong tests with per-cell Benjamini-Hochberg correction: Color FERET 4 anchors at 112 px / 1024 B, KK 4 anchors at 112 and 224 px / 1024 B, 55 codec pairs per cell | Tables 76, 77; Figure 44 |
| `posthoc_scalars.txt` | Friedman chi^2, Kendall's W and the codec mean EER ranks over the 14 x 12 matcher x codec grid (Color FERET, 112 px / 1024 B), derived from `metrics.csv` | Figure 45 |

### `fairness/`

Written by the scripts in `experiments/fairness/`.

| File | Content | Paper |
|---|---|---|
| `fairness_colorferet.csv`, `fairness_kk.csv` | EER per subgroup (Color FERET: pose, ethnicity, gender, age; KK: Monk skin tone, gender, age) for the four anchors; `__overall__` is the whole cell | Tables 59, 60, 64; Figures 40, 41 |
| `fairness_disparity_colorferet.csv`, `fairness_disparity_kk.csv` | Min / max / std of the subgroup EERs per attribute and cell, with the number of subgroups that pass the minimum-impostor rule | Tables 61, 62, 64; Figures 40, 41 |
| `fmr_fairness_colorferet.csv`, `fmr_fairness_kk.csv` | Differential FMR: per-subgroup FMR at the global threshold for a target FMR of 1e-2 and 1e-3, and the max-min disparity on the `__overall__` rows | [static] Table 65 (target 1e-2, ethnicity, 112 px / 1024 B) |
| `disparity_ci_kk.csv` | Subject-level cluster-bootstrap CIs of the KK skin-tone disparity and its ratio to the aligned reference (112 px / 1024 B) | [static] Table 63 |

The column definitions of the `accuracy/` and `fairness/` files are in
[metrics.md](metrics.md).

### Other folders

| File | Content | Paper |
|---|---|---|
| `ablation/s7_ablation.csv`, `.json` (same rows) | Side-stream ablation of the ACCURATE codec: checkpoint (`control`, `deployed`, `noside`, `topofr`) x matcher x dataset x budget; id-cos median / mean, PSNR median, n = 200 | Section 7 (side-stream ablation) |
| `ablation/preprocessing_ablation.csv` | Clean (uncompressed) Color FERET 112 px EER per anchor matcher of the crop presets (`study=crop`: standard, tight, mid, fill) and the preprocessing variants (`study=preproc`: std, A1-A4, B1, B2, C1, C2), with the identity cosine to the standard crop and the genuine / impostor pair counts | Tables 49, 50 |
| `adversarial/sanitization.csv` | Attack strength and post-codec sanitization per (dataset, matcher, attack, eps, codec, budget) with the crop count `n` | Tables 70-75; Figure 43 |
| `annex/scores.csv`, `scores_pop2000.csv`, `ci.csv` | ISO/IEC 29794-5 Annex E/F study: per-cell EER / FNMR (symmetric and asymmetric protocols, all and frontal pairs), bytes, fit rate, self-similarity; Color FERET re-scored on the 2000-crop prefix; bootstrap CIs | Tables 51-58 |
| `codec_properties.csv` | Provenance, licensing, method, chroma, rate control, implementation, footprint, <= 1 kB fit, hardware / browser support, determinism and maturity of the ten baseline codecs | [static] Table 2 |
| `codec_comparison/file_size_summary.csv` | Achieved file size per cell: n, min / mean / median / max, p5 / p95, share over budget, share holding the budget | Tables 9-16 (bytes), 30-32; Figure 20 |
| `codec_comparison/jpegai_decoders.csv` | JPEG-AI operation points SOP / BOP / HOP on CPU and GPU: encode / decode latency, bytes, PSNR, SSIM, LPIPS, id-cos (24 crops) | [static] Table 37; Figure 30 |
| `codec_comparison/jpegai_kk.csv` | JPEG-AI operation points on KK (600 crops): EER, id-cos, PSNR, SSIM, bytes for ArcFace and EdgeFace-XS | Table 38 |
| `codec_comparison/res112/comparison.json`, `res224/comparison.json` | Per (dataset, codec, budget) median identity cosine, PSNR, SSIM, LPIPS and bytes on 64 held-out crops | Tables 7, 8, 40, 41 |
| `contamination/contamination_summary.csv` | Train/test overlap, dataset level only: number of evaluation identities, max / median of their maximum cosine to a WebFace42M sample, counts above 0.8 and 0.9; number of crops, pHash Hamming-distance histogram and share within 6 bits | Section 7 (overlap check) |
| `difficulty/difficulty_correlations.csv` | Image-level Spearman rho / p / n between each image descriptor and the per-image difficulty, per dataset | Table 44; Figure 33 |
| `difficulty/difficulty_contrast.csv` | Color FERET easiest vs. hardest difficulty decile (k = 30 of n = 300 crops): mean of each image descriptor, and for glasses the count per decile | Table 45 |
| `difficulty/cross_codec_sharing.txt` | Mean pairwise cross-codec correlation of the per-image difficulty | Section 8 |
| `quality/quality_summary.csv`, `quality_colorferet.csv`, `quality_kk.csv` | Per-cell medians of PSNR, SSIM, MS-SSIM, LPIPS and DISTS with the crop count `n` | Tables 7, 8, 33, 35; Figure 22 |
| `quality/fiq_colorferet.csv`, `fiq_kk.csv` | SER-FIQ-style face image quality per cell and its drop against the original | Table 34; Figure 23 |
| `quality/idcos_tail_512.csv` | 5th percentile and median of the per-image ArcFace identity cosine (decoded vs original) at 512 B, 112 px | Table 43 |
| `quality/codec_verify.csv` | Reconstruction checks of the two face1kb codecs per cell (on KK also per skin-tone tertile, light / mid / dark): byte fill, PSNR, SSIM, brightness shift, colour cast, non-finite outputs | Section 7 |
| `quality/speed_*.csv` | Encode / decode latency (median, IQR, mean, std) per codec, device, resolution and budget | Table 36; Figure 29 |
| `recompression/recompression_<dataset>_<budget>.csv` | Chain EER per (first codec, second codec) at 112 px, EdgeFace-XS, and the change against a single pass | Tables 66-69; Figure 42 |
| `recompression/recompression_colorferet_arcface_antelopev2.csv` | The Color FERET chains re-scored with ArcFace | Section 12 |
| `report_summary/key_findings.csv`, `.json` (same rows) | Register of the 18 headline findings: id, section, theme, statement, metric, value, unit, dataset, condition and the `source` file (relative to this folder) the value is read from | Main-findings box |
| `report_summary/derived_stats.json` | Scalars derived by `render_figures.py` (see above) | Sections 6, 11 |
| `report_summary/side_channel_leakage.json` | Identity leakage of the ACCURATE side code: EER / AUC of the raw and centred code and of a random code, with the trial counts | Section 7 (privacy) |
| `resolution_information/eer_by_resolution.csv`, `embedding_decomposition.csv`, `spectral_retention.csv` | EER of the aligned crops per resolution and matcher; cosine between a resized crop and its 224 px original; radial-PSD energy retained per resolution | Table 46; Figure 34 |

The id-cos columns of `comparison.json` (and the tables and key finding F02 built on
them) were measured with a proprietary face matcher that is not part of this release.
They cannot be re-measured with the public code; a run with a matcher registered
through `face1kb.fr.register_onnx(...)` gives different values. These files are their
only source.

## Paper items that need the datasets

The following paper items cannot be rebuilt from `results/` alone: they need the
aligned crops, per-image records or a GPU run (see the owning area's documentation):
the face montages (Figures 1, 3, 4, 8, 24-28, 32, 35-38), the KK dataset histograms
(Figures 5, 6), Table 4 (needs the NIST Color FERET labels), Figure 21 (per-file
size box plots) and Figure 39.

## What is not shipped

`collect_results.py` copies only the files listed in its `FILES` table. Not shipped:

- per-image and per-identity files: the per-sample difficulty table (image paths,
  per-image demographics), the per-identity and per-crop overlap scans (identity names,
  training-set row ids; replaced by `contamination/contamination_summary.csv`) and the
  JPEG-AI rate-control collapse evidence (bitstreams and a list of the affected crops);
- per-cell significance files for resolutions and budgets other than the cells above.
  They come from partial runs with incomplete codec sets (between 1 and 55 pairs per
  cell, so the per-cell BH correction covered different families) and are not
  consistent with the final codec runs: 1 - accuracy of a codec differs from its EER
  in `metrics.csv` by up to 155 % of that EER, against at most 1.2 % in the merged cells.
  No paper table or figure reads them;
- intermediate shards and superseded copies (significance shards, snapshots from before
  the face1kb codecs were added, backups, legacy classical-only recompression files,
  an older sanitization CSV without the `n` column, and `annex/exp1.csv`, the Experiment 1
  intermediate of the annex study that no table generator reads);
- LaTeX and PNG renderings (the generators recreate them) and the dataset figures.

## Changes made while collecting

Relative to the output files of the study run:

- rows of a codec that is not part of the published study were removed from
  `fairness_colorferet.csv` (88 rows), `fmr_fairness_colorferet.csv` (144 rows) and
  `codec_properties.csv` (1 row); no other row changed;
- `codec_properties.csv` lost the `encode_ms` / `decode_ms` columns, which held a
  "not measured" placeholder; latencies are in `quality/speed_*.csv`;
- a stray duplicate row with the tag `aligned_112.npy.tmp` (same values as
  `aligned_112`) was removed from `resolution_information/eer_by_resolution.csv` and
  `embedding_decomposition.csv`;
- the seven fairness files are under `fairness/`, where `experiments/fairness/`
  writes them (the study run kept them under `accuracy/`; `collect_results.py` also
  looks there, see `SOURCE_ALIASES`);
- the `source` fields of the findings register name the files of this layout
  (`accuracy/posthoc_scalars.txt`, `fairness/fairness_kk.csv`; the Color FERET
  significance cell as a slice of `accuracy/significance_colorferet.csv`);
- `contamination/contamination_summary.csv` was derived from the two overlap scans;
- `ablation/preprocessing_ablation.csv` and `difficulty/difficulty_contrast.csv` are
  not in the output folder of the study run. They were written by
  `experiments/preprocessing/ablation.py` and `experiments/difficulty/analyze.py` on
  the study data, and Tables 45, 49 and 50 rendered from them have the arXiv content.

All other files are byte-for-byte copies, and kept rows keep their exact text.

## Known deviations between the paper and `results/`

The code and these files are what produced the published numbers; where the paper
text or a hand-maintained table disagrees, the file is the reference.

- **Table 24 (best configuration)**, JPEG-AI at 1024 B: the table prints `0.73 @224`;
  the mean FNMR@1e-4 of ArcFace and LVFace-L in `metrics.csv` is 0.71 % (0.287 % and
  1.126 %) at 224 px. All other 19 cells match.
- **Table 63 (disparity CIs)**: the aligned, Ours-ACCURATE and Ours-FAST rows match
  `disparity_ci_kk.csv`; the WebP and JPEG 2000 rows do not. File values (ratio
  [95 % CI]) for ArcFace / LVFace-L / TopoFR-R100 / EdgeFace-XS: WebP 1.6 [1.1,2.9],
  1.1 [0.7,2.3], 1.1 [0.8,2.5], 1.1 [0.9,1.6]; JPEG 2000 10.7 [4.5,29.4],
  8.7 [3.9,31.3], 12.6 [4.9,33.9], 4.3 [2.1,9.5]. The file also has rows for AVIF,
  HEIF, JPEG, JPEG XL and JPEG-FzT. See [metrics.md](metrics.md) for how the impostor
  pairs are assigned to subgroups in this computation.
- **Table 80 (artifacts)** lists per-cell significance files, `contamination/*.csv`
  and a per-sample difficulty file; they are not shipped (see above). It also lists
  the fairness files under `accuracy/`; they are shipped under `fairness/`, the
  folder the public pipeline writes.
- **Key finding F16**: its statement says that "only two" modern-codec pairs are
  non-significant on Color FERET and that KK has a "single non-significant row"; its
  own value column, like `significance_*.csv`, lists four Color FERET pairs in the
  EdgeFace-XS cell (AVIF/WebP, HEIF/JPEG XL, HEIF/Ours-FAST, JPEG XL/Ours-FAST) and
  7 of 440 KK tests. The Section 14 text also reports four pairs and seven KK tests.
  The Figure 44 caption names only AVIF/WebP and HEIF/JPEG XL, while the figure itself
  marks all four pairs.
- **Figure 44** was drawn from a copy of the EdgeFace-XS cell; the merged
  `significance_colorferet.csv` holds the same rows, with the last digit of five
  p-values differing at the float-printing level. The figure uses only chi^2 and the
  significance decisions, which are identical, and renders byte-identically.
- `difficulty_correlations.csv`: the Color FERET |yaw| row comes from per-image head-pose
  estimates; the public NIST labels carry only nominal pose-code angles, so a public run
  gives a different value for it (see the difficulty documentation).
- `quality_summary.csv`: the 100-crop minimum of `quality_vs_identity.png` removes no
  shipped cell (the smallest cells, the CompressAI subsets, hold 300 crops).

## Rendering the paper tables from `results/`

Every table generator of the experiment areas that works from aggregates accepts
`--from-results` (some need an extra flag that selects the render step:
`experiments/accuracy/idcos_tail.py --render-only --from-results`,
`experiments/compress/generate_file_size_boxplots.py --from-summary --from-results`,
`experiments/compress/measure_jpegai_kk.py --phase table --from-results`;
`experiments/compress/generate_jpegai_speed_table.py` reads results/ by default (`--input-root`) and has no `--from-results` flag). Run on a CPU
with `FACE1KB_OUTPUT_ROOT` pointing at an empty folder and the flags of the block in
[reproduce.md](reproduce.md#two-ways-to-reproduce) (`--paper-header`, and
`--shading raw` for `fiq_grid`), they write 76 `.tex` tables, 69 of which have an
arXiv counterpart:

- 66 are byte-identical to the arXiv table;
- 3 differ in content: `best_config` and `disparity_ci_kk` (see the deviations above;
  both are hand-maintained in the paper) and `fit_rate_kk_1024` (the arXiv table has
  two all-"--" rows for the CompressAI baselines, which the generator drops).

Without `--paper-header` the tables differ from the arXiv files in the first `%`
comment line (the line that names the generator). Without `--shading raw`, `fiq_grid`
has the same numbers but a different tie handling of the best/worst shading.

`cf_attributes` (Table 4) needs the NIST Color FERET labels and is not rendered from
`results/`. Six of the seven tables without an arXiv counterpart (`attack_strength`,
`ours_defense`, `fairness_subgroup_{cf,kk}`, `fairness_disparity_kk`, `res_decomp`)
render paper tables that were typed inline in the paper source. The seventh,
`posthoc_stats_excl_edgeface`, is not a paper table: it holds the rank statistics
without the EdgeFace matchers that Section 14.5 quotes in the text.

Of the 24 PNG figures these generators and `render_figures.py` write, 23 are
byte-identical to the arXiv figures with the paper environment. The exception is
`resolution_summary.png` (Figure 34, `experiments/resolution/render.py`): the legends
are placed outside the panels, and the EER panel has no AI-Solutions-KK TopoFR-R200
line, which the arXiv figure draws from the rows of `eer_by_resolution.csv`
(`embedding_decomposition.csv` has only an empty 64 px row for that matcher).
