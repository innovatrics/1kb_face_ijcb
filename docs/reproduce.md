# Reproducing the paper

This page maps every section, table and figure of the extended study
([arXiv:2608.22866](https://arxiv.org/abs/2608.22866)) to the commands that produce it.
For each item it gives the inputs, the cost of a full run, and whether the item renders
from the shipped aggregates in [`results/`](../results/README.md) or needs the datasets.
Each area has its own page with the options, outputs, validation status and known
deviations. The pages are linked in the [area table](#areas-and-order-of-a-full-run).
All differences between the paper text and the code are collected in
[errata.md](errata.md).

- [Two ways to reproduce](#two-ways-to-reproduce)
- [Setup](#setup)
- [Areas and order of a full run](#areas-and-order-of-a-full-run)
- [Paper map](#paper-map)
- [Environment for exact numbers](#environment-for-exact-numbers)

## Two ways to reproduce

**1. Render from the shipped results (CPU, no data, minutes).** `results/` holds the
aggregate numbers of the paper run: CSV and JSON files per cell or subgroup, with no
images and no per-image values. Every table generator accepts `--from-results`; the
`run.sh` of accuracy, quality, codec_comparison, fairness, compress and speed also take
`FROM_RESULTS=1` (the generators of the other areas are called directly below).
Outputs go to `FACE1KB_OUTPUT_ROOT` (default `outputs/`):

```bash
pip install -e ".[eval]"        # pandas, SciPy and matplotlib for the renderers
FROM_RESULTS=1 PAPER_HEADER=1 SKIP="1 2 3" bash experiments/accuracy/run.sh
FROM_RESULTS=1 PAPER_HEADER=1 SKIP="1 2 4" FIQ_SHADING=raw bash experiments/quality/run.sh
FROM_RESULTS=1 PAPER_HEADER=1 SKIP="1" bash experiments/codec_comparison/run.sh
FROM_RESULTS=1 PAPER_HEADER=1 SKIP="1 2 3" bash experiments/fairness/run.sh
python experiments/compress/generate_file_size_boxplots.py --from-summary --from-results
python experiments/compress/generate_codec_properties_table.py --from-results
python experiments/compress/generate_jpegai_speed_table.py
python experiments/compress/measure_jpegai_kk.py --phase table --from-results
python experiments/speed/generate_speed_table.py --from-results
python experiments/recompression/make_tables.py --from-results --paper-header
python experiments/preprocessing/make_tables.py --from-results --paper-header
python experiments/resolution/render.py --from-results
python experiments/difficulty/make_tables.py --from-results --paper-header
python experiments/annex/generate_annex_tables.py --from-results
python experiments/adversarial/generate_sanitization_table.py --from-results
python experiments/figures/render_figures.py --from-results
```

This writes 76 `.tex` tables, 69 of which have an arXiv counterpart, and 24 figures.
With the paper environment ([below](#environment-for-exact-numbers)), 66 of the 69
tables are byte-identical, including `fiq_grid` (with `FIQ_SHADING=raw`, as above).
3 differ in content: `best_config`, `disparity_ci_kk` and `fit_rate_kk_1024`. 23 of
the 24 figures are byte-identical; the exception is `resolution_summary.png`. The three
content differences and the figure are explained in [errata.md](errata.md).
Six of the seven tables without an arXiv counterpart (`attack_strength`,
`ours_defense`, `fairness_subgroup_{cf,kk}`, `fairness_disparity_kk`, `res_decomp`)
render paper tables that are typed inline in the paper source. The seventh,
`posthoc_stats_excl_edgeface`, is not a paper table: it is the rank statistics without
the EdgeFace matchers, whose values Section 14.5 quotes in the text.
`--paper-header` (`PAPER_HEADER=1`) writes the published first comment line of each
table. The file-by-file description of `results/` is in [results.md](results.md).

**2. Full run from aligned crops (GPU, days).** The public pipeline starts from aligned
face crops ([datasets.md](datasets.md): AI-Solutions-KK crops on request from the
authors via a [repository issue](https://github.com/innovatrics/1kb_face_ijcb/issues),
Color FERET from NIST, aligned by you). The `run.sh` of each area runs its paper
section at full scale. Its header lists the stages, their cost and a `SKIP`
variable. Several hundred GPU-hours in total, most of them for JPEG-AI.

## Setup

```bash
git lfs pull                                 # released weights
scripts/setup_env.sh --paper                 # venv with the exact paper package versions (Python 3.11)
python scripts/fetch_models.py               # the 14 face-recognition evaluators (about 5.7 GiB), sha256-pinned
scripts/setup_jpegai.sh                      # JPEG-AI reference software (pinned clone + patch), only for JPEG-AI cells
export FACE1KB_DATA_ROOT=/path/to/data       # aligned crops: <root>/<dataset>/aligned_<res>/<subject>/<stem>.png
export FACE1KB_WORK_ROOT=/path/to/work       # compressed files, decoded caches, embeddings (large)
export FACE1KB_OUTPUT_ROOT=/path/to/outputs  # aggregates, tables, figures
```

Every path goes through `face1kb.config`. The variables are listed in the
[`face1kb/config.py`](../face1kb/config.py) docstring. The dataset names are
`colorferet` and `kk`. The codecs, baselines and evaluators are described in
[codec.md](codec.md), [baselines.md](baselines.md) and [models.md](models.md).

## Areas and order of a full run

Run the areas top to bottom: each row reads the outputs of the rows above it. Costs are
for one NVIDIA RTX 2080 Ti and a single CPU core unless stated otherwise, both
datasets, full scale.

| # | Area (`experiments/…`) | Reads | Writes | Cost of a full run | Doc |
|---|---|---|---|---|---|
| 1 | `prepare/` | aligned crops, NIST ground truth (Color FERET) | `index.csv`, `pairs.parquet`, `labels.csv`, `attributes.csv`, crop variants, dataset figures | CPU, minutes; learned preprocessing operators ~25 GPU-min | [datasets.md](datasets.md) |
| 2 | `compress/` | crops, index | compressed grid of all codecs, decoded caches, file-size summary | classical codecs + JPEG-FzT ~200 CPU core-h; JPEG-AI ~230 GPU-h; face1kb ~45 GPU-h; CompressAI ~4 GPU-h; decoded caches ~35 GPU-h | [reproduce_compress.md](reproduce_compress.md) |
| 3 | `speed/` | Color FERET crops | `quality/speed_*.csv` | ~1 h CPU core + ~20 GPU-min (quiet host) | [reproduce_compress.md](reproduce_compress.md#speed) |
| 4 | `embed/` | crops, decoded cells | `embeddings/<model>/<tag>.npy` | ~30 GPU-h core grid, plus the budget sweep and variants | [reproduce_accuracy.md](reproduce_accuracy.md#1-embeddings) |
| 5 | `accuracy/` | embeddings, index, pairs | `metrics.csv`, significance CSVs, identity tail, tables and figures | ~8 GPU-h (CUDA scorer); significance ~5-13 CPU-min per cell | [reproduce_accuracy.md](reproduce_accuracy.md) |
| 6 | `quality/` | crops, compressed cells | quality CSVs, FIQ, montages | ~10 GPU-h (JPEG-AI decoded live adds ~40 h); FIQ ~1 GPU-h | [reproduce_quality.md](reproduce_quality.md) |
| 7 | `codec_comparison/` | crops (codecs run on the fly) | `res{112,224}/comparison.json` | ~3 h at 112 px, ~4 h at 224 px, 1 GPU + CPU | [reproduce_quality.md](reproduce_quality.md#codec-comparison) |
| 8 | `fairness/` | anchor embeddings, labels / attributes | fairness CSVs, tables | ~4-5 GPU-h | [reproduce_quality.md](reproduce_quality.md#fairness) |
| 9 | `recompression/` | crops | chain EER CSVs, tables | ~20 h CPU (8 workers) for Color FERET, ~4-5 h for KK; scoring ~1-2 GPU-h per dataset and budget | [reproduce_studies.md](reproduce_studies.md#recompression) |
| 10 | `preprocessing/` | crops, crop variants, embeddings | through-codec and ablation CSVs, tables, montages | ~1 GPU-h + ~1 h CPU | [reproduce_studies.md](reproduce_studies.md#preprocessing-through-the-codec) |
| 11 | `resolution/` | embeddings | resolution CSVs, Table 46, Figure 34 | CPU, 30-60 min | [reproduce_studies.md](reproduce_studies.md#resolution-information-analysis) |
| 12 | `difficulty/` | Color FERET / KK crops (300 each) | difficulty aggregates, tables, figure | ~3-4 GPU-h (+1-2 h with JPEG-AI) | [reproduce_studies.md](reproduce_studies.md#sample-difficulty) |
| 13 | `adversarial/` | crops, Li-AE proxy weights | adversarial crops, compressed cells, `sanitization.csv` | ~20 GPU-h compress + ~3 GPU-h embed | [reproduce_adversarial.md](reproduce_adversarial.md), [adversarial.md](adversarial.md) |
| 14 | `annex/` | crops (56-224 px) | annex scores, CIs, tables | ~4 h CPU grid + ~8 GPU-h JPEG-AI + ~2 h | [reproduce_annex.md](reproduce_annex.md) |
| 15 | `figures/` | the aggregates above | seven summary figures, `derived_stats.json`, optional copy into `results/` | CPU, ~15 s | [results.md](results.md) |

The metric definitions (pair sampling, EER grid, FNMR at FMR, bootstrap, McNemar,
DeLong, BH-FDR) are in [metrics.md](metrics.md). Codec training
([training.md](training.md)) is not part of reproducing the tables: the tables use the
released weights.

## Paper map

Table and figure numbers are those of the arXiv PDF. Section numbers are the printed
ones. **Renders** means the item can be rebuilt from `results/`:

- **results**: the item renders on a CPU from the shipped aggregates.
- **data**: the item needs aligned crops, per-image data or a GPU run.
- **text**: the item is typed in the paper source, or is TikZ; nothing to run.
- **shipped**: the numbers exist only as a shipped file; no public generator.

"Cost" is the cost of recomputing the item's inputs from crops.

### Front matter, Sections 1-4 (introduction, related work, datasets, protocol)

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Main-findings box | typed from `results/report_summary/key_findings.csv` | findings register | -- | shipped |
| Table 1 `tab:deployment-guidance`, Table 3 `tab:dataset-summary`, Tables 5-6 `tab:environment`, `tab:codec-settings` | -- | -- | -- | text |
| Figure 1 (codec strip) | `experiments/quality/visual_quality.py` | crops, compressed cells | 1 GPU, ~1 min | data (faces, keep local) |
| Figures 2, 7 `fig:workflow`, `fig:protocol-pipeline` | -- | -- | -- | text (TikZ) |
| Table 2 `tab:codec-properties` | `experiments/compress/generate_codec_properties_table.py --from-results` | `codec_properties.csv` | -- | results (the CSV itself is shipped only) |
| Figure 3 `fig:ds-cf` | `experiments/prepare/dataset_report.py` | Color FERET crops | CPU, seconds | data; the published source-image montage cannot be regenerated, the script builds a montage of your crops |
| Table 4 `tab:cf-attributes` | `experiments/prepare/cf_attribute_table.py` | NIST Color FERET ground truth (`labels.csv`) | CPU, seconds | data |
| Figures 4-6 `fig:ds-kk-examples`, `fig:ds-kk-demo`, `fig:ds-kk-orig` | `experiments/prepare/dataset_report.py` | KK photos, crops, `attributes.csv` | CPU, seconds | data |

### Section 5: Benchmark of existing codecs

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Figure 8 `fig:cmp-1kb` | `experiments/quality/visual_quality.py` | crops, compressed cells | 1 GPU, ~1 min | data (faces) |
| Tables 7, 8 `tab:codec-cmp-224`, `tab:codec-cmp-112` | `experiments/codec_comparison/tables.py --from-results` | `codec_comparison/res*/comparison.json`, `quality/quality_summary.csv` | ~7 h (`evaluate.py`) | results; the id-cos columns come from a proprietary matcher and cannot be regenerated |
| Tables 9-16 `tab:rate-eer-*` | `experiments/accuracy/rate_eer.py --from-results` | `accuracy/metrics.csv`, `codec_comparison/file_size_summary.csv` | compress + embed + accuracy | results |
| Tables 17-23 `tab:frr-summary`, `tab:frr-{cf,kk}-*`; Figures 9-14 `fig:frr-*`, `fig:det-*` | `experiments/accuracy/frr_far.py --from-results` | `metrics.csv` | same | results |
| Table 24 `tab:best-config` | `experiments/accuracy/best_config.py --from-results` | `metrics.csv` | same | results (one cell differs, see errata) |
| Table 25 `tab:h4-budget`, Figure 15 `fig:h4-curve` | `experiments/accuracy/h4_curve.py --from-results` | `metrics.csv` | same | results |
| Tables 26-29 `tab:model-effect*`, Figures 16-19 | `experiments/accuracy/frr_far.py --from-results` | `metrics.csv` | same | results |
| Tables 30-32 `tab:fit-kk`, `tab:fit-cf`, `tab:fit-1024`; Figure 20 `fig:size-compliance` | `experiments/compress/generate_file_size_boxplots.py --from-summary --from-results` | `file_size_summary.csv` | compress | results |
| Figure 21 `fig:size-box` | `experiments/compress/generate_file_size_boxplots.py` | every compressed file (size scan) | compress, ~20 min scan | data |

`FROM_RESULTS=1 SKIP="1 2 3" bash experiments/accuracy/run.sh` runs all accuracy
renderers with their exact arguments ([reproduce_accuracy.md](reproduce_accuracy.md)).

### Section 6: Reconstruction quality

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Table 33 `tab:quality-matrix`, Table 35 `tab:budget-sweep` | `experiments/quality/quality_tables.py --from-results` | `quality/quality_summary.csv` | ~10 GPU-h (`measure_quality.py`) | results |
| Figure 22 `fig:quality-vs-identity` | `experiments/figures/render_figures.py --from-results --only quality_vs_identity` | `metrics.csv`, `quality_summary.csv` | -- | results |
| Table 34 `tab:fiq`, Figure 23 `fig:fiq` | `experiments/quality/fiq_report.py --from-results --shading raw` | `quality/fiq_{colorferet,kk}.csv` | ~1 GPU-h (`fiq.py`) | results |
| Figures 24-28 (codec grids, Ours grids, budget and resolution sweeps) | `experiments/quality/visual_quality.py` | crops, compressed cells | 1 GPU, minutes | data (faces) |
| Table 36 `tab:speed`, Figure 29 `fig:speed-trend` | `experiments/speed/generate_speed_table.py --from-results` | `quality/speed_*.csv` | ~1.5 h (`experiments/speed/run.sh`) | results; timings depend on the host |
| Table 37 `tab:jpegai-speed` | `experiments/compress/generate_jpegai_speed_table.py` | `codec_comparison/jpegai_decoders.csv` | -- | results (the CSV itself is shipped only) |
| Figure 30 `fig:jpegai-tradeoff` | `experiments/codec_comparison/jpegai_tradeoff.py --from-results` | `jpegai_decoders.csv` | -- | results |
| Table 38 `tab:jpegai-kk` | `experiments/compress/measure_jpegai_kk.py --phase table --from-results` | `codec_comparison/jpegai_kk.csv` | ~1.5 GPU-h | results |

### Section 7: A custom identity-preserving codec (face1kb)

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Figure 31 `fig:codec-arch`, Table 39 `tab:codec-container` | -- ([codec.md](codec.md) documents both from the code) | -- | -- | text |
| Parameter counts, container, budget search | `pytest tests/codec` | released weights | CPU / GPU, minutes | -- |
| Training | `python -m face1kb.codec.train` ([training.md](training.md)) | WebFace42M (not distributed) | FAST ~2.3 days, ACCURATE ~15 days on one GPU | recipe only, not the released weights |
| Tables 40, 41 `tab:codec-results`, `tab:codec-results-112` | `experiments/codec_comparison/tables.py --from-results` | `comparison.json`, `quality_summary.csv` | ~7 h | results; the id-cos columns come from a proprietary matcher |
| Budget fit rates of the Ours rows | Tables 30-32 above | `file_size_summary.csv` | -- | results |
| Reconstruction check (byte fill, cast, black frames) | -- | `results/quality/codec_verify.csv` | -- | shipped |
| Side-stream ablation (`subsec:codec-side`) | -- | `results/ablation/s7_ablation.csv` | -- | shipped; the ablation checkpoints are not released |
| Train/test overlap check | -- | `results/contamination/contamination_summary.csv` | -- | shipped (dataset-level summary only) |
| Side-code linkability (privacy paragraph) | -- | `results/report_summary/side_channel_leakage.json` | -- | shipped |
| Table 42 `tab:heldout-cvlface` | `experiments/accuracy/heldout_table.py --from-results` | `metrics.csv` | embed + accuracy | results |
| Table 43 `tab:idcos-tail` | `experiments/accuracy/idcos_tail.py --render-only --from-results` | `quality/idcos_tail_512.csv` | CPU < 1 min from ArcFace embeddings | results |

### Section 8: Trivial and difficult samples

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Tables 44, 45 `tab:difficulty-predictors`, `tab:difficulty-contrast`; Figure 33 | `experiments/difficulty/make_tables.py --from-results --paper-header` | `difficulty/difficulty_{correlations,contrast}.csv` | ~3-4 GPU-h (`compute.py`, 2 x 300 crops) | results; a public rerun differs (proprietary id-cos, Color FERET pose estimates) |
| Figure 32 `fig:difficulty-montage` | `experiments/difficulty/compute.py` (montage stage of `run.sh`) | Color FERET crops | 1 GPU, ~1 min | data (faces) |

### Section 9: Ablations and preprocessing

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Table 46 `tab:res-decomp`, Figure 34 `fig:res-summary` | `experiments/resolution/render.py --from-results` | `resolution_information/*.csv` | CPU 30-60 min (`analyze.py`) | results (the figure is not byte-identical, see errata) |
| Tables 47, 48 `tab:preproc-through`, `tab:preproc-through-edge` | `experiments/preprocessing/make_tables.py --from-results --paper-header` | `accuracy/preproc_through_codec_kk.csv` | ~50 GPU-min + ~40 CPU-min (`through_codec.py`) | results |
| Figure 35 `fig:preproc-through` | `experiments/preprocessing/through_codec.py` (montage) | KK crops | 1 GPU, ~1 min | data (faces) |
| Tables 49, 50 `tab:ablation-preproc`, `tab:ablation-crop` | `experiments/preprocessing/make_tables.py --from-results --paper-header` | `ablation/preprocessing_ablation.csv` | CPU ~10 min (`ablation.py`) after embedding the variants | results |
| Figure 36 `fig:preproc-impact` | `experiments/preprocessing/impact_figure.py` | Color FERET crops | ~1 min | data (faces) |
| Figures 37, 38 `fig:align-check`, `fig:align-variants` | `experiments/preprocessing/montages.py` | Color FERET crops | CPU, ~10 s | data (faces) |
| Figure 39 `fig:fzt-ablation` | -- | -- | -- | not reproduced (no generator; the triangle-kernel variant is not implemented) |

### Section 10: ISO/IEC 29794-5 Annex E/F parameter tables

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Tables 51-58 `tab:annex-*` | `experiments/annex/generate_annex_tables.py --from-results` | `annex/{scores,scores_pop2000,ci}.csv` | ~4 h CPU grid + ~8 GPU-h JPEG-AI + ~2 h (`experiments/annex/run.sh`) | results (byte-identical) |

### Section 11: Demographic fairness

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Tables 59-61, 64 (subgroup EERs, KK disparity, 512 B) | `experiments/fairness/tables.py --from-results` | `fairness/fairness_*.csv`, `fairness_disparity_*.csv` | ~3-4 GPU-h (`subgroup.py`) | results (59-61 are inline in the paper) |
| Table 62 `tab:fairness-disparity-cf` | same | `fairness_disparity_colorferet.csv` | same | results |
| Table 63 `tab:disparity-ci` | same | `disparity_ci_kk.csv` | ~30 GPU-min (`disparity_ci.py`) | results (two rows differ from the paper, see errata) |
| Table 65 `tab:fmr-fairness-cf` | same | `fmr_fairness_colorferet.csv` | ~20 GPU-min (`fmr_fairness.py`) | results |
| Figures 40, 41 `fig:fairness-disparity*` | `experiments/figures/render_figures.py --from-results` | fairness CSVs | -- | results |

The KK subgroups need `attributes.csv`; the Color FERET subgroups need the NIST labels.

### Section 12: Recompression

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Tables 66-69 `tab:recompression-eer-*` | `experiments/recompression/make_tables.py --from-results --paper-header` | `recompression/recompression_*.csv` | ~25 h CPU + ~4-8 GPU-h (`recompress.py`) | results |
| Figure 42 `fig:recompression-heatmap` | `experiments/figures/render_figures.py --from-results --only recompression_heatmap` | same | -- | results |

### Section 13: Adversarial robustness and sanitization

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Tables 70-75 (`tab:attack-strength`, `tab:sanitization-{hfc,clip,liae,hfc-kk}`, `tab:ours-defense`) | `experiments/adversarial/generate_sanitization_table.py --from-results` | `adversarial/sanitization.csv` | ~23 GPU-h (`experiments/adversarial/run.sh`) | results; Tables 70 and 74 are inline in the paper |
| Figure 43 `fig:sanitization` | `experiments/figures/render_figures.py --from-results --only sanitization_residual` | `sanitization.csv` | -- | results |

### Section 14: Statistical significance

| Item | Command | Inputs | Cost | Renders |
|---|---|---|---|---|
| Tables 76, 77 `tab:sig-cf-edgeface`, `tab:sig-kk` | `experiments/accuracy/significance_tables.py --from-results` | `accuracy/significance_{colorferet,kk}.csv` | CPU, ~5-13 min per cell (`compute_significance.py`, 12 cells) | results |
| Table 78 `tab:posthoc` | `experiments/accuracy/posthoc_stats.py --from-results` | `metrics.csv` | CPU, seconds | results |
| Figures 44, 45 `fig:significance-matrix`, `fig:codec-mean-rank` | `experiments/figures/render_figures.py --from-results` | significance CSV, `posthoc_scalars.txt` | -- | results |
| CVLface IR-101 check in Section 14.2 | `experiments/accuracy/compute_significance.py --datasets colorferet --models cvlface_ir101 --res 112 --budgets 1024 --out-suffix _cvlface_ir101` | embeddings | CPU, ~8 min | data (not in the shipped CSVs, see errata) |

### Sections 15-17 (discussion, conclusion, artifacts)

Table 79 `tab:status` and Table 80 `tab:artifacts` are typed in the paper source.
Table 80 lists some files that are not shipped because they hold per-image or
per-identity data ([results.md](results.md#what-is-not-shipped)).

## Environment for exact numbers

Byte-identical outputs need the paper environment (`requirements/paper.txt`, installed
with `scripts/setup_env.sh --paper`) and, for GPU stages, an RTX 2080 Ti class
(Turing, sm_75) GPU with torch 2.10.0 / CUDA 12.8. Other GPUs and library versions give
comparable numbers that can differ in the last digits. Rendering from `results/` is
exact on any machine with the pinned matplotlib (3.11.0) and fonts. The reproducibility
limits of each stage are listed in [errata.md](errata.md#reproducibility-limits).
