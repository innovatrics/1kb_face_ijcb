# Reproducing the verification-accuracy results

This page covers the embedding stage and everything computed from the embeddings in
the accuracy part of the paper:

- the `metrics.csv` grid;
- the rate-EER, operating-point, best-configuration, budget-floor, held-out-matcher
  and model-effect tables, with their figures (Sections 5.2-5.6 and 7.6);
- the identity-cosine tail (Section 7.8);
- the significance tests and rank statistics (Section 14).

The metric definitions (pair sampling, EER grid, FNMR@FMR, bootstrap, McNemar,
DeLong, BH-FDR) are in [metrics.md](metrics.md). The face-recognition models are
described in [models.md](models.md).

| Step | Script | Input | Output | Cost |
|---|---|---|---|---|
| embeddings | `experiments/embed/compute_embeddings.py` | aligned crops, compressed cells | `WORK_ROOT/<ds>/embeddings/<model>/<tag>.npy`, `manifest.csv` | GPU, ~35-45 s per array |
| accuracy grid | `experiments/accuracy/compute_accuracy.py` | embeddings, `index.csv`, `pairs.parquet` | `OUTPUT_ROOT/accuracy/metrics.csv` | 1 GPU; ~4 s (Color FERET) / ~9 s (AI-Solutions-KK) per array with CIs, ~8 h for the ~4,600 arrays |
| significance | `experiments/accuracy/compute_significance.py` | same | `OUTPUT_ROOT/accuracy/significance_<ds>.csv` | CPU; ~5-8 min per Color FERET cell, ~12-13 min per AI-Solutions-KK cell |
| identity tail | `experiments/accuracy/idcos_tail.py` | ArcFace embeddings | `OUTPUT_ROOT/quality/idcos_tail_512.csv` | CPU, < 1 min |
| tables, figures | `experiments/accuracy/{rate_eer,frr_far,best_config,h4_curve,heldout_table,idcos_tail,posthoc_stats,significance_tables}.py` | the CSVs above | `OUTPUT_ROOT/tables/*.tex`, `OUTPUT_ROOT/figures/*.png`, `OUTPUT_ROOT/accuracy/posthoc_scalars.txt` | CPU, ~1 min |

`experiments/embed/run.sh` and `experiments/accuracy/run.sh` run these steps at full
scale. Each script prints its options with `--help`. All paths come from
`face1kb.config` (`FACE1KB_DATA_ROOT`, `FACE1KB_WORK_ROOT`, `FACE1KB_OUTPUT_ROOT`,
`FACE1KB_MODELS_ROOT`); the scripts take `--out-dir` (and, for the generators,
`--input-root` / `--from-results`) to override them.

## Rendering the paper tables without any data

The aggregate results of the paper are shipped in `results/`. No crops, embeddings or
GPU are needed to render every table and figure of this page from them:

```bash
FROM_RESULTS=1 PAPER_HEADER=1 SKIP="1 2 3" bash experiments/accuracy/run.sh
```

This writes the following to `outputs/tables/` and `outputs/figures/`:

- 27 tables that are byte-identical to the arXiv sources:
  - `rate_eer_{colorferet,kk}_{arcface_antelopev2,lvface_l,topofr_r100,edgeface_xs}`
  - `frr_far_{colorferet,kk}_{1024,512}[_224]`, `frr_far_summary`
  - `model_effect_{colorferet,kk}_1024[_224]`
  - `h4_budget`, `heldout_cvlface`, `idcos_tail`, `posthoc_stats`
  - `sig_cf_edgeface`, `sig_kk`
- `best_config.tex`, which matches the paper except for one cell (see
  [Differences from the paper](#differences-from-the-paper));
- 11 figures that are byte-identical to the paper PNGs with matplotlib 3.11.0 (the
  paper environment, `requirements/paper.txt`):
  - `frr_far_{colorferet,kk}_1024.png`
  - `det_{colorferet,kk}_{1024,512}.png`
  - `model_effect_{colorferet,kk}_1024[_224].png`
  - `h4_budget_curve.png`

`--paper-header` (`PAPER_HEADER=1`) writes the first comment line of each published
table, which names the script that originally produced it; without it, the comment
names the script in `experiments/accuracy/`.

## 1. Embeddings

```bash
python scripts/fetch_models.py --only all          # 14 evaluators, 5.7 GiB, once
python experiments/compress/decode_cache.py --datasets colorferet,kk \
    --budgets 1024,512,768,960 --gpus 0          # = step 5 of experiments/compress/run.sh
python experiments/embed/compute_embeddings.py --models all --gpus 0,1
```

JPEG-AI cells are embedded only from their decoded PNG cache (or with the slow
`--jpegai-live`); a JPEG-AI cell without a cache is skipped with a warning. The
768 / 960 B budgets above are the Color FERET budget sweep of `tab:h4-budget`. Crop
variants (step 3 of `experiments/embed/run.sh`) need their own cache first:
`decode_cache.py --align-suffix <suffix>` for each variant suffix whose JPEG-AI cells
you want embedded (for example `_adv_hfc_006`).

**Sources.** The script scans the crop folders that exist on disk, so run it after the
compression stage:

- every aligned crop set: `config.aligned_dir(ds, res, suffix)`;
- every compressed cell: `config.compressed_dir(ds, res, budget, codec, suffix)`.

The selection is restricted with `--datasets`, `--kind`, `--resolutions`, `--codecs`,
`--budgets` and `--suffixes`. By default the core grid is scanned: 5 resolutions,
1024 / 512 B, the 12 codecs, and base crops only.

**Output.** One array per (dataset, model, source): `(N, 512)` float32 raw
embeddings. Row `i` is image id `i` of `index.csv`, and a NaN row marks a crop that is
missing or could not be decoded. The tag is `aligned_<res><suffix>` or
`<codec>_<res><suffix>_<budget>`. At the end of every run,
`embeddings/manifest.csv` is rebuilt from the arrays on disk, with the columns
`dataset, model, source_tag, kind, res, codec, budget, n_images, n_ok, n_missing,
dim`.

**Decoding.**

- Bitstreams are decoded with `face1kb.baselines.decode_file`:
  - JPEG-FzT at the source resolution;
  - CompressAI and face1kb containers on CUDA, the latter from the released weights.
- A decoded PNG cache (`config.decoded_dir`, written by
  `experiments/compress/decode_cache.py`) replaces decoding for JPEG-AI and the two
  face1kb codecs. For the face1kb codecs it only saves time: a live decode on the
  same GPU model gives the same pixels.
- JPEG-AI is embedded only from its cache:
  - If the cache folder of a cell exists, a bitstream without a cached PNG counts as a
    missing crop. This is how the reference decoder's failures appear in the paper
    grid, for example one Color FERET 224 px / 512 B crop.
  - A cell without a cache folder is skipped with a warning. `--jpegai-live` decodes it
    with the reference software instead, at about 0.5 s per crop and per model.

**Embedding.**

- Crops that are not 112 px are resized bilinearly to 112 px
  (`face1kb.fr.embedders.resize_batch`, OpenCV `INTER_LINEAR`). Every matcher consumes
  a 112 px input.
- Crops are embedded with `face1kb.fr.load(model).embed`. Image ids are processed in
  chunks of `--batch` (default 256), and the decodable crops of one chunk form one
  forward batch.

**Execution.**

- Each model runs in its own spawned worker process, pinned to one GPU of `--gpus`.
  This also keeps the two CVLface snapshots in separate processes.
- Finished arrays are skipped, and arrays are written atomically, so an interrupted
  run resumes where it stopped.
- `--limit N` embeds only the first N ids of every source; the other rows stay NaN.
- `--dry-run -v` lists the plan.
- Your own matcher (`FACE1KB_FR_PLUGINS`, see [models.md](models.md)) is accepted by
  `--models`.

**Cost.** About 35-45 s per array on an RTX 2080 Ti with cached decodes. The core grid
has about 250 arrays per matcher on the two datasets, which is about 30 GPU-hours for
the 14 matchers. Without a cache, a face1kb cell adds one GPU decode per crop and per
matcher: 11-18k decodes, about 5-10 min.

The other studies embed their crop variants with the same script (for example
`--suffixes _A1,...,_C2,_tight,_mid,_fill,_adv_hfc_003,...` for the four anchors at
112 px; step 3 of `experiments/embed/run.sh`).

## 2. The `metrics.csv` grid

```bash
python experiments/accuracy/compute_accuracy.py --models all --device cuda --no-cpu-fallback
# or one process per dataset, then merge:
python experiments/accuracy/compute_accuracy.py --datasets colorferet --device cuda:0 --no-combined
python experiments/accuracy/compute_accuracy.py --datasets kk --device cuda:1 --no-combined
python experiments/accuracy/compute_accuracy.py --merge
```

**What is scored.** Every embedding array of the selected models is scored with
`face1kb.eval.verification.accuracy_rows`:

- all mated pairs and a seed-0 sample of 5,000,000 non-mated pairs;
- EER, and FNMR at FMR = 1e-2 / 1e-3 / 1e-4 with the realised FMR;
- 95 % subject-cluster bootstrap intervals (200 replicates, impostors capped at 1M) for
  the clean cells of ArcFace, LVFace-L, TopoFR-R100, EdgeFace-XS and CVLface IR-101;
- `eer_aligned` / `delta_eer` against the clean aligned source of the same resolution.

`--tags a,b,...` scores only the listed sources. The clean `aligned_<res>` source of
every listed resolution is always scored and written with them, so `eer_aligned` and
`delta_eer` keep their values; rows whose baseline array is missing are logged.

**Output files.** `metrics_<dataset>.csv` holds each dataset's rows, and `metrics.csv`
holds all of them. The combined file is written with
`face1kb.eval.verification.write_metrics_csv`, which applies the single CSV read/write
of the published merge step, so its text is identical to the published file.
`--merge` rebuilds `metrics.csv` from the per-dataset files; it uses their values as
written and does not recompute any column. A run that covers fewer models than an
existing `metrics.csv` refuses to replace it (`--allow-shrink` overrides).

**Scorer.** The published grid was scored with the CUDA scorer. On the CPU (NumPy
scorer), a few cells move by one trial. `--no-cpu-fallback` makes a GPU that runs out
of memory raise an error, instead of silently scoring that array on the CPU.

**`median_bytes`** is NaN in every row, as in the published file. File sizes are in
`codec_comparison/file_size_summary.csv`.

## 3. Significance tests

```bash
python experiments/accuracy/compute_significance.py --datasets colorferet --models anchor --res 112 --budgets 1024
python experiments/accuracy/compute_significance.py --datasets kk --models anchor --res 112,224 --budgets 1024
```

**Tests.** For each (dataset, model, res, budget) cell,
`face1kb.eval.significance.significance_rows` scores all sources of the cell on one
shared trial set: all mated pairs and a seed-0 sample of 2,000,000 non-mated pairs,
restricted to the trials valid for every source. Every pair of sources gets a McNemar
test, at each source's own EER threshold, and DeLong's paired AUC test. The McNemar
p-values are BH-FDR adjusted within the cell: 55 tests for 11 sources, as in the
published tables. Sources with less than 50 % valid trials are dropped, which is why
the CompressAI 300-crop subsets are not in any cell.

**Output.** One frame per cell, in order of model name, resolution and budget, written
with `write_significance_csv`. Its text matches the published CSV.

**Scorer.** The published files used the NumPy scorer (the default). `--device cuda`
changes the AUC columns in the last digits.

**Sharded runs.** Run one process per model with `--out-suffix _<model>`, then
concatenate the shards with `--merge OUT IN...`.

**Rank statistics.** `posthoc_stats.py` computes the Friedman test, Kendall's W, the
mean ranks and Holm-Wilcoxon with Cliff's delta over the matcher x codec EER matrix
of `metrics.csv` (Color FERET, 112 px, 1024 B; 14 matchers x 12 codecs). With
`--exclude-prefix edgeface` it gives the figure without the EdgeFace family that
Section 14.5 quotes: n = 10, W = 0.893.

## 4. Tables and figures

| Script | Paper items |
|---|---|
| `rate_eer.py` | tab:rate-eer-{cf,kk}-{arcface,edgeface,lvface,topofr}. Over-budget daggers come from `codec_comparison/file_size_summary.csv` (`pct_holding` < 99.5 %) |
| `frr_far.py` | tab:frr-summary, tab:frr-{cf,kk}-{1024,512,224}, tab:model-effect{,-kk,-224,-224-kk}, fig:frr-{cf,kk}, fig:det-{cf,kk}{,-512}, fig:model-effect* |
| `best_config.py` | tab:best-config |
| `h4_curve.py` | tab:h4-budget, fig:h4-curve |
| `heldout_table.py` | tab:heldout-cvlface |
| `idcos_tail.py` | tab:idcos-tail (and `quality/idcos_tail_512.csv`) |
| `posthoc_stats.py` | tab:posthoc (and `accuracy/posthoc_scalars.txt`) |
| `significance_tables.py` | tab:sig-cf-edgeface, tab:sig-kk |

**Shading.** Best cells are shaded green and bold, worst cells red and underlined,
through `face1kb.report.latex`. The frr_far and model_effect tables carry colour only,
as published.

**Support rules.** In the operating-point, model-effect and H4 tables, a codec (or
cell) scored on fewer than half of the median number of mated pairs is marked
`$^{\ddagger}$` and is not ranked. This applies to the Color FERET CompressAI subsets.

The significance-matrix and codec-mean-rank figures of Section 14 are drawn by
`experiments/figures/render_figures.py` from the same CSVs. The findings register of
Sections 4 and 17 (`results/report_summary/key_findings.{csv,json}`) is shipped as a
static file; it has no generator in this repository.

## Reproduction status

These checks were run in the paper container (RTX 2080 Ti, torch 2.10.0, CUDA 12.8,
onnxruntime 1.23.2, numpy 2.4.6, pandas and matplotlib as in
`requirements/paper.txt`):

- **Tables and figures.** Rendering from `results/`: 27 of 27 tables and 11 of 11
  figures are byte-identical to the paper, and `best_config` differs in one cell.
  `posthoc_scalars.txt` is identical to the shipped file.
- **`metrics.csv`.** 112 distinct rows re-scored from the original embedding arrays
  (CUDA scorer, `--tags` runs) are text-identical to the published file in all 28
  columns. They cover both datasets, 11 matchers, clean, crop-variant and adversarial
  sources (including the NaN FNMR@1e-4 path), 768 / 960 B cells, CompressAI subsets,
  empty cells and bootstrap CIs. `--merge` of two per-dataset runs gives the same
  text. The only extra row is the AI-Solutions-KK CVLface IR-101 Ours-ACCURATE 168 px
  cell described under [Differences from the paper](#differences-from-the-paper).
- **Significance.** Color FERET, four anchors, 112 px / 1024 B in one run
  (`--models anchor`): the written `significance_colorferet.csv` is byte-identical to
  the published file (220 rows). AI-Solutions-KK ArcFace and EdgeFace-XS, 112 + 224 px
  / 1024 B, one run per model: 110 of 110 rows each text-identical, in the published
  order. `--merge` of per-model shards keeps their text byte for byte.
- **Identity tail.** `idcos_tail_512.csv` recomputed from the ArcFace arrays is
  byte-identical.
- **Embeddings.** Re-embedding with `compute_embeddings.py`, compared with the
  original arrays:
  - First 256 ids of 80 (model, source) arrays for the four anchors:
    - Color FERET 64 / 112 px / 1024 B: aligned, JPEG, JPEG-FzT, JPEG-AI, bmshj2018,
      Ours-FAST and Ours-ACCURATE, both face1kb variants decoded live from the
      released weights;
    - AI-Solutions-KK 224 px / 512 B: aligned, WebP, HEIF, JPEG-AI, Ours-FAST and
      Ours-ACCURATE, from the decoded caches.
  - First 256 ids of AI-Solutions-KK 112 px / 1024 B, all 11 sources, for LVFace-L
    and CVLface ViT-B (Ours-FAST and Ours-ACCURATE decoded live): LVFace-L is
    bit-exact on 10 of 11 sources (JPEG-AI: float noise, 2.4e-6 relative); CVLface
    ViT-B is bit-exact on aligned, JPEG-AI and Ours-FAST, float noise (< 8e-7
    relative) on the classical codecs, and Ours-ACCURATE is covered by the
    limitation below.
  - Full 11,335-row Color FERET arrays (aligned, WebP) for ArcFace and EdgeFace-XS.
  - Results:
    - The aligned crops and the classical codecs at the stored arrays' batch size are
      bit-exact, as are all Ours-FAST arrays, the cached Ours-ACCURATE arrays and the
      full arrays.
    - Some stored arrays were computed with 128- or 64-id chunks (the chunk size
      varies by matcher and cell, mostly among the non-112 px Color FERET cells) or
      on an older software stack. Re-embedding them at batch 256 differs by float noise
      only: at most 1.8e-5 absolute and 6e-6 relative per row. With `--batch 64`,
      the Color FERET 96 px compressed cells (TopoFR-R100, JPEG-XL and AVIF at
      512 B, first 256 ids) are bit-exact.

## Reproducibility limits

- **Hardware and software.** Bit-exact embeddings need the same GPU model and software
  stack (see [models.md](models.md)). Other GPUs and batch sizes give float noise.
- **Ours-ACCURATE arrays of some matchers.** For a few matchers, the stored arrays of
  the Ours-ACCURATE cells differ from a re-embedding in 6-20 % of the rows, by up to
  1.1e-3 relative (row cosine > 0.9999997), at every batch size (256, 128, 64):
  - ArcFace, Color FERET 64 px / 1024 B: 15 of 256 rows;
  - ArcFace, Color FERET 112 px / 1024 B: 48 of 256 rows;
  - CVLface ViT-B, AI-Solutions-KK 112 px / 1024 B: 51 of 256 rows (up to 6.6e-4
    relative);
  - EdgeFace-S, the Ours-ACCURATE cells (see [models.md](models.md)).

  On the same crops, LVFace-L, TopoFR-R100 and EdgeFace-XS are bit-exact, and so is
  every Ours-FAST array checked, so the decoded pixels agree. The cause is not
  established. EERs recomputed from such arrays can move by a trial. When comparing a
  re-embedding with a stored Ours-ACCURATE array, use a tolerance (row cosine at least
  0.99999, or relative difference at most 1e-3).
- **CompressAI.** CompressAI decodes are reproducible only to within one grey level
  ([baselines.md](baselines.md)). Their embeddings differ from the stored ones in
  about half the rows, by up to 6e-3 relative.
- **JPEG-AI and Color FERET.** The JPEG-AI cells of Color FERET partly come from an
  earlier fit rule ([baselines.md](baselines.md)). Public users align NIST Color FERET
  themselves ([datasets.md](datasets.md)), so their index, pairs and impostor sample
  differ from the paper's; their numbers are comparable, not identical.

## Differences from the paper

- **tab:best-config.** The table was assembled by hand. `best_config.py` recomputes it
  from `metrics.csv` with the rule stated in Section 5.4: the Color FERET mean of
  ArcFace and LVFace-L FNMR@1e-4, minimised over resolution. It reproduces every cell
  except JPEG-AI at 1024 B, which is `0.71 @224` in the data (0.7066 %); the paper
  prints `0.73 @224`. The best resolution and the shading are unchanged.
- **KK CVLface IR-101 Ours-ACCURATE at 168 px.** The published `metrics.csv` has no
  rows for these two cells (`ours_accurate_168_{1024,512}`). Their embedding arrays
  exist, so a full `--models all` run adds these two rows (with CIs). No table uses
  them.
- **Matcher input size.** Section 4.1 says every matcher consumes the aligned crop at
  the working resolution. In the code, crops at 64 / 96 / 168 / 224 px are resized to
  112 px before embedding.
- **Section 14.2, independent-matcher check.** The text says that under CVLface IR-101
  the Ours-ACCURATE vs AVIF pair at Color FERET 112 px / 1024 B is not significant
  (p_adj = 0.34). This cell is not among the published CSVs. Running it with the
  published protocol (NumPy scorer, per-cell BH) gives a significant difference; the
  only non-significant pair of that cell is JPEG-AI vs WebP. The command is:

  ```bash
  python experiments/accuracy/compute_significance.py --datasets colorferet \
      --models cvlface_ir101 --res 112 --budgets 1024 --out-suffix _cvlface_ir101
  ```

  The |ΔAUC| ≤ 2e-4 part of the statement holds.
- **Rate-EER figure.** The rate-EER script originally also drew EER against the median
  file size. The figure was never produced, because `median_bytes` is NaN, and it is
  not in the paper; `rate_eer.py` writes the tables only.
