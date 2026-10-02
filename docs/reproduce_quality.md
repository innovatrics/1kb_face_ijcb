# Reproducing the image-quality, codec-comparison and fairness results

This page covers three experiment areas:

- **image quality** (`experiments/quality/`, Section 6 and Figure 1): the
  full-reference quality of every compressed crop (PSNR, SSIM, MS-SSIM, LPIPS, DISTS),
  the quality matrix and budget sweep, the face-image-quality (FIQ) grid, and the
  decoded-crop montages;
- **codec comparison** (`experiments/codec_comparison/`, Sections 5.1, 6.8 and 7.5):
  the rate-identity comparison on a 64-crop sample and the JPEG-AI operation-point
  figure;
- **fairness** (`experiments/fairness/`, Section 11): subgroup EER, disparity,
  differential FMR and the bootstrap CIs of the skin-tone disparity.

The metric definitions are in [metrics.md](metrics.md), the codecs in
[baselines.md](baselines.md) and [codec.md](codec.md), the face-recognition models in
[models.md](models.md), and the attribute files in [datasets.md](datasets.md). The
fairness figures (Figures 40, 41) and the quality-vs-identity figure (Figure 22) are
drawn by `experiments/figures/` (see [results.md](results.md)).

| Step | Script | Input | Output | Cost |
|---|---|---|---|---|
| quality | `experiments/quality/measure_quality.py` | compressed cells, aligned crops | `WORK_ROOT/<ds>/quality/*.parquet` (per image, local), `OUTPUT_ROOT/quality/quality_<ds>.csv`, `quality_summary.csv` | 1 GPU; tens of ms per crop, ~10 h for the full grid of both datasets (JPEG-AI from its decoded cache; decoding JPEG-AI live adds ~40 h) |
| FIQ | `experiments/quality/fiq.py` | same, ArcFace (`arcface_antelopev2`) | `OUTPUT_ROOT/quality/fiq_<ds>.csv` | 1 GPU, ~6 min per cell, ~1 h for the 8 cells |
| quality tables | `experiments/quality/quality_tables.py`, `fiq_report.py` | the CSVs above | `OUTPUT_ROOT/tables/{quality_matrix,budget_sweep,fiq_grid}.tex`, `OUTPUT_ROOT/figures/fiq_by_codec.png` | CPU, seconds |
| montages | `experiments/quality/visual_quality.py` | compressed cells, aligned crops, quality CSVs | `OUTPUT_ROOT/figures/*.png` (dataset faces, local only) | 1 GPU, ~1 min |
| codec comparison | `experiments/codec_comparison/evaluate.py` | aligned crops (codecs run on the fly) | `OUTPUT_ROOT/codec_comparison/res<res>/comparison.json` | 1 GPU + CPU (the classical scans dominate); for 64 crops x 2 budgets and one dataset ~70 min at 112 px and ~90 min at 224 px without JPEG-AI, JPEG-AI adds ~15 / ~20 min, i.e. ~3 h (112 px) and ~4 h (224 px) for both datasets |
| comparison tables | `experiments/codec_comparison/tables.py` | `comparison.json`, `quality_summary.csv` | `OUTPUT_ROOT/tables/codec_comparison_{112,224}.tex`, `codec_results{,_112}.tex` | CPU, seconds |
| JPEG-AI figure | `experiments/codec_comparison/jpegai_tradeoff.py` | `codec_comparison/jpegai_decoders.csv` | `OUTPUT_ROOT/figures/jpegai_decoder_tradeoff.{png,pdf}` | CPU, seconds |
| subgroup EER | `experiments/fairness/subgroup.py` | anchor embeddings, `index.csv`, `pairs.parquet`, `labels.csv` / `attributes.csv` | `OUTPUT_ROOT/fairness/fairness_<ds>.csv`, `fairness_disparity_<ds>.csv` | 1 GPU, ~10 s per embedding array, ~3-4 h for 4 anchors x both datasets |
| differential FMR | `experiments/fairness/fmr_fairness.py` | same | `OUTPUT_ROOT/fairness/fmr_fairness_<ds>.csv` | 1 GPU, ~1 min per anchor and dataset |
| disparity CIs | `experiments/fairness/disparity_ci.py` | KK embeddings and attributes | `OUTPUT_ROOT/fairness/disparity_ci_kk.csv` | 1 GPU, ~40 s per (anchor, codec), ~30 min |
| fairness tables | `experiments/fairness/tables.py` | the fairness CSVs | `OUTPUT_ROOT/tables/*.tex` (7 tables) | CPU, seconds |

`experiments/quality/run.sh`, `experiments/codec_comparison/run.sh` and
`experiments/fairness/run.sh` run the steps of each area at full scale; each script
prints its options with `--help`. All paths come from `face1kb.config`
(`FACE1KB_DATA_ROOT`, `FACE1KB_WORK_ROOT`, `FACE1KB_OUTPUT_ROOT`,
`FACE1KB_MODELS_ROOT`, `FACE1KB_WEIGHTS_DIR`); the scripts take `--out-dir` (and the
table generators `--input-root` / `--from-results`) to override them. The GPU costs
above were measured on an RTX 2080 Ti.

The dependencies are in the `eval` and `codecs` extras (`pip install -e
".[eval,codecs]"`); JPEG-AI needs `scripts/setup_jpegai.sh` (the `jpegai` extra).

## Rendering the paper tables without any data

The aggregates of the paper are shipped in `results/`. No crops, embeddings or GPU are
needed to render the tables and the two data figures of this page from them:

```bash
FROM_RESULTS=1 PAPER_HEADER=1 SKIP="1 2 4" FIQ_SHADING=raw bash experiments/quality/run.sh
FROM_RESULTS=1 PAPER_HEADER=1 SKIP="1" bash experiments/codec_comparison/run.sh
FROM_RESULTS=1 PAPER_HEADER=1 SKIP="1 2 3" bash experiments/fairness/run.sh
```

This writes to `outputs/tables/` and `outputs/figures/`:

- 10 tables byte-identical to the arXiv sources: `quality_matrix`, `budget_sweep`,
  `fiq_grid` (with `--shading raw`, see below), `codec_comparison_112`,
  `codec_comparison_224`, `codec_results`, `codec_results_112`,
  `fairness_disparity_cf`, `fairness_disparity_512`, `fmr_fairness_cf`;
- `fairness_subgroup_cf`, `fairness_subgroup_kk` and `fairness_disparity_kk`: the
  tables that the paper types inline in Section 11 (Tables 59, 60, 61); the emitted
  rows and values equal the paper's, the whitespace differs;
- `disparity_ci_kk` (Table 63), which differs from the paper in two rows (see
  [Differences from the paper](#differences-from-the-paper));
- `fiq_by_codec.png` (Figure 23) and `jpegai_decoder_tradeoff.png` (Figure 30),
  pixel-identical to the paper PNGs in the paper environment
  (`requirements/paper.txt`).

`--paper-header` (`PAPER_HEADER=1`) writes the first comment line of each published
table, which names the script that originally produced it; without it, the comment
names the generator in `experiments/`. The montages (step 4 of the quality area) need
the crops and the compressed cells, see below.

## Image quality

### Full-reference metrics (`measure_quality.py`)

For each (dataset, codec, resolution, budget) cell, every bitstream under
`WORK_ROOT/<ds>/compressed/<res>px_<budget>B/<codec>/` is decoded with
`face1kb.baselines.decode_file` and scored against the lossless crop
`DATA_ROOT/<ds>/aligned_<res>/<subject>/<stem>.png`:

| Metric | Implementation | Better |
|---|---|---|
| PSNR | `pyiqa` `psnr` (RGB) | higher |
| SSIM | `pyiqa` `ssim` (Y channel) | higher |
| MS-SSIM | `piq.multi_scale_ssim`, 7 px kernel, 4 scales (valid down to 64 px crops) | higher |
| LPIPS | `pyiqa` `lpips` (AlexNet, v0.1) | lower |
| DISTS | `pyiqa` `dists` | lower |

Details are in [metrics.md](metrics.md#image-quality-quality_datasetcsv). Crops are
scored in batches of 64 on `--device` (default `cuda`). Decoding:

- classical codecs through Pillow and its plugins, JPEG-FzT through its inverse
  F-transform, CompressAI through the `.ptci` container (checked by re-encoding, see
  [baselines.md](baselines.md));
- JPEG-AI and the two face1kb codecs from the decoded PNG cache
  (`WORK_ROOT/<ds>/decoded/...`) when it exists, else with the reference decoder /
  `face1kb.decode` on the GPU. The face1kb bitstreams must be decoded on CUDA.

A file that fails to decode, decodes to another size or has no reference crop is
skipped (and counted in the log). Each cell writes one parquet shard of per-image
records to `--shard-dir` (default `WORK_ROOT/<ds>/quality/`); a cell whose shard
exists is skipped unless `--overwrite`, so a run can be resumed. The shards are
derived from the dataset images and must stay local. `--aggregate-only` rebuilds the
CSVs from the shards: `quality_<ds>.csv` holds the per-cell medians
(`dataset, codec, res, budget, n, psnr, ssim, ms_ssim, lpips, dists`) of the codecs of
the study, and `quality_summary.csv` concatenates the per-dataset files present in the
output folder.

```bash
python experiments/quality/measure_quality.py --datasets colorferet          # all cells
python experiments/quality/measure_quality.py --datasets kk --codecs webp \
    --resolutions 112 --budgets 1024 --limit 200                             # a pilot
python experiments/quality/measure_quality.py --datasets kk --aggregate-only
```

`quality_tables.py` renders the quality matrix (Table 33: Color FERET, 1024 B, 112 and
224 px; codecs sorted by SSIM; best / worst per column shaded; a codec with fewer than
half of the block's median crop count, i.e. the 300-crop CompressAI subset, is marked
`‡` and not ranked) and the budget sweep (Table 35: WebP and Ours-ACCURATE, 112 px,
1024 and 512 B, both datasets).

### Face image quality (`fiq.py`, `fiq_report.py`)

The FIQ of an image is the mean pairwise cosine of the ArcFace embeddings of 10 mildly
augmented copies (flip, 90-100 % crop, gamma, brightness; SER-FIQ with the
stochasticity at the input). The same augmentation parameters are used for an original
and all its reconstructions; everything is resized to the 112 px matcher input. Per
cell ({112, 224} px x {1024, 512} B), 250 crops are drawn (seed 0) from the crops
present for every stored baseline codec; the face1kb codecs are encoded and decoded on
the fly with the released weights (`paper_compat=True`). `fiq_<ds>.csv` holds
`n, fiq_mean, fiq_median, fiq_drop_vs_original` per codec and cell.

`--codecs` recomputes only the named codecs and replaces their rows in existing CSVs.
In that mode the sample is drawn from the crops common to the *selected* stored codecs
(all aligned crops when only face1kb codecs are selected), so it can differ from the
sample of a full run.

`fiq_report.py` writes the grid (Table 34) and `fiq_by_codec.png` (Figure 23). Its
`--shading printed` (default) ranks the printed three-decimal values and shades every
tied extreme; `--shading raw` ranks the unrounded means and shades one best and one
worst cell per column, which is how the published table was shaded.

### Montages (`visual_quality.py`)

Renders Figures 1, 8 and 24-28 under their paper file names (`visual_codec_grid_cf/kk`,
`visual_ours_cf/kk`, `visual_budget_sweep`, `visual_resolution_sweep`,
`budget_224_cf/kk`, `comparison_1kb`, `abstract_codec_strip`) from the stored
bitstreams; tiles are captioned with the cell medians of `quality_<ds>.csv`
(`--quality-root` or `--from-results`), the abstract strip with the metrics of the
shown crop. The default samples are those of the paper: Color FERET images by NIST
file name, AI-Solutions-KK images by their row id in `index.csv`; `--cf-samples`,
`--kk-samples` and `--abstract-sample` pick others. `--only` renders a subset.

**The montages show dataset faces.** Write them to a local folder only; do not commit,
publish or redistribute them. Rendering needs CUDA (the face1kb, CompressAI and
JPEG-AI tiles).

## Codec comparison

`evaluate.py --res {112,224}` draws 64 crops per dataset (`random.Random(0).sample` of
the sorted `aligned_<res>/*/*.png` files), compresses each to 1024 and 512 B on the
fly and compares the reconstruction with the crop:

- classical codecs: a linear scan of the benchmark grid, keeping the largest file that
  fits (if none fits, the smallest setting, which exceeds the budget); JPEG-FzT: even
  qualities 2..94, stopping at the first overflow; JPEG-AI: the analytic target with up
  to two step-down re-encodes (`--no-jpegai` or a missing reference software skips
  it); face1kb: `encode(..., paper_compat=True)`;
- `id_cos`: cosine of the two embeddings of the identity model at its 112 px input
  (224 px crops are resized with an antialiased bilinear filter);
- `psnr`, `ssim`, `lpips`, `bytes`: native-resolution diagnostics and the emitted
  size.

`--codecs` evaluates only the named codecs (`jpeg`, `webp`, `avif`, `heif`,
`jpeg_xl`, `jpeg2000`, `jpeg_fzt`, `jpeg_ai`, `ours_fast`, `ours_accurate`); the sample
does not depend on the selection, so its rows equal the same rows of a full run.
With `--codecs` an existing `comparison.json` in the output directory is updated: only
the recomputed (dataset, codec, budget) rows are replaced or added and all other rows
are kept, so one codec can be added to a full comparison. A run without `--codecs`
rewrites the file.
`comparison.json` holds the medians per (dataset, codec, budget) and the name of the
identity model. `tables.py` takes id-cos from it and PSNR / SSIM / LPIPS from
`quality_summary.csv` (the full measurement at native resolution).

**Identity model.** The id-cos columns of the paper (Tables 7, 8, 40, 41 and the
id-cos numbers quoted in the text) were measured with a proprietary face matcher that
is not distributed. They cannot be regenerated; the shipped
`results/codec_comparison/res*/comparison.json` files are their only source, and
`tables.py --from-results` renders the published tables from them. `evaluate.py`
defaults to the public held-out matcher `cvlface_ir101` (`--id-model`), whose
id-cos values differ from the paper's. Any model of `face1kb.fr` can be used,
including your own: register it in a plugin named in `FACE1KB_FR_PLUGINS`
([models.md](models.md)), or pass an ONNX file directly:

```bash
python experiments/codec_comparison/evaluate.py --res 112 --id-model my_matcher \
    --id-onnx my_matcher.onnx --id-onnx-color BGR --id-onnx-normalize "[-1,1]"
```

The identity model receives uint8 crops: the face1kb reconstructions and the
112 px resize of 224 px crops are rounded to uint8 before embedding. The original
study embedded these as floats, which moves the median id-cos by up to about 5e-3 even
with the same matcher (see [Differences from the paper](#differences-from-the-paper)).

`jpegai_tradeoff.py` draws Figure 30 from `jpegai_decoders.csv` (the 24-crop 224 px
timing study of the SOP / BOP / HOP operation points, rounded to whole ms and 0.01 dB
as in Table 37); `--gpu-label` sets the legend entry of the GPU bars.

## Fairness

All three computations score all mated pairs plus a seeded non-mated sample (seed 0)
of the four anchor matchers (ArcFace, LVFace-L, TopoFR-R100, EdgeFace-XS) with the
CUDA scorer (`--device cuda`, the published numbers; `--device cpu` uses the NumPy
scorer, which can move an EER by one impostor trial). `--allow-cpu-fallback` lets a
GPU that runs out of memory fall back to the CPU. The library functions and exact
definitions are in [metrics.md](metrics.md#fairness-fairness_csv-fmr_fairness_csv-disparity_ci_kkcsv).

| Script | Sample | Grid | Output |
|---|---|---|---|
| `subgroup.py` | 5,000,000 non-mated | every core-grid source (aligned and all codec cells, no crop variants) | per-subgroup EER; max - min / std over subgroups with >= 1,000 impostor pairs |
| `fmr_fairness.py` | 300,000 | 112 px, 512 / 1024 B | per-subgroup FMR at the global threshold of overall FMR 1e-2 and 1e-3; `--nan-policy` |
| `disparity_ci.py` | 400,000 | KK, 112 px / 1024 B, aligned + 9 codecs | skin-tone disparity (uniform MST5/6/7/9/10 basis) and amplification ratio with 500-replicate identity-bootstrap CIs |

Attributes: Color FERET pose class, ethnicity (the NIST race field), gender and age
band from `labels.csv`; AI-Solutions-KK Monk skin tone, gender and age band from
`attributes.csv`, propagated per identity ([datasets.md](datasets.md)).

`tables.py` renders Tables 59-65 (`--tables` selects a subset; `--res`, `--budget`,
`--target-fmr` and `--subgroup-model` change the slice). `fmr_fairness_cf` omits a
codec whose disparity prints 0.00 for every anchor; `fairness_disparity_512` is
unshaded, as published.

## Differences from the paper

The code and the shipped files are what produced the published numbers; where the
paper disagrees, they are the reference.

- **Table 63** (`disparity_ci_kk`): the aligned, Ours-ACCURATE and Ours-FAST rows match
  `disparity_ci_kk.csv`; the WebP and JPEG 2000 rows of the paper do not (the CSV gives
  WebP 1.6 [1.1,2.9] / 1.1 [0.7,2.3] / 1.1 [0.8,2.5] / 1.1 [0.9,1.6] and JPEG 2000
  10.7 [4.5,29.4] / 8.7 [3.9,31.3] / 12.6 [4.9,33.9] / 4.3 [2.1,9.5] for ArcFace /
  LVFace-L / TopoFR-R100 / EdgeFace-XS), so the regenerated table differs in those
  rows and their shading.
- **Table 34** (`fiq_grid`): the published shading ranks unrounded means
  (`--shading raw`); the default rule shades all cells tied at the printed precision.
  The Ours-ACCURATE rows (both datasets) and the AI-Solutions-KK JPEG-AI rows were
  computed in the `--codecs` mode, i.e. on a different 250-crop sample than the other
  rows of their column; a full run of `fiq.py` gives, e.g., 0.954 instead of 0.952 for
  Ours-ACCURATE on Color FERET at 112 px / 1024 B.
- **Table 65** caption: JPEG-AI and the CompressAI baselines are omitted because their
  disparity is 0.00 for every anchor; the cause is that a few crops have no embedding
  (NaN scores), which makes the threshold NaN and every FMR 0 under the default
  `--nan-policy propagate` ([metrics.md](metrics.md)).
- **Tables 7, 8, 40, 41**: at 224 px / 512 B several codecs cannot reach the budget
  on the comparison sample and fall back to their smallest setting (median bytes in
  `comparison.json`, e.g. WebP 775 B on Color FERET and 1,155 B on AI-Solutions-KK,
  also at 1024 B on AI-Solutions-KK); the tables do not show the sizes.
- `comparison.json` id-cos: the study embedded the unrounded float reconstructions of
  the face1kb codecs and the float 112 px resize of the 224 px crops; `evaluate.py`
  rounds both to uint8 (the input type of `face1kb.fr`). Rerun with the study's
  matcher, all other fields (`n`, `bytes`, `psnr`, `ssim`, `lpips`) are reproduced
  exactly, while the median id-cos moves by up to 1.1e-3 (face1kb codecs, 112 px) and
  5.4e-3 (224 px, all codecs; largest for JPEG 2000 at 512 B); the 112 px rows of the
  other codecs, whose inputs are already uint8, differ by < 2e-6 (runtime noise).
- `quality_kk.csv`: the Ours-ACCURATE rows at 64, 96 and 168 px (both budgets) were
  measured on the bitstreams of an earlier ACCURATE checkpoint; a regenerated file
  differs in those six rows (e.g. KK 64 px / 512 B: all 64 checked crops differ, by up
  to 0.9 dB PSNR). No paper table uses them (the tables use 112 and 224 px).
- CompressAI quality rows: the study decoded the `.ptci` files without the
  re-encoding check, and about 1-2 % of the mbt2018 crops were decoded corrupt
  (e.g. one of 64 checked Color FERET 112 px / 1024 B crops: 24.2 dB instead of
  37.9 dB). The decoder of this release detects and repairs those; mbt2018 also
  decodes with GPU float nondeterminism (up to ~2e-4 dB). Regenerated CompressAI medians
  therefore differ slightly. The quality files hold CompressAI rows only for Color
  FERET at 112 px / 512 and 1024 B (300 crops); none at 768 or 960 B.
- `quality_colorferet.csv` and `quality_summary.csv` in `results/` went through one
  extra pandas read/write after aggregation; a regenerated file has the same values up
  to the last printed digit (<= 4e-15) and is text-identical after one read/write
  (`quality_kk.csv` is text-identical directly).

## Validation

- The table generators above were run on the shipped `results/` and compared with the
  arXiv sources (tests: `tests/experiments/quality/test_paper_tables.py`).
- Fairness: `subgroup.py` (EdgeFace-XS, 112 px, aligned + 1024 B, both datasets),
  `fmr_fairness.py` (LVFace-L, both datasets) and `disparity_ci.py` (ArcFace and
  TopoFR-R100) were rerun on the study embeddings, and so were `subgroup.py` (ArcFace,
  112 px, aligned + 512 B) and `fmr_fairness.py` (TopoFR-R100); every row
  (1,033 + 1,488 + 20 + 525 + 1,488) is text-identical to the shipped CSVs.
- Quality: 12 cells x 64 crops (classical, JPEG-FzT, mbt2018, JPEG-AI, both face1kb
  codecs; 64, 112 and 224 px) were rescored and compared per image with the study
  records: exact for PSNR, SSIM, MS-SSIM and DISTS apart from 1-ulp float32 PSNR
  differences, LPIPS within 6e-8, the two exceptions listed above; the aggregation of
  the study records reproduces `quality_{colorferet,kk}.csv` (108 / 94 rows).
- FIQ: the Color FERET 112 px / 1024 B cell was rerun; all 11 rows match
  `fiq_colorferet.csv` to 1e-16 (Ours-ACCURATE in the `--codecs` mode).
- Codec comparison: `evaluate.py` was rerun at 112 and 224 px on both datasets (64
  crops, 1024 and 512 B, every codec, the study's matcher registered with `--id-onnx`;
  JPEG-AI and the classical codecs at 224 px in separate `--codecs` runs). All 80 rows
  of `comparison.json` reproduce `n`, `bytes` and `psnr` exactly and `ssim` / `lpips`
  to 1.2e-7; id-cos differs as described above.
- Montages: rendered from the study bitstreams, 5 of 10 are pixel-identical to the
  paper PNGs and the other 5 differ in at most 12 pixels by one grey level.
