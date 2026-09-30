# Reproducing the compression benchmark, budget compliance and codec speed

This page covers the stage that turns aligned crops into bitstreams, and everything
computed from the bitstreams themselves rather than from face embeddings:

- the compressed grid of every codec (Sections 4-5 of the paper; every later section
  embeds or scores these files);
- the decoded PNG caches used by the embedding and quality stages;
- budget compliance: `tab:fit-kk`, `tab:fit-cf`, `tab:fit-1024`,
  `fig:size-compliance`, `fig:size-box` (Section 5.7);
- the codec-properties table `tab:codec-properties` (Section 2);
- codec speed: `tab:speed`, `fig:speed-trend` (Section 6);
- the JPEG-AI side studies: operation points on AI-Solutions-KK (`tab:jpegai-kk`),
  the operation-point table `tab:jpegai-speed`, and a scan for collapsed streams.

The codecs themselves are documented in [baselines.md](baselines.md) (the ten
baselines: knobs, budget searches, setup of the JPEG-AI reference software) and
[codec.md](codec.md) (face1kb-FAST and face1kb-ACCURATE).

- [Scripts](#scripts)
- [Rendering the tables without any data](#rendering-the-tables-without-any-data)
- [Compression](#compression)
- [Decoded caches](#decoded-caches)
- [Budget compliance](#budget-compliance)
- [Codec-properties table](#codec-properties-table)
- [Speed](#speed)
- [JPEG-AI side studies](#jpeg-ai-side-studies)
- [Cost](#cost)
- [Reproducibility](#reproducibility)
- [Known deviations from the paper](#known-deviations-from-the-paper)

## Scripts

| Script | Output | Cost |
|---|---|---|
| `experiments/compress/compress.py` | `WORK_ROOT/<ds>/compressed/<cell>/<codec>/<rel_stem>.<ext>`, manifests | see [Cost](#cost) |
| `experiments/compress/decode_cache.py` | `WORK_ROOT/<ds>/decoded/<cell>/<codec>/<rel_stem>.png` | GPU |
| `experiments/compress/generate_file_size_boxplots.py` | `OUTPUT_ROOT/codec_comparison/file_size_summary.csv`, `tables/fit_rate_<ds>_<budget>.tex`, `figures/file_size_{compliance,boxplots,boxplots_all}.png` | CPU |
| `experiments/compress/generate_codec_properties_table.py` | `OUTPUT_ROOT/tables/codec_properties.tex` | CPU, seconds |
| `experiments/compress/generate_jpegai_speed_table.py` | `OUTPUT_ROOT/tables/jpegai_speed.tex` | CPU, seconds |
| `experiments/compress/measure_jpegai_kk.py` | `OUTPUT_ROOT/codec_comparison/jpegai_kk.csv`, `tables/jpegai_kk.tex` | GPU, ~1.5 h |
| `experiments/compress/scan_jpegai_collapses.py` | `OUTPUT_ROOT/codec_comparison/jpegai_collapses.csv` | CPU, minutes |
| `experiments/speed/benchmark_speed.py` | `OUTPUT_ROOT/quality/speed_*.csv` | CPU core / GPU, minutes |
| `experiments/speed/benchmark_jpegai_speed.py` | `OUTPUT_ROOT/quality/speed_{cpu,gpu}_jpegai.csv` | CPU core / GPU, minutes |
| `experiments/speed/generate_speed_table.py` | `OUTPUT_ROOT/tables/speed_benchmark.tex`, `figures/speed_trend.png` | CPU, seconds |

`experiments/compress/run.sh` and `experiments/speed/run.sh` run everything at full
scale; the header of each lists its stages and their cost, and `SKIP="..."` skips
stages. Every script prints its options with `--help`. Paths come from
`face1kb.config` (`FACE1KB_DATA_ROOT`, `FACE1KB_WORK_ROOT`, `FACE1KB_OUTPUT_ROOT`,
`FACE1KB_MODELS_ROOT`, `FACE1KB_JPEGAI_DIR`); `compress.py` and `decode_cache.py` take
`--work-root`, the generators `--input-root` / `--from-results` and `--output-root`.

## Rendering the tables without any data

The aggregates behind these tables are shipped in `results/`
(`codec_comparison/file_size_summary.csv`, `jpegai_decoders.csv`, `jpegai_kk.csv`,
`codec_properties.csv`, `quality/speed_*.csv`). Without crops, bitstreams or a GPU:

```bash
python experiments/compress/generate_file_size_boxplots.py --from-summary --from-results
python experiments/compress/generate_codec_properties_table.py --from-results
python experiments/compress/generate_jpegai_speed_table.py
python experiments/compress/measure_jpegai_kk.py --phase table --from-results
python experiments/speed/generate_speed_table.py --from-results
```

This writes `fit_rate_{colorferet,kk}_{1024,512}.tex`, `codec_properties.tex`,
`jpegai_speed.tex`, `jpegai_kk.tex` and `speed_benchmark.tex` to `outputs/tables/`,
and `file_size_compliance.png` and `speed_trend.png` to `outputs/figures/`. All
tables are byte-identical to the published ones except `fit_rate_kk_1024.tex` (see
[Known deviations](#known-deviations-from-the-paper)); the two figures are
pixel-identical with the paper environment's matplotlib (3.11). The size boxplots
(`fig:size-box`) need the bitstreams.

## Compression

```bash
# all classical codecs + JPEG-FzT, Color FERET, full grid, 32 processes
python experiments/compress/compress.py --dataset colorferet --codecs cpu --workers 32
# JPEG-AI on AI-Solutions-KK 112/224 px, shard 1 of 4 (one process per GPU)
python experiments/compress/compress.py --dataset kk --codecs jpeg_ai \
    --resolutions 112,224 --gpu 1 --shard 1:4
# face1kb on the 768 / 960 B budget sweep
python experiments/compress/compress.py --dataset colorferet --codecs ours \
    --resolutions 112 --budgets 768,960 --gpu 0
```

**Input.** The aligned crops `DATA_ROOT/<ds>/aligned_<res>[<suffix>]/<subject>/<stem>.png`
and the index `index.csv`, whose row order defines `--shard`, `--offset` and
`--limit` (see [datasets.md](datasets.md)). `--suffix` (alias `--align-suffix`)
selects another crop set, e.g. `_adv_hfc_006` reads `aligned_112_adv_hfc_006` and
writes the cell `112_adv_hfc_006px_<budget>B`; crops missing from such a set are
recorded as `missing`.

**Codecs.** `--codecs` takes names (`jpeg`, `jpeg2000`, `webp`, `jpeg_xl`, `avif`,
`heif`, `jpeg_fzt`, `jpeg_ai`, `neural_bmshj2018`, `neural_mbt2018_mean`,
`ours_fast`, `ours_accurate`) or groups: `cpu` (the six classical codecs and
JPEG-FzT), `gpu`, `compressai`, `ours`, `baselines`, `all`.

- Classical codecs, JPEG-FzT, JPEG-AI and CompressAI use
  `face1kb.baselines.get_codec(name).encode_to_budget` with the benchmark defaults
  (binary search; JPEG-AI's analytic fit with the HOP profile; see
  [baselines.md](baselines.md)).
- CompressAI in the 768 B and 960 B cells runs with cuDNN's deterministic algorithms
  (`--compressai-deterministic auto`, the default), as the paper's streams of those
  cells were encoded.
- face1kb uses `face1kb.load(variant).encode(img, budget, paper_compat=True)`: the
  paper's budget accounting (the 4-byte raw-geometry trailer at 96 and 168 px is not
  reserved) and the over-budget floor when nothing fits (see [codec.md](codec.md)).
- When no setting fits, every codec stores its smallest output (the paper's
  "failure to compress"). JPEG-AI is the exception when the reference software gives
  no plausible stream at all: the crop is recorded as `failed` and no file is
  written.

**Paper scope.** `experiments/compress/run.sh` encodes:

| Codecs | Color FERET | AI-Solutions-KK |
|---|---|---|
| classical + JPEG-FzT | 64-224 px x 1024/512 B; 112 px x 768/960 B | 64-224 px x 1024/512 B |
| JPEG-AI | 64-224 px x 1024/512 B; 112 px x 768/960 B | 112, 224 px x 1024/512 B |
| bmshj2018, mbt2018 | first 300 index rows of 64-224 px x 1024/512 B; all crops of 112 px x 768/960 B | -- |
| face1kb FAST, ACCURATE | 64-224 px x 1024/512 B; 112 px x 768/960 B | 64-224 px x 1024/512 B |

The adversarial cells are encoded by `experiments/adversarial/run.sh` through the same
script.

**Resume, shards, atomic writes.** An existing output file is never re-encoded
(unless `--overwrite`); its row is recorded from the file size with status
`existing`. `--shard I:N` takes the index rows `I, I+N, ...`; `--offset` and `--limit`
then select a contiguous slice of those rows. Files are written under a temporary
name and renamed, so an interrupted run leaves no partial bitstream.

**Manifests.** Each run writes
`WORK_ROOT/<ds>/compressed/manifest/<codecs>_<res><suffix>_<budgets>[_sh<I>of<N>][_o<offset>n<limit>].parquet`
with one row per task:

| Column | Meaning |
|---|---|
| `dataset, image, subject, rel_path, pose` | the crop (`subject` = crop folder) |
| `codec, res, suffix, budget` | the cell |
| `setting` | selected knob (quality, ratio, bpp x 100, CompressAI quality, face1kb rate index) |
| `bytes`, `fitted` | size of the stored file and `bytes <= budget` |
| `search_bytes`, `search_fitted` | what the codec's search compared with the budget; for CompressAI the entropy-coded bytes, without the 125 / 135 B `.ptci` container overhead |
| `encode_ms` | wall-clock time of the whole budget search |
| `status`, `error` | `encoded`, `existing`, `missing`, `failed` (JPEG-AI) or `error` |

Budget compliance is always judged from the stored file sizes
([below](#budget-compliance)), never from `search_fitted`.

## Decoded caches

```bash
python experiments/compress/decode_cache.py --datasets colorferet,kk --gpus 0,1
python experiments/compress/decode_cache.py --datasets kk --check   # coverage only
```

Decodes every stored JPEG-AI and face1kb bitstream once into
`WORK_ROOT/<ds>/decoded/<cell>/<codec>/<rel_stem>.png` (`config.decoded_dir`), one
process per GPU, skipping PNGs that exist. `face1kb.baselines.decode_file(...,
cache=True)` -- used by the embedding and quality stages -- reads the cached PNG when
it exists and decodes the bitstream otherwise, so the cache changes run time, not
results (a JPEG-AI decode takes about 0.5 s and the 14-matcher roster would pay it 14
times). `--codecs` caches other codecs too; `--budgets 1024,512,768,960` includes the
budget sweep. Each decode is bounded by `--timeout` seconds (default 60, `0`
disables), enforced with `SIGALRM`: a stream on which the JPEG-AI reference decoder
does not return counts as failed instead of stalling its GPU worker. The timeout is
Unix-only; elsewhere decodes are unbounded.

## Budget compliance

```bash
python experiments/compress/generate_file_size_boxplots.py          # scan the files
python experiments/compress/generate_file_size_boxplots.py --from-summary --from-results
```

The scan reads the size of every stored bitstream of the core grid (5 resolutions x
1024/512 B x 12 codecs x 2 datasets; the files themselves are not opened) and writes
`codec_comparison/file_size_summary.csv` (`n, min, mean, median, max, p5, p95,
over_budget_pct, pct_holding` per cell), the fit-rate tables, and the figures.

Fit-rate table rules: `pct_holding` = share of files with `size <= budget`, printed
rounded; a cell with fewer than 30 files reads `--`; a codec without any reported
cell is omitted; per resolution column the lowest printed value is shaded worst and
the highest only if it is unique; columns whose printed values all tie are not
shaded. The CompressAI rows count the `.ptci` container, which is why the 112 px
fit rates are 59 % / 67 % although the search itself always fits.

## Codec-properties table

`generate_codec_properties_table.py` renders `tab:codec-properties`. The descriptive
columns (owner and reference implementation, year of the standard, licensing,
hardware decode) are the paper's condensed wording of `results/codec_properties.csv`
(the codec provenance record) and live in the script; the fit column is computed from
the size summary: the rounded `pct_holding` of the Color FERET 1024 B cells at 112 and
224 px, `n/a` for codecs run only on a subset of the crops (CompressAI).

## Speed

```bash
taskset -c 0 python experiments/speed/benchmark_speed.py --device cpu \
    --codecs jpeg,webp,jpeg_xl,avif,heif,jpeg2000,jpeg_fzt --n 200 \
    --out outputs/quality/speed_cpu_classical.csv
python experiments/speed/benchmark_speed.py --device gpu --gpu 0 \
    --codecs ours_fast,ours_accurate,neural_bmshj2018,neural_mbt2018_mean --n 60 \
    --out outputs/quality/speed_gpu_learned.csv
taskset -c 0 python experiments/speed/benchmark_jpegai_speed.py --device cpu --n 10
python experiments/speed/generate_speed_table.py
```

What is timed: the single-shot encode at the budget-fitting setting and the
single-shot decode, per crop, with models loaded and one untimed warm-up per crop;
the hard-budget search multiplies the encode by its trial count and is not included.
Crops: `--n` Color FERET crops per resolution, stride-sampled from the sorted file
list. CPU rows run on one pinned core with single-threaded libraries.

- Classical codecs and JPEG-FzT: the setting is fitted on the first crop (linear
  scan up to the first overflow) and used for all crops, so the realised median size
  can exceed the budget (e.g. JPEG-FzT 1075 B at 112 px / 1024 B). The JPEG-FzT decode
  time is the full decode, JPEG decode plus the inverse F-transform.
- CompressAI: quality 1, its lowest rate point (about 224 B at 112 px, below the
  budget; the realised size is reported).
- face1kb: the rate index the budget search selects on the first crop; encode is one
  `net.compress` at that gain, decode `Codec.decode_tensor` of the crop's container.
- JPEG-AI (`benchmark_jpegai_speed.py`): target bpp x 100 = `round(1024 * 8 / px *
  100)` capped to 2..65 (65 at 64-112 px, above the benchmark's cap of 50; realised
  about 1130 B at 112 px). The script refuses to measure when the load average exceeds
  0.35 per core (`--allow-load` overrides).

`generate_speed_table.py` reads `speed_cpu_classical.csv`, `speed_gpu_learned.csv`,
`speed_cpu_learned.csv`, `speed_cpu_jpegai.csv` and `speed_gpu_jpegai.csv` in this
order (the first row of a `(codec, device, res, budget)` key wins) and shades the
fastest / slowest cell of each timing column; the realised-size columns are not
ranked.

## JPEG-AI side studies

**Operation points on AI-Solutions-KK** (`tab:jpegai-kk`):

```bash
python experiments/compress/measure_jpegai_kk.py --phase all --gpu 0
```

A subject-stratified sample of 600 AI-Solutions-KK 224 px crops (identities with at
least two crops, shuffled with seed 0, at most 8 per identity, taken round-robin) is
encoded with each JPEG-AI operation point -- SOP, BOP, HOP (`cfg/profiles/simple`,
`base`, `high`) -- at a fixed target of bpp x 100 = 16, which realises about 1111 B
(about 8.5 % over 1024 B). Per operation point: verification EER over all pairs of the
sample (ArcFace antelopev2 and EdgeFace-XS on 112 px bilinear resizes; mated = same
identity; threshold minimising |FMR - FNMR| over the unique scores), mean identity
cosine to the original, median PSNR / SSIM (scikit-image) and median size. The
reconstructions are the encoder-side reconstructions, which equal the decoder
output. The run folder (`--run-dir`, default `WORK_ROOT/kk/jpegai_kk`) holds the
per-crop streams and reconstructions; they are derived from AI-Solutions-KK and must
not be redistributed. Needs the matchers (`python scripts/fetch_models.py --only
anchors`).

**Operation-point table** (`tab:jpegai-speed`): `generate_jpegai_speed_table.py`
renders the shipped `results/codec_comparison/jpegai_decoders.csv` (24 Color FERET
crops at 224 px; CPU and GPU decode, GPU encode, PSNR, SSIM, ArcFace identity cosine,
bytes). The measurement that produced the CSV has no script in this repository.

**Collapsed streams:** `scan_jpegai_collapses.py` counts, per compressed cell, the
`.jpegai` files below 25 % of the budget (the encoder's plausibility floor;
`--frac`), and with `--reencode` whether a fresh encode of the crop collapses again.
It writes only per-cell counts; `--list FILE` adds a per-file list, which names
dataset images and is not for publication. On the paper's streams it finds 3 such
files, all in adversarial cells (112 px HFC 6 px / 512 B: 2, HFC 10 px / 1024 B: 1).

## Cost

One RTX 2080 Ti and one CPU core, indicative, both datasets:

| Stage | Cost |
|---|---|
| classical + JPEG-FzT, all cells | ~200 CPU core-hours (HEIF and JPEG XL dominate) |
| JPEG-AI, all paper cells | ~230 GPU-hours (1-3 encodes of 2-4 s per crop and cell) |
| CompressAI | ~4 GPU-hours |
| face1kb FAST + ACCURATE | ~45 GPU-hours (0.13-0.15 s + 0.34-0.41 s per crop and cell) |
| decoded caches | ~35 GPU-hours (JPEG-AI ~0.5 s per file) |
| size scan | ~20 min (file stat of ~2 M files on network storage) |
| speed benchmarks | ~1 h CPU core + ~20 min GPU |
| JPEG-AI operation points (n=600) | ~1.5 GPU-hours |

## Reproducibility

The scripts reproduce the paper's stored bitstreams byte for byte given the paper
environment ([requirements/paper.txt](../requirements/paper.txt): Pillow 12.2.0,
pillow-heif 1.4.0, pillow-jxl-plugin 1.3.7, compressai 1.2.8, torch 2.10.0 / CUDA
12.8) and, for the GPU codecs, an RTX 2080 Ti (Turing). Checked on the first index
rows (and later slices) of every paper cell: all classical, JPEG-FzT, JPEG-AI,
CompressAI and face1kb files tested were identical, except the Color FERET JPEG-AI
streams that come from an earlier run with another bpp search (see
[Known deviations](#known-deviations-from-the-paper)), and the decoded caches were
pixel- and byte-identical. Other library versions or GPU generations can change the
bytes; see [baselines.md](baselines.md#reproducibility-limits) and
[codec.md](codec.md#limitations). Speed numbers depend on the hardware and the load
of the machine; the settings and realised sizes do not.

## Known deviations from the paper

| Item | Paper | This code |
|---|---|---|
| `tab:speed`, JPEG-FzT decode | 0.16 ms (shaded fastest): only the JPEG decode of the half-resolution image was timed | the full decode including the inverse F-transform, about 30 ms at 112 px on one core |
| `tab:speed`, JPEG-FzT encode | timed as the F-transform downsampling, a JPEG encode to a temporary file and a second in-memory JPEG encode | the downsampling and one in-memory JPEG encode (slightly faster) |
| `tab:speed`, JPEG-AI GPU row (112 px) | an earlier measurement of 38 crops, realised median 1128 B | `benchmark_jpegai_speed.py --device gpu --n 38` stride-samples its crops and realises 1131 B |
| `tab:speed`, face1kb rows | measured before the release weights were final (the ACCURATE rows predate the released 3M-step ACCURATE model); the sizes and rate indexes of the published rows cannot be regenerated | the released weights select slightly different rate indexes and sizes, e.g. FAST at 112 px / 1024 B on GPU: rate index 54 and 1011 B instead of 55 and 1007 B; ACCURATE: 60 and 1007 B instead of 58 and 1001 B. For FAST the cause is not established: the released FAST weights with the same budget search also select 54 |
| `tab:fit-kk` 1024 B (`fit_rate_kk_1024.tex`) | lists bmshj2018 and mbt2018 as rows of `--` | these rows are omitted (AI-Solutions-KK has no CompressAI cell with at least 30 files), as in the 512 B table |
| `tab:jpegai-kk` | ArcFace EER column shaded (SOP best, BOP worst) although the three values are statistically flat | reproduced as published; `--no-eer-shading` leaves the column unshaded |
| Color FERET JPEG-AI cells | 31.6 % of the streams per cell come from an earlier run with another bpp search; 3-4 crops per cell have no stream | a re-run encodes every crop with the final fit, so the Color FERET JPEG-AI results change ([baselines.md](baselines.md#reproducibility-limits)) |
| CompressAI Color FERET 224 px / 1024 B | 303 files (the 300-row subset plus 3 other crops) | 300 files with `--limit 300` |
| mbt2018 main-grid encodes | -- | files with a boundary scale can re-encode to different bytes of the same size (about 2 in 720); the 768 / 960 B cells need deterministic cuDNN, which `compress.py` selects |
| JPEG-AI speed CSVs | the shipped `speed_{cpu,gpu}_jpegai.csv` have the `benchmark_speed.py` columns only | `benchmark_jpegai_speed.py` also writes `device_name` and `loadavg` (ignored by the table generator) |
