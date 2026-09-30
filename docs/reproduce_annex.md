# Reproducing the ISO/IEC 29794-5 annex study

This page covers the drivers in `experiments/annex/`. They reproduce the paper section
"The ISO/IEC 29794-5 Annex E/F parameter tables": `tab:annex-exp1`, `tab:annex-ci`,
`tab:annex-joint`, `tab:annex-axes`, `tab:annex-manip`, `tab:annex-bytes`,
`tab:annex-selfsim` and `tab:annex-jpegai`.

The study compares three sets of 1024 B compression settings for six classical codecs
(JPEG, JPEG 2000, WebP, JPEG XL, AVIF, HEIF) and JPEG-AI:

* the per-codec resolution, colour mode, image manipulation and flags of Annexes E
  and F;
* our recommended configuration;
* a joint sweep over resolution x colour x manipulation, followed by a flag sweep.

Each configuration is scored with the annexes' objective (self-similarity under their
matcher, CVLface AdaFace ViT-B) and with verification EER.

- [Quick start: the paper tables from the shipped results](#quick-start-the-paper-tables-from-the-shipped-results)
- [Full run](#full-run)
- [Scripts](#scripts)
- [Outputs](#outputs)
- [Cost](#cost)
- [Reproducibility and deviations](#reproducibility-and-deviations)
- [Third-party components](#third-party-components)

## Quick start: the paper tables from the shipped results

```bash
python experiments/annex/generate_annex_tables.py --from-results
# -> outputs/annex/annex_{exp1,ci,joint,axes,manip,selfsim,bytes,jpegai}.tex
```

The generator reads `results/annex/{scores,scores_pop2000,ci}.csv`. All eight tables
are byte-identical to the paper's. This step takes seconds on a CPU.

## Full run

```bash
bash experiments/annex/run.sh        # DEVICE=cuda WORKERS=24 SKIP="..."
```

| Step | Command | What |
|---|---|---|
| 0 | `prep.py` | evaluation subsets, crop cache (56-224 px), 106-point landmarks |
| 1 | `sweep.py --stage baseline` | uncompressed references, both datasets |
| 2 | `sweep.py --stage exp1` | ours vs Annex E/F, both datasets, full population |
| 3 | `sweep.py --stage A --limit 2000` | 336-cell grid, first 2000 Color FERET crops |
| 4 | `sweep.py --stage jpegai --limit 500` | JPEG-AI arm, first 500 Color FERET crops |
| 5 | `score.py --pop-limit 2000`, `stageb.py`, `sweep.py --stage B`, `score.py --pop-limit 2000` | Stage-B flag sweep; `scores_pop2000.csv` |
| 6 | `stageb.py --confirm`, `sweep.py --stage confirm` | the six winners, full population, both datasets |
| 7 | `score.py` | `scores.csv` |
| 8 | `ci.py` | `ci.csv` |
| 9 | `generate_annex_tables.py` | the eight tables |

`scores_pop2000.csv` is final after step 5, before the confirmation run. This
matters because the confirmation run records its six cells under the arm `winner`
and recomputes their arrays with more matchers. Scoring the prefix again after step
6 changes `tab:annex-joint` (the grid winners of AVIF and JPEG, among others, move)
and one cell of `tab:annex-bytes`, which we checked on the study's data. The shipped
results were scored in the order of `run.sh`.

Prerequisites:

* the aligned crops `aligned_<res>` and `index.csv` of both datasets
  ([datasets.md](datasets.md), `experiments/prepare/run.sh`);
* `pip install -e ".[codecs,eval,jpegai]"`, `scripts/setup_jpegai.sh` (for step 4)
  and `python scripts/fetch_models.py --only anchors,cvlface_vit_b`;
* the command-line tools `cwebp` (libwebp) and `jpegtran` (libjpeg-turbo built with
  arithmetic coding). The sweep stops with an error when a cell needs a tool that is
  missing.

## Scripts

**`prep.py`** builds, per dataset, `WORK_ROOT/<dataset>/annex/cache/`
(`--cache-root DIR` writes `DIR/<dataset>/` instead):

* `meta.parquet`: the evaluation subset (`sid, image, rel_path, frontal, id`).
  * Color FERET keeps at most 4 crops per subject, AI-Solutions-KK at most 30 per
    identity.
  * Frontal crops come first; rows are sorted by (subject, frontal first, image stem).
  * This gives 3,976 Color FERET crops (994 subjects, 2,483 frontal) and 3,150
    AI-Solutions-KK crops (105 identities).
  * A Color FERET crop is frontal when its pose code is `fa` or `fb`; every
    AI-Solutions-KK crop counts as frontal.
  * The row order matters: the impostor samples of `score.py` and `ci.py` are seeded
    draws over row indices.
* `crops_<res>.npy` for 56, 64, 80, 96, 112, 168 and 224 px, read from
  `aligned_<res>` in the order of `meta.parquet`.
  * A resolution without a folder is derived from a larger one: exactly by
    subsampling when the size divides (56 px = `aligned_112[::2, ::2]`), otherwise by
    resampling the 224 px crop (80 px), which logs a warning.
  * `crops.json` records the source of each resolution.
* `lmk106.npy`: the InsightFace `2d106det` landmarks of the 224 px crops, with the
  whole crop as the face box. `--ctx-id -1` runs it on the CPU. The model comes from
  the pinned `buffalo_l` pack (`face1kb.data.fetch`).

**`sweep.py --stage S --dataset D`** runs the cells of one stage.

* A cell is (codec, resolution, colour, manipulation, flags, 1024 B), with the id
  `<codec>_r<res>_<color>_<manip>_<flags>_b1024`.
* Each crop of the subset is manipulated, encoded to the budget, decoded, resized to
  112 px and embedded, all in memory.
* Per cell the sweep writes the arrays `WORK_ROOT/<dataset>/annex/emb/<cell_id>__<model>.npy`
  (`(N, 512)` float16, NaN rows for crops that were not run) and one statistics row
  in `WORK_ROOT/<dataset>/annex/cells_<stage>_<pid>.parquet`. The row holds the median
  bytes, the median bytes at a fixed quality (50; JPEG 2000 rate 100), the fit rate
  and `n`.
* Manipulations:
  * `mean`: a 3 x 3 box blur of the whole crop.
  * `rectangle_mean`: the crop is kept sharp inside a hard rectangle around the five
    template landmarks, grown by 20 % of the resolution on each side, and blurred
    outside it.
  * `ofiq_landmarks`: the crop is kept sharp inside the hard face-contour polygon of
    the 106-point landmarks and blurred outside it.
* Encoding: a binary search for the largest quality (2..94) that fits, or for JPEG
  2000 the largest rate on a geometric ladder from 400 down to about 2. The search
  starts from the previous crop's answer.
* Codec specifics:
  * JPEG uses `optimize=True`; its `arithmetic` flag is applied by
    `jpegtran -arithmetic`.
  * WebP cells with an `sns` flag use `cwebp -size 1024`, since Pillow does not
    expose `sns`.
  * JPEG XL effort 10 runs as effort 9.
  * JPEG-AI (`--stage jpegai`) uses the reference software with a warm-started
    target bit rate. Greyscale crops are passed as grey RGB.
* Stages and their matchers:
  * `baseline`, `exp1`, `confirm` and `jpegai`: the four anchors plus `cvlface_vit_b`.
  * `A` and `B`: `arcface_antelopev2` and `cvlface_vit_b`.
* A cell is skipped when all its arrays exist with at least the requested number of
  finite rows.
* `--only ID,...` and `--shard k/K` select cells, and `--models` overrides the
  matchers.

**`score.py`** scores every array whose cell has a statistics row. When several runs
recorded a cell, the most recent statistics file wins.

* `self_sim` is the mean cosine to the uncompressed 112 px baseline cell of the same
  matcher.
* `eer` and `fnmr_0.001` / `fnmr_0.0001` are computed on all mated pairs plus
  2,000,000 non-mated pairs drawn with `numpy.random.default_rng(0)`.
* Both protocols are computed: `sym` (both sides compressed) and `asym` (compressed
  against uncompressed).
* Both populations are computed: `all` and `frontal`.
* `--pop-limit 2000` scores only the first 2000 crops, the population the grid ran
  on.
* `--device cuda` computes the cosines on the GPU, as in the paper run.

**`stageb.py`** selects cells on the 2000-crop scores. The criterion is the EER of the
`all` population, symmetric protocol, averaged over `arcface_antelopev2` and
`cvlface_vit_b`.

* By default it writes `stage_b_cells.json`: every codec flag set on top of the two
  best grid cells of each classical codec (62 cells).
* With `--confirm` it writes `confirm_cells.json`: the best cell per codec over the
  grid and Stage B (6 cells).
* The JPEG-AI grid cells are excluded from both lists; they have no flag sets.

**`ci.py`** runs a subject-level bootstrap per dataset and codec.

* It covers four arms: ours, Annex E, Annex F and the confirmed winner.
* Scores are fused as the mean over the four anchors, on all mated pairs plus 400,000
  seeded non-mated pairs.
* The arm's best cell gets a 95 % interval of its EER over 200 subject resamples, and
  the paired EER difference to ours on the same resamples.
* The default device is `cuda`; `--device cpu` works but is slow.

**`generate_annex_tables.py`** writes the eight tables from `scores.csv`,
`scores_pop2000.csv` and `ci.csv`. It reads `--input-dir`, `--from-results` or, by
default, `OUTPUT_ROOT/annex`.

## Outputs

| File | Content |
|---|---|
| `WORK_ROOT/<dataset>/annex/cache/` | subset, crop cache, landmarks (per-image data; keep them local) |
| `WORK_ROOT/<dataset>/annex/emb/*.npy` | per-image embeddings (keep them local) |
| `WORK_ROOT/<dataset>/annex/cells_*.parquet` | per-cell byte statistics |
| `OUTPUT_ROOT/annex/stage_b_cells.json`, `confirm_cells.json` | cell lists |
| `OUTPUT_ROOT/annex/scores.csv`, `scores_pop2000.csv` | one row per (cell, matcher): cell definition, byte statistics, `self_sim`, `<pop>_<protocol>_{eer,fnmr_0.001,fnmr_0.0001}` |
| `OUTPUT_ROOT/annex/ci.csv` | `dataset, codec, kind, cell_id, eer, lo, hi, d_lo, d_hi, d` |
| `OUTPUT_ROOT/annex/annex_*.tex` | the eight tables |

## Cost

One RTX 2080 Ti. Compression is CPU-bound: a cell of 256 crops takes about 7 s with
16 worker processes.

| Step | Cost |
|---|---|
| prep | about 3 min per dataset |
| baseline + exp1 | about 30 min per dataset |
| Stage A (336 cells x 2000 crops) | about 4 h |
| JPEG-AI arm (16 cells x 500 crops) | about 8 GPU-hours (2-3 s per encode) |
| Stage B (62 cells x 2000 crops) | about 1 h, plus about 5 min per scoring pass |
| confirm | about 15 min per dataset |
| score, ci, tables | about 15 min on the GPU |

## Reproducibility and deviations

These checks used the study's stored embedding arrays and a subset rebuilt by
`prep.py` from the aligned crops.

* **Subset.** The subset rebuilt from the aligned crops has the same rows, in the
  same order, as the study's subset on both datasets.
* **Scores.** `score.py` reproduces all 1,152 rows of `scores.csv` exactly, in the
  same order. For `scores_pop2000.csv`, all 984 rows are present and every byte
  statistic, FNMR and frontal-population value is identical. 982 of the 984 rows are
  identical in the remaining columns as well. The two exceptions belong to JPEG XL
  80 px, whose stored arrays were later recomputed by the confirmation run; they
  differ by at most 2.6e-5 in self-similarity and 2e-6 in EER.
* **Bootstrap.** `ci.py` reproduces all 44 rows of `ci.csv` exactly.
* **Tables.** The eight tables rendered from the shipped aggregates are
  byte-identical to the paper's.
* **Cell selection.** `stageb.py` regenerates the 62 Stage-B cells and the 6 winners
  of the study.
* **Color FERET crops** at 56, 64, 96, 112, 168 and 224 px are identical to the
  study's; the 80 px crops are an approximation (below).
  * Re-running the Experiment-1 cells on the first 256 crops with
    `arcface_antelopev2` gives identical float16 embeddings for 15 of the 17 cells.
    This includes the 56 px Annex cells and the contour-masked JPEG XL cell, which
    checks the landmarks.
  * The 80 px Annex-F WebP cell differs, because the paper's 80 px crops were warped
    directly from the source photographs. The public 80 px crops are resampled from
    224 px (median PSNR about 38 dB against a direct warp). The 80 px cells enter
    `tab:annex-exp1` (Annex-F WebP), the JPEG XL winner and 48 grid cells.
  * The other difference is one crop in 256 of the default WebP cell (see
    "Warm-started search" below).
* **AI-Solutions-KK annex crops** were warped separately from the source photographs
  for the study, and differ from the released crops at every resolution
  ([datasets.md](datasets.md#known-deviations-from-the-paper)). The KK columns of
  `tab:annex-exp1` and `tab:annex-ci` therefore do not reproduce bit for bit from
  the released crops.
* **Warm-started search.** The budget search starts from the answer of the previous
  crop handled by the same worker process, so it depends on how crops are scheduled
  across workers.
  * Where a codec's file size is not monotone in its quality, the fitted setting can
    differ between runs.
  * Example: one WebP crop gave 1030 B at quality 68 and 1024 B at quality 70. The
    search returns quality 70 or 66, depending on its starting point.
  * The effect is rare (1 of 256 crops in one of 17 cells above) and changes the
    embedding of that crop only.
* **JPEG-AI.** On the first 12 crops of two cells, the reconstructions embed to
  within one float16 step of the stored arrays (cosine 1.000000). The difference comes
  from the batch size of the embedding (12 instead of 64).
* **Stage B.** `stageb.py` restricts the Stage-A winners to the six classical codecs.
  The JPEG-AI grid cells carry the arm `grid` as well; selecting them for a flag sweep
  would fail, since no flag sets exist for JPEG-AI.
* **Rate ladder.** The paper describes the JPEG 2000 rate ladder as extended "to
  ratio 2". The code ladder ends at a ratio of 1.94 (`400 * 0.97**175`).
* **`bytes_fixed_q`** of a `cwebp` cell is the size of the fitted file, since
  `cwebp` searches the size itself. `tab:annex-bytes` uses only cells without flags,
  so this does not affect it.

## Third-party components

* **Annex E/F.** The configurations and the manipulation semantics follow
  github.com/dasec/1kB-FaceImage (MIT). They are re-implemented here; no code is
  copied.
* **InsightFace `2d106det`** (buffalo_l pack): the code is MIT and the models are for
  non-commercial research only. It is downloaded at run time.
* **`cwebp`** (libwebp, BSD-3) and **`jpegtran`** (libjpeg-turbo, BSD-style/IJG) are
  system tools; install them with your package manager.
* **JPEG-AI reference software** is fetched by `scripts/setup_jpegai.sh`
  ([baselines.md](baselines.md)).
