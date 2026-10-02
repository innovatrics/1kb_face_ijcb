# Side studies: recompression, preprocessing, resolution, sample difficulty

This page covers four experiment areas of the paper (arXiv 2608.22866):

| Area | Paper | Scripts |
|---|---|---|
| `experiments/recompression/` | Section "Compressed-on-Compressed Recompression" (Tables 66-69, Figure 42) | `recompress.py`, `make_tables.py`, `run.sh` (Figure 42: `experiments/figures/render_figures.py`) |
| `experiments/preprocessing/` | Section "Ablations and Preprocessing": preprocessing through the codec (Tables 47-48, Figure 35), clean-crop preprocessing and crop-tightness ablations (Tables 49-50, Figures 36-38) | `through_codec.py`, `ablation.py`, `make_tables.py`, `impact_figure.py`, `montages.py`, `run.sh` |
| `experiments/resolution/` | Section "Ablations and Preprocessing": resolution-information trade-off (Table 46, Figure 34) | `analyze.py`, `render.py`, `run.sh` |
| `experiments/difficulty/` | Section "Trivial and Difficult Samples" (Tables 44-45, Figures 32-33) | `compute.py`, `analyze.py`, `make_tables.py`, `run.sh` |

Every area has a compute step, which needs the aligned crops (and usually a GPU), and a
render step, which needs only the aggregate CSVs. The aggregates of the paper ship in
`results/` (see [results.md](results.md)), so all tables of these sections can be
rendered on a CPU without the datasets:

```bash
python experiments/recompression/make_tables.py --from-results --paper-header
python experiments/preprocessing/make_tables.py --from-results --paper-header
python experiments/resolution/render.py --from-results
python experiments/difficulty/make_tables.py --from-results --paper-header
```

With `--paper-header` the first comment line of each table is the one of the published
LaTeX source; the tables of the recompression, preprocessing and difficulty areas are
then byte-identical to the paper. Without it the comment names the generating script.
Tables go to `FACE1KB_OUTPUT_ROOT/tables/`, figures to `FACE1KB_OUTPUT_ROOT/figures/`.

All paths come from `face1kb.config` (`FACE1KB_DATA_ROOT`, `FACE1KB_WORK_ROOT`,
`FACE1KB_OUTPUT_ROOT`, `FACE1KB_MODELS_ROOT`, `FACE1KB_WEIGHTS_DIR`) or from command-line
flags; see [datasets.md](datasets.md) for the data layout. Each `run.sh` reproduces its
section at full scale and lists its CPU/GPU cost in its header. The costs quoted below
were measured on 8 CPU cores and one RTX 2080 Ti.

What stays local: the figures of these areas that show dataset faces
(`alignment_check.png`, `alignment_variants.png`, `preprocessing_impact.png`,
`preproc_through_codec_kk.png`, `difficulty_montage.png`), the per-image difficulty
CSV, the one-subject `ablation/preprocessing_impact.csv`, the recompressed files and
manifests, and the decoded-crop caches. None of them is redistributable.

## Recompression

A face is compressed once and then compressed again: aligned crop -> source codec at the
budget -> decode -> second codec at the same budget -> decode. The matrix spans the six
classical codecs (JPEG, JPEG 2000, WebP, JPEG XL, AVIF, HEIF), JPEG-FzT and
Ours-ACCURATE (8 x 8), at 112 px and 1024 / 512 B.

```bash
# one dataset and budget, all three stages
python experiments/recompression/recompress.py --dataset colorferet --budget 1024
python experiments/recompression/recompress.py --dataset kk --budget 1024 --limit 2500
# ArcFace rescore (writes recompression_colorferet_arcface_antelopev2.csv)
python experiments/recompression/recompress.py --dataset colorferet --budget 1024 \
    --model arcface_antelopev2 --stages score,ours
python experiments/recompression/make_tables.py
```

Stages (`--stages`, default all three):

* `produce` (CPU pool, `--workers`): writes the 7 x 7 classical / JPEG-FzT block to
  `WORK_ROOT/<ds>/recompressed/<src>__<dst>_<res>_<budget>/<subject>/<stem>.<ext>`
  (`--recompressed-root`), plus per-cell manifests. Existing files are skipped. Both
  passes use the benchmark's binary search (quality 2..94 step 2, JPEG 2000 ratio
  400..12 step 4); JPEG-FzT searches the even qualities 2..94.
* `score` (GPU): embeds every cell and the single-pass reference of the second codec
  (the benchmark's `compressed/112px_<budget>B/<codec>` files) with the matcher
  (`--model`, default `edgeface_xs`) in index blocks of 256 crops, and computes the EER
  over all mated pairs plus 5,000,000 seeded non-mated pairs (`--sample-nonmated`,
  `--seed`). Crops missing from a cell are dropped from its trials.
* `ours` (GPU): the 15 cells with Ours-ACCURATE as source and/or second codec. Because
  every crop needs face1kb-ACCURATE encodes, these cells use a deterministic sample:
  subjects in name order with a stride, up to 10 crops per subject, 120 crops in all
  (`--ours-n`, `--ours-max-per-subject`); trials are the pairs with both members in the
  sample (Color FERET: 15 subjects, 456 mated and 6,684 non-mated pairs). The
  single-pass references of these rows are computed on the same sample, for
  Ours-ACCURATE from the stored benchmark `.bin` files when present. The codec runs
  with `paper_compat=True`.

Output: `OUTPUT_ROOT/recompression/recompression_<ds>_<budget>.csv` with the columns
`dataset, res, budget, source_codec, second_codec, eer, delta_eer_vs_singlepass`
(another `--model` writes `recompression_<ds>_<model>.csv` for all budgets). Rows are
merged into an existing file; cells already present are skipped unless `--rescore`.

`make_tables.py` writes `recompression_eer_<ds>_<budget>.tex`: chain EER (%), per column
the best and worst classical source shaded on the printed value (ties all shaded, an
all-tied column unshaded). The Ours-ACCURATE row and column are not shaded, because the
default matcher belongs to the EdgeFace family of the codec's identity anchor.

Figure 42 (`recompression_heatmap.png`) is rendered from the same CSVs by
`experiments/figures/render_figures.py`:

```bash
python experiments/figures/render_figures.py --only recompression_heatmap          # OUTPUT_ROOT
python experiments/figures/render_figures.py --only recompression_heatmap --from-results
```

Cost: produce ~20 h CPU for Color FERET (both budgets, 11,335 crops) and ~4-5 h for the
AI-Solutions-KK subset; score ~1-2 h per dataset and budget (mostly decoding); ours
~25 min per dataset and budget on one GPU.

Differences between the paper text and the code:

* The AI-Solutions-KK classical block uses the first 2,500 crops of the index
  (`--limit 2500`), while its single-pass references use all 17,534 crops.
* The Ours-ACCURATE rows and columns are scored on the 120-crop sample described
  above, not on the full image set (the paper says the full matrix was run), and their
  delta-EER is relative to a single-pass reference on that sample.
* The decoded first pass keeps its metadata, as in the paper runs. A decoded JPEG XL
  image carries a 536-byte ICC profile that the AVIF and HEIF encoders write into their
  file, leaving about half of a 1024-byte budget for the image: on the first 20 Color
  FERET crops the mean second-pass quality drops from 55 to 6 (AVIF) and from 42 to 7
  (HEIF) compared with an encode without the profile. This is
  what makes the JPEG XL -> AVIF and JPEG XL -> HEIF chains fail; the paper attributes
  it to an interaction of the codecs' artefacts. `--strip-metadata` encodes the second
  pass without metadata (not the paper setting; write it to another
  `--recompressed-root` and `--out-dir`).

## Preprocessing through the codec

Operators applied to the aligned AI-Solutions-KK 224 px crop, then compressed to 1024 B,
decoded, resized to 112 px (bilinear) and embedded. Reported per (operator, codec,
matcher): EER over the pairs with both members in the sample, and the mean identity
cosine of the reconstruction to the unprocessed crop.

```bash
python experiments/preprocessing/through_codec.py --phase encode --codecs webp,avif
python experiments/preprocessing/through_codec.py --phase encode --codecs ours_accurate
python experiments/preprocessing/through_codec.py --phase aggregate
python experiments/preprocessing/make_tables.py
```

* Sample: `numpy.random.default_rng(0).choice` of 800 index rows without replacement
  (`--n`, `--seed`). The paper calls it subject-stratified; it is a uniform draw.
* Operators (`--operators`, from `face1kb.data.preprocess`): `std`, `A1` edge-preserving,
  `A2` bilateral, `A3` chroma reduction, `A4` non-local means, `B2` MediaPipe background
  flattening; `B1`, `C1`, `C2` (BiRefNet) are supported too but were not in the paper
  run.
* Codecs (`--codecs`): WebP and AVIF (binary search), JPEG-FzT, Ours-ACCURATE
  (`paper_compat=True`; an identity-only container would give a mid-grey frame).
* Matchers (`--models`): `arcface_antelopev2`, `edgeface_xs`.
* `--phase encode` caches the decoded 112 px crops per unit
  (`WORK_ROOT/kk/preproc_through_codec/`, `--cache-dir`) and resumes; `--phase aggregate`
  writes `OUTPUT_ROOT/accuracy/preproc_through_codec_kk.csv` (columns `operator, codec,
  res, budget, model, eer, id_cos_vs_original, delta_eer_vs_std, n_images`).
* `--montage-only --montage-samples 'subject/stem:label,...'` renders
  `preproc_through_codec_kk.png` (operators on the given crops, decoded with the codec of
  lowest mean ArcFace EER). The paper figure used three subjects spanning skin tones.

`make_tables.py` writes `preproc_through_codec.tex` (ArcFace) and
`preproc_through_codec_edgeface.tex` (EdgeFace-XS). Cost: WebP ~1 min and AVIF ~5 min
per operator on the CPU, Ours-ACCURATE ~8 min per operator on one GPU, aggregate a few
minutes.

## Clean-crop ablations (crop tightness, preprocessing)

`ablation.py` scores the uncompressed Color FERET 112 px crop sets from their
embeddings on one trial set (all mated pairs plus 2,000,000 non-mated pairs, seed 0) and
writes `OUTPUT_ROOT/ablation/preprocessing_ablation.csv` (`study, variant, suffix,
model, eer, id_cos_vs_std, n_pos, n_neg`) for the four anchor matchers:

* crop tightness: `standard` (the benchmark crop, IOD/W about 0.31) and the presets
  `tight`, `mid`, `fill` (IOD/W 0.40 / 0.46 / 0.52);
* preprocessing: `std` and `A1`-`A4`, `B1`, `B2`, `C1`, `C2`, with the mean identity
  cosine of each variant to the `std` crop.

The crop sets are made by `experiments/prepare/run.sh` (steps 4 and 5) and embedded with
`experiments/embed/compute_embeddings.py --datasets colorferet --models anchor --kind
aligned --resolutions 112 --suffixes ,_tight,_mid,_fill,_A1,_A2,_A3,_A4,_B1,_B2,_C1,_C2`.
`make_tables.py` renders `ablation_croptightness.tex` and `ablation_preprocessing.tex`.
Cost: CPU, about 10 minutes (13 crop sets x 4 matchers).

Figures (dataset faces, keep them local):

* `impact_figure.py`: `preprocessing_impact.png` and `ablation/preprocessing_impact.csv`
  -- each preprocessing variant WebP-encoded with a descending quality scan (94, 92, ...;
  first file within 1024 B) and scored with PSNR, SSIM and LPIPS against its own
  uncompressed crop, averaged over the first four index crops (all of one subject, so
  the CSV stays local too). `--figure-crop` selects
  the crop shown; the published figure shows a profile crop of another subject than the
  table sample. The figure labels are WebP qualities (the paper caption says JPEG quality
  factor).
* `montages.py`: `alignment_check.png` (the fixed ArcFace 5-point template drawn over
  crops of six subjects, three poses and the five resolutions) and
  `alignment_variants.png` (the four crop presets). The dots are the template, not
  landmarks detected on the crops (the paper caption says detected interest points).

Differences to the paper: the paper's crop presets were warped from landmarks of a
proprietary detector. The public presets are sub-windows of the standard ArcFace frame
rendered from the 224 px crops (`face1kb.data.alignment.render_variant`), about 1.6 grey
levels mean absolute difference from the paper crops, so the crop-tightness EERs of a
public run move slightly: on Color FERET the public presets gave tight 0.26 / 0.53 /
0.19 / 4.72 %, mid 1.55 / 1.85 / 0.99 / 12.76 % and fill 5.64 / 10.11 / 3.89 / 26.57 %
(ArcFace / LVFace-L / TopoFR-R100 / EdgeFace-XS) against 0.25 / 0.58 / 0.20 / 4.73,
1.55 / 1.95 / 0.99 / 12.85 and 5.60 / 10.44 / 3.88 / 26.71 % in the paper, with the same
ordering and shading. The shipped `ablation/preprocessing_ablation.csv` holds the paper
values.

## Resolution-information analysis

```bash
python experiments/resolution/analyze.py            # CPU
python experiments/resolution/render.py
```

`analyze.py` writes to `OUTPUT_ROOT/resolution_information/`:

* `embedding_decomposition.csv`: per matcher, the median over images of the embedding
  cosine between the clean 112 px crop and the clean crop at each resolution
  (`resize_cos`); with `--codecs` also the compressed sources `<codec>_<res>_<budget>`
  (`compress_cos` against the clean crop at the same resolution, `net_cos` against the
  clean 112 px crop);
* `spectral_retention.csv`: the fraction of AC spectral energy of 400 seeded clean
  224 px crops (grey, mean removed, radially averaged power spectrum) that lies below
  each resolution's Nyquist frequency;
* `eer_by_resolution.csv`: clean EER per matcher and resolution over all mated pairs plus
  3,000,000 seeded non-mated pairs.

Matchers: `--models` (default `roster`: the public matchers with an embeddings folder).
Only the tags `aligned_<res>` (and, with `--codecs`, `<codec>_<res>_<budget>`) are read;
crop-variant arrays and stray files are ignored. `render.py` writes `res_decomp.tex`
(AI-Solutions-KK, four anchors; `--dataset`) and `resolution_summary.png`. Cost: CPU,
30-60 minutes for all matchers of both datasets (embedding loads and 140 EERs).

Differences to the paper: the paper typed Table 46 by hand from these CSVs; its
EdgeFace-XS 96 px EER reads 1.45 % where the CSV gives 1.444 % (printed 1.44 by
`render.py`). The paper's CSV also carried a duplicate row from a temporary file, which
the public code never reads. The same paper run read the 112 px TopoFR-R200 array of
AI-Solutions-KK while it was still being written, so the shipped
`embedding_decomposition.csv` has an empty (NaN) `resize_cos` for TopoFR-R200 at 64 px on
AI-Solutions-KK; a fresh `analyze.py` run gives 0.954 there. `render.py` does not use
that row (TopoFR-R200 is not one of the table's anchors). The published Figure 34 has one legend inside the first
and one inside the third panel. `render.py` instead draws a single legend panel on the
right, gives each dataset/matcher the same colour in both panels, and lists every
plotted series. The curves are the same as in the paper figure.

## Sample difficulty

```bash
python experiments/difficulty/compute.py            # GPU
python experiments/difficulty/analyze.py
python experiments/difficulty/make_tables.py
python experiments/difficulty/compute.py --montage-only
```

`compute.py` draws `random.Random(0).sample` of 300 crops per dataset (`--n`, `--seed`)
from the sorted 112 px crop files, compresses each at 1024 and 512 B with the six
classical codecs, JPEG-AI and both face1kb variants, and scores every reconstruction
against the clean crop: identity cosine with `--id-model`, PSNR, SSIM
(`pytorch_msssim`), LPIPS (AlexNet), and the size. It also records eight
codec-independent descriptors of the clean crop and the dataset attributes (Color FERET
`labels.csv`; AI-Solutions-KK `attributes.csv`). Codec settings of this study:

* classical codecs: a linear scan of quality 2..94 in steps of 4 (JPEG 2000 ratios
  400..12 in steps of 12), keeping the largest file within the budget
  (`--quality-step`); settings that fail to encode are skipped;
* JPEG-AI: the analytic bpp target with up to three downward corrections and no
  plausibility floor (`--no-jpegai` skips it);
* face1kb: `paper_compat=True`.

The output `OUTPUT_ROOT/difficulty/sample_difficulty.csv` is per-image and holds dataset
attributes: keep it local. `analyze.py` reduces it to the aggregates
`difficulty_correlations.csv` (image-level Spearman correlation of difficulty,
`1 - id_cos` averaged over the codecs at 512 B, with each descriptor and attribute),
`cross_codec_sharing.txt` (mean / range of the pairwise Spearman correlation of
per-image difficulty between codecs) and `difficulty_contrast.csv` (Color FERET
easiest vs. hardest image deciles). `make_tables.py` renders
`difficulty_predictors.tex`, `difficulty_contrast.tex` and `difficulty_predictors.png`
from them; `compute.py --montage-only` re-encodes the five easiest and five hardest
Color FERET crops for WebP and Ours-ACCURATE (`difficulty_montage.png`, local).

Cost: `compute.py` about 20 s per crop and dataset without JPEG-AI on one GPU (about
3-4 h for 2 x 300 crops), more with JPEG-AI; the other steps take seconds.

Differences to the paper and reproducibility:

* The paper's `id_cos` came from a proprietary matcher that is not distributed. The
  default `--id-model` is the public held-out matcher `cvlface_ir101`; any matcher can be
  plugged in with `face1kb.fr.register_onnx` / `FACE1KB_FR_PLUGINS`
  ([models.md](models.md)). A public run therefore gives different difficulty values and
  correlations; the shipped aggregates are the paper's.
* The matcher receives `uint8` crops, so the float reconstructions of the face1kb codecs
  are rounded before embedding (the paper embedded them unrounded; with the same matcher
  this changes their `id_cos` by up to about 2e-3). All other per-image values
  reproduce: with the paper's matcher registered as a plugin, 20 crops per dataset gave
  identical sizes for all codecs and identical descriptors, and PSNR / SSIM / LPIPS
  within 1e-6.
* The Color FERET `|yaw|` and pitch of the paper were per-image head-pose estimates.
  The public `labels.csv` (NIST ground truth, [datasets.md](datasets.md)) carries the
  nominal angles of the pose codes, so the `|yaw|` rows of Tables 44 and 45 cannot be
  regenerated publicly.
* The AI-Solutions-KK attributes cover up to eight crops per subject for age and gender
  and one per subject for the Monk skin tone. Of the 300 sampled crops, 12 have an age
  (the paper calls these "age-bucket means"; they are 12 images) and none has a skin
  tone, so the skin-tone row prints `--`.
