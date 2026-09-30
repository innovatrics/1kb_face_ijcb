# Datasets and data preparation (`face1kb.data`, `experiments/prepare/`)

The study evaluates on two face datasets, **Color FERET** (controlled studio
portraits) and **AI-Solutions-KK** (in-the-wild photographs). Neither is distributed
with this repository. The public pipeline starts from **aligned face crops**: square
crops warped onto the ArcFace five-point template at 64, 96, 112, 168 and 224 px.
From those crops, the data layer builds everything the later stages read: the image
index, the verification pairs, the Color FERET ground-truth labels, the estimated
AI-Solutions-KK attributes, the derived crop sets of the ablations, and the dataset
statistics of Section 3 of the paper (arXiv:2608.22866).

- [Installation](#installation)
- [Obtaining the datasets](#obtaining-the-datasets)
- [Directory layout](#directory-layout)
- [Alignment template](#alignment-template)
- [Commands](#commands)
- [Image index and verification pairs](#image-index-and-verification-pairs)
- [Color FERET labels](#color-feret-labels)
- [AI-Solutions-KK attributes](#ai-solutions-kk-attributes)
- [Derived crop sets](#derived-crop-sets)
- [Preprocessing operators](#preprocessing-operators)
- [Dataset statistics and figures](#dataset-statistics-and-figures)
- [Numerical agreement with the paper](#numerical-agreement-with-the-paper)
- [Known deviations from the paper](#known-deviations-from-the-paper)
- [Licences and data-handling rules](#licences-and-data-handling-rules)

## Installation

The index, pairs and labels steps need only the base package; without OpenCV,
`build_index.py` skips its `aligned_112 == aligned_224[::2, ::2]` consistency check
(and says so). Deriving crops (`make_crop_variants.py`, also used by `run.sh` to
derive a missing `aligned_112`) and the other steps use optional dependencies:

```bash
pip install -e ".[eval,preprocess,mst,train]"
# eval:       insightface + onnxruntime-gpu (attributes), matplotlib (figures)
# preprocess: transformers, kornia and einops (BiRefNet), OpenCV (crop variants)
# mst:        skin-tone-classifier (Monk Skin Tone estimate, GPL-3.0)
# train:      datasets (export_release.py)
pip install --no-deps mediapipe==0.10.35 sounddevice==0.5.5   # selfie segmenter (B2)
```

`scripts/setup_env.sh` installs all of these. The `mst` extra installs
`skin-tone-classifier` (`stone`, GPL-3.0), which only the Monk Skin Tone estimate of
`estimate_attributes.py` imports; it is not vendored, and without it
`estimate_attributes.py --no-mst` (and `run.sh`) estimate age and gender only.

## Obtaining the datasets

### AI-Solutions-KK

The raw photographs (105 identities, 17,534 images, MIT licence) are public on
Hugging Face:
[`AI-Solutions-KK/face_recognition_dataset`](https://huggingface.co/datasets/AI-Solutions-KK/face_recognition_dataset).

The paper's results are computed on the **aligned crops** of these photographs. The
five landmarks were predicted by a face detector and landmarker that are not part of
this release, and the crops were warped with bicubic interpolation, so they cannot be
regenerated from the raw photos with this repository. The aligned crops are
**available on request from the authors**: open an issue on the
[repository](https://github.com/innovatrics/1kb_face_ijcb/issues) to ask for them.
They are delivered as a Hugging Face `datasets` dataset with the records `image` (PNG), `identity`, `file_name`
(`<identity folder>/<stem>.png`) and `resolution`; write them to the expected layout
with

```bash
python experiments/prepare/export_release.py --source /path/to/release   # or a hub id
```

If a delivery holds only the 224 px crops, the 112 px (and 56 px) crops are exact
subsamplings of them (see [Alignment template](#alignment-template));
`experiments/prepare/run.sh` derives a missing `aligned_112` automatically.
`build_index.py` warns when an existing `aligned_112` is not `aligned_224[::2, ::2]`
(crops of two different alignments, e.g. an earlier crop release). Replace it by the
subsampled 224 px crops with

```bash
python experiments/prepare/make_crop_variants.py --dataset kk --variants "" \
    --resolutions 112 --overwrite
```

or run `run.sh` with `REDERIVE_112=1` (without `--overwrite`, existing crops are
kept). The 64, 96 and 168 px crops of the resolution study are warped from the source
photographs and cannot be derived exactly from another resolution.

The raw download is still useful for the source-photo statistics (file size and
dimension histograms), see [Dataset statistics](#dataset-statistics-and-figures).

### Color FERET

Color FERET is distributed by NIST
(<https://www.nist.gov/itl/products-and-services/color-feret-database>) under the NIST
licence and cannot be redistributed; obtain it from NIST. In the paper, the 11,338
colour images of 994 subjects were aligned to the ArcFace five-point template at 64,
96, 112, 168 and 224 px (five landmarks per image from a 23-point landmark annotation,
similarity transform fitted at 112 px and scaled by `size / 112`, bilinear
`cv2.warpAffine` with a black border); 3 images without a complete annotation were
dropped, which leaves 11,335 crops. Users align their NIST copy themselves with a
five-point landmarker of their choice. The template math is in
`face1kb.data.alignment` (`ARCFACE_DST_112`, `alignment_matrix`, `warp_to_template`).
Crops must be written as `aligned_<res>/<subject>/<image>.png`, where `<subject>` is
the five-digit NIST subject number and `<image>` the NIST image name (for example
`00001/00001_930831_fa_a.png`; the extension must be lower-case `.png`): the pose
is read from the file name, and the labels are joined on it.

With another landmarker the crops differ from the paper's, and so can the number of
aligned images, the pair universe, the seeded impostor sample and the first-300-row
subsets of some studies. Expect Color FERET numbers that differ slightly from the
paper.

The NIST distribution also carries the ground-truth labels used for the fairness and
difficulty analyses; `prepare_colorferet_labels.py` parses them
([Color FERET labels](#color-feret-labels)).

## Directory layout

All paths come from `face1kb.config` (`FACE1KB_DATA_ROOT`, default `<repo>/data`;
`FACE1KB_OUTPUT_ROOT`, default `<repo>/outputs`; `FACE1KB_MODELS_ROOT`, default
`<repo>/models`):

```
$FACE1KB_DATA_ROOT/
  colorferet/
    aligned_{64,96,112,168,224}/<subject>/<image>.png   # your NIST alignment
    aligned_112_{tight,mid,fill}/...                    # crop-tightness variants
    aligned_112_{A1,A2,A3,A4,B1,B2,C1,C2}/...           # preprocessed crops
    aligned_56/...                                      # 112[::2, ::2]
    variants_templates.json
    index.csv          # id, rel_path, subject, pose
    pairs.parquet      # idx1, idx2, label
    labels.csv         # NIST ground truth (prepare_colorferet_labels.py)
  kk/
    aligned_{64,96,112,168,224}/<identity>/<stem>.png   # on-request release
    aligned_56/...
    index.csv          # id, rel_path, subject
    pairs.parquet
    attributes.csv     # estimated age / gender / Monk Skin Tone
```

AI-Solutions-KK identity folders are named as in the raw download
(`pins_<Name>`), and the file stems are the raw photo stems.

## Alignment template

All crops use the canonical ArcFace template on a 112 px canvas
(`face1kb.data.alignment.ARCFACE_DST_112`; slot order left eye, right eye, nose tip,
left and right mouth corner in image space):

```
(38.2946, 51.6963)  (73.5318, 51.5014)  (56.0252, 71.7366)
(41.5493, 92.3655)  (70.7299, 92.2041)
```

A `size` px crop uses the same framing: with `M_112` the least-squares similarity
(Umeyama, as `skimage.transform.SimilarityTransform`) that maps the five landmarks
onto the template, the crop is `cv2.warpAffine(img, (size / 112) * M_112,
(size, size))`. Because `cv2.warpAffine` samples destination pixel `(u, v)` at
`M^-1 (u, v)`, a crop whose side divides another's by an integer `k` is exactly
`crop[::k, ::k]` of the larger one, for any interpolation mode. For the paper's
crops, `aligned_112 == aligned_224[::2, ::2]` holds bit for bit on all crops of both
datasets, and `aligned_56 == aligned_112[::2, ::2]` on the 3,976 Color FERET 56 px
crops of the ISO annex. The paper's AI-Solutions-KK annex crops were warped
separately and are not subsamplings of the released crops (see
[Known deviations](#known-deviations-from-the-paper)). Other ratios (e.g. 80 px from
224 px) can only be resampled, which approximates a direct warp
(`face1kb.data.alignment.derive_resolution` reports whether a result is exact).

## Commands

`experiments/prepare/run.sh` runs all steps below for both datasets
(`NIST=/path/to/colorferet[.tar]` enables the labels,
`KK_RAW=/path/to/face_recognition_dataset` the source-photo figures,
`REDERIVE_112=1` replaces `aligned_112` by `aligned_224[::2, ::2]`,
`DATASETS="kk"` restricts it to one dataset). A dataset without an `aligned_112` or
`aligned_224` folder is skipped with a notice, so `run.sh` also works with only the
AI-Solutions-KK crops. It skips the index, pairs, labels and attributes files that
exist; the crop-set and preprocessing steps always run and write only missing crops,
so an interrupted run is completed by running it again; the statistics, figures and
the attribute table are regenerated on every run. Every script takes `--help`; outputs
default to the layout above. `build_index.py`, `define_pairs.py`,
`prepare_colorferet_labels.py` and `estimate_attributes.py` refuse to replace an
existing output file without `--overwrite`, and the crop scripts
(`make_crop_variants.py`, `preprocess.py`) keep existing crops unless `--overwrite`
is given (they log how many were kept); `dataset_report.py`,
`cf_attribute_table.py` and the `variants_templates.json` of `make_crop_variants.py`
are rewritten on every run. A script that cannot read one of its input crops exits
with an error.

| Step | Command | Output | Cost |
|---|---|---|---|
| Index | `build_index.py --dataset {colorferet,kk} [--check-res 64,96,112,168,224]` | `index.csv` | seconds |
| Pairs | `define_pairs.py --dataset {colorferet,kk} [--sample]` | `pairs.parquet` (+ `pairs_sample.csv`) | < 1 min, ~3 GB RAM (KK) |
| CF labels | `prepare_colorferet_labels.py --nist <folder or tar> [--format xml\|name_value]` | `labels.csv` | ~0.5 min |
| KK attributes | `estimate_attributes.py --dataset kk [--no-mst]` | `attributes.csv` | ~1 min CPU |
| Crop variants | `make_crop_variants.py --dataset colorferet [--resolutions 56]` | `aligned_112_{tight,mid,fill}`, `aligned_56` | ~1 min CPU |
| Preprocessing | `preprocess.py --dataset colorferet [--ops A1,...] [--limit N]` | `aligned_112_<op>` | A1-A4 minutes CPU; B1/C1/C2 ~25 GPU min (RTX 2080 Ti) |
| Statistics | `dataset_report.py --dataset {kk,colorferet} [--raw-dir ...] [--examples]` | `outputs/dataset_stats/`, `outputs/figures/ds_*.png` | seconds |
| CF table | `cf_attribute_table.py [--paper-header]` | `outputs/tables/cf_attributes.tex` | < 1 s |

All scripts are under `experiments/prepare/`; the library code is in
`face1kb/data/`.

## Image index and verification pairs

`index.csv` fixes the row order used by every later stage: embedding row `i` is
image `id == i`, and the pairs refer to these ids. It is a directory scan of
`aligned_112` sorted by `(subject, stem)` as strings, with the columns `id`,
`rel_path` (`<subject>/<stem>.png`, relative to any `aligned_<res>` folder),
`subject` (the folder name) and, for Color FERET, `pose` (the pose description of
the file-name pose code: `fa` frontal image, `fb` frontal image taken shortly after
the previous image, `hl`/`hr` half left/right, `pl`/`pr` profile left/right, `ql`/`qr`
quarter left/right, `ra`/`rb`/`rc`/`rd`/`re` head turned about 45/15 degree left and
15/45/75 degree right). For the paper's crops this order is the paper's index order
exactly. `face1kb.data.read_index_csv` and `face1kb.config.read_index` both read
`subject` as a string (`00001`, as in `labels.csv`), so indices and labels join on
`subject` directly. Crop files must use the lower-case `.png` extension;
`build_index.py` stops with an error on other spellings.

`pairs.parquet` holds every unordered image pair once (`idx1 < idx2`, row-major, the
order of `numpy.triu_indices`), with `idx1`, `idx2` (int32) and `label` (int8,
1 = same subject), zstd-compressed. For the paper's crop sets this is 64,235,445
pairs (95,839 mated) on Color FERET and 153,711,811 pairs (1,515,807 mated) on
AI-Solutions-KK. The accuracy stage scores all mated pairs and a seeded sample of
the non-mated ones, so with the same index it draws the same impostor pairs.

Per-image tables (labels, attributes, results) are joined on `rel_path` with
`face1kb.data.align_to_index`, which checks that every index row is matched exactly
once.

## Color FERET labels

`prepare_colorferet_labels.py --nist <path>` reads the NIST ground truth either from
an extracted copy (a `dvd1/` or `dvd2/` folder, or a folder with `dvd1/` and `dvd2/`
at most three levels below it) or directly from the distribution tar archive, in
either of its two equivalent formats:

```
dvd{1,2}/data/ground_truths/xml/<subject>/<subject>.xml   # Gender, YOB, Race
dvd{1,2}/data/ground_truths/xml/<subject>/<image>.xml     # CaptureDate, Pose, Wearing, Hair
dvd{1,2}/data/ground_truths/name_value/<subject>/<...>.txt  # the same as key=value text
```

It writes one row per recording (11,338 rows, 994 subjects), sorted by
`(subject, image)`:

| Column | Content |
|---|---|
| `subject`, `image`, `rel_path` | NIST subject number (`00001`), image name, `<subject>/<image>.png` |
| `pose_code`, `pose` | NIST pose code and its description (as in the index) |
| `gender` | `male` / `female` |
| `race` | NIST race merged into 7 classes: `Black-or-African-American` -> `Black`; `Asian-Middle-Eastern` and `Asian-Southern` -> `Asian`; `White`, `Hispanic`, `Pacific-Islander`, `Native-American`, `Other` unchanged |
| `race_nist` | the 9 NIST classes |
| `glasses` | `yes` / `no` |
| `beard`, `mustache` | `1` = no, `2` = yes |
| `age_from`, `age_to` | age at capture, capture year minus year of birth (both columns equal) |
| `year_of_birth`, `capture_date` | as given by NIST |
| `yaw`, `pitch`, `roll` | NIST nominal pose angles of the pose code (degrees; yaw 0 for `fa`/`fb`, +-22.5 `ql`/`qr`, +-67.5 `hl`/`hr`, +-90 `pl`/`pr`, 45/15/-15/-45/-75 for `ra`-`re`; pitch and roll are 0) |

The eye/nose/mouth coordinates of the ground truth are not read. The fairness
analysis uses `pose`, `gender`, `race` and the age; the difficulty analysis also
`yaw`, `pitch`, `glasses`, `beard` and `mustache`. `cf_attribute_table.py` renders
the gender, race, glasses and pose distribution (Table `tab:cf-attributes`).

## AI-Solutions-KK attributes

AI-Solutions-KK has no demographic ground truth; `estimate_attributes.py` estimates

- **age and gender** with the insightface `buffalo_l` `genderage` model on the
  aligned 112 px crops (the crop is already aligned, so the ArcFace template serves as
  its five key points and no detector runs), for the first 8 crops of each identity in
  index order. The model runs on the CPU by default (`--ctx-id -1`, a few seconds),
  which reproduces the paper's genderage outputs bit for bit given the same crops
  (840/840 ages and genders on the 112 px crops the paper's estimates were made
  from; the released 112 px crops differ from those, see
  [Known deviations](#known-deviations-from-the-paper)); `--ctx-id N` runs it on CUDA device
  `N` among the visible ones (`CUDA_VISIBLE_DEVICES`), whose results differ in the
  last float bits and can move an age that sits on a rounding boundary by one year;
- **Monk Skin Tone** (MST1-MST10) with the `stone` package (`skin-tone-classifier`)
  and the 10-tone Monk palette on the aligned 224 px crop of the first image of each
  identity.

`attributes.csv` has one row per processed crop: `subject, image, rel_path, age,
gender (M/F), mst_label, mst_index, skin_hex` (840 rows). The fairness analysis
summarises them per identity: the MST label, the gender mode and the band of the
median age. `stone` clusters skin colours with OpenCV k-means seeded from OpenCV's
global random generator, so an estimate depends on the sequence of `stone` calls in
the process; run the script over the whole dataset (identities in sorted order) to
reproduce the paper's sequence. Without `stone` (`--no-mst`) only age and gender are
estimated. `--dir-112` / `--dir-224` select other crop folders. If a crop cannot be
read, the script exits with an error and writes nothing.

The model files are downloaded on first use into `FACE1KB_MODELS_ROOT` and verified
by SHA-256: `insightface/buffalo_l/` (from the insightface v0.7 release; every model
of the pack is pinned, and `face1kb.data.fetch.insightface_model_path(name)` returns a
verified model such as `2d106det.onnx`, re-extracting a missing or corrupt one) and
`mediapipe/selfie_segmenter.tflite`; BiRefNet goes to the Hugging Face cache
`FACE1KB_MODELS_ROOT/hf` at a pinned revision.

## Derived crop sets

**Crop-tightness variants** (Section 8, alignment tightness): the `tight`, `mid` and
`fill` templates are the ArcFace template scaled to an inter-ocular distance of 0.40,
0.46 and 0.52 of the crop width, with the eye line at 0.34, 0.27 and 0.20 of the crop
height (the standard template has 0.315). Each is a similarity image of the standard
template, and least-squares similarity fitting is equivariant under it, so a variant
crop is a fixed sub-window of the standard frame. `make_crop_variants.py` renders it
at 112 px from the standard 224 px crop (bicubic) without landmarks and writes
`variants_templates.json` with the template coordinates and the sub-window. The paper
warped the variants directly from the source photographs, so the rendered crops
differ by the second interpolation (mean absolute difference 1.56/255, per-image
PSNR 38.5-39.1 dB on Color FERET). The clean EERs (%) of the crop-tightness table
(Color FERET, 112 px, all mated pairs and 2M seeded non-mated pairs) move by at most
0.33 points:

| Variant | ArcFace | LVFace-L | TopoFR-R100 | EdgeFace-XS |
|---|---|---|---|---|
| tight: paper / rendered from 224 px | 0.25 / 0.26 | 0.58 / 0.53 | 0.20 / 0.19 | 4.73 / 4.72 |
| mid | 1.55 / 1.55 | 1.95 / 1.85 | 0.99 / 0.99 | 12.85 / 12.76 |
| fill | 5.60 / 5.64 | 10.44 / 10.11 | 3.88 / 3.89 | 26.71 / 26.57 |

**Other resolutions**: `--resolutions 56` writes `aligned_56` (exact subsampling of
`aligned_112`; identical to the paper's Color FERET annex crops). A non-divisor
resolution such as 80 px is resampled from the 224 px crop and flagged as
approximate (median PSNR 37.9 dB against a direct warp on Color FERET).

## Preprocessing operators

`face1kb.data.preprocess` holds the pre-compression operators of the preprocessing
studies, shared by the clean-crop ablation (`preprocess.py`, Color FERET 112 px) and
the preprocessing-through-the-codec study (in-memory on AI-Solutions-KK 224 px crops):

| Op | Operation |
|---|---|
| A1 | edge-preserving smoothing, `cv2.edgePreservingFilter(RECURS_FILTER, sigma_s=30, sigma_r=0.25)` |
| A2 | bilateral filter, `d=7`, `sigma_color=sigma_space=75` |
| A3 | chroma de-emphasis, Gaussian blur (sigma 1.5) of the Cr/Cb planes |
| A4 | non-local means, `cv2.fastNlMeansDenoisingColored(h=8, hColor=8, 7, 21)` |
| B1 | background flattened to grey 128 with a BiRefNet foreground mask |
| B2 | background flattened to grey 128 with the MediaPipe selfie-segmenter mask |
| C1 | B1 followed by A1 |
| C2 | B1, then a bilateral filter (`d=9`, sigma 100) outside a soft elliptical mask over the eye-nose-mouth region |

BiRefNet (`ZhengPeng7/BiRefNet`, MIT) is loaded through `transformers` with
`trust_remote_code=True` at revision `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4` and
run in fp32 at 512 x 512 (its remote code also needs `kornia` and `einops`). The
MediaPipe selfie segmenter (Apache-2.0) runs on the crop resized to 256 x 256; install
MediaPipe with `pip install --no-deps mediapipe==0.10.35 sounddevice==0.5.5` (see
`scripts/setup_env.sh`).
`apply_operator(img, op)` applies one operator to an RGB (or, with `bgr=True`, BGR)
crop.

## Dataset statistics and figures

`dataset_report.py --dataset kk` writes `outputs/dataset_stats/kk_summary.json`
(images, identities, images per identity, pair counts and, with `--raw-dir` or
`--raw-stats`, the image and identity counts of the raw download) and the Section 3
figures into `outputs/figures/`: `ds_kk_imgs.png` (images per identity), `ds_kk_age.png`,
`ds_kk_gender.png`, `ds_kk_mst.png` (from `attributes.csv`) and, with `--raw-dir`
pointing at the raw download (or `--raw-stats` at a statistics JSON saved with
`--save-raw-stats`), `ds_kk_filesize.png` and `ds_kk_dims.png` (width/height of a
seeded sample of 20 photos per identity). `--examples` adds `ds_kk_examples.png`, a
montage of aligned-112 crops of 10 seeded identities. `--dataset colorferet` writes
the summary (with the pose distribution of the aligned crops) and, with
`--examples`, a montage of your aligned crops.

## Numerical agreement with the paper

Checked against the paper's data files with the paper's crops:

- `index.csv`: row-for-row identical to the paper's index for both datasets (Color
  FERET: identical CSV up to the zero padding of `subject`; AI-Solutions-KK: the
  paper's index lists `.jpg` names for the `.png` crops).
- `pairs.parquet`: byte-identical files for both datasets; `pairs_sample.csv`
  byte-identical (Color FERET) or identical up to the file extension (KK).
- `labels.csv` from the NIST tar, the extracted XML or the name/value files
  (all three byte-identical): 11,338/11,338 rows agree with the paper's labels in
  subject, image, pose, gender, race, glasses, beard, mustache and age; the attribute
  table is byte-identical to the paper's `cf_attributes.tex` (with `--paper-header`;
  without it only the comment line differs). Pose angles: see below.
- `attributes.csv`: run (on the CPU, the default) on the crops the paper's estimates
  were made from, all 840 ages and genders and all 105 MST labels and skin colours
  agree; apart from the column names (`subject` for `identity`, the added
  `rel_path`) the file is the paper's attribute table byte for byte. The MST
  histogram is the paper's (MST6 45, MST7 30, MST5 12, MST9 7, MST10 5, MST8 4,
  MST1 1, MST2 1). With `--ctx-id 0` (CUDA provider) one age differs by a year: its
  raw output is 36.5000305 on the CPU and 36.4999962 on the GPU.
- Preprocessing: A1-A4 (first 500 Color FERET crops) and B2 (first 50) are
  pixel-identical to the paper's crop sets; B1, C1 and C2 (BiRefNet on the GPU, first
  400) are pixel-identical for 398/400 crops, the other two differ in a single pixel
  by one grey level (GPU floating-point non-determinism); the operators applied to
  AI-Solutions-KK 224 px crops are identical to the through-the-codec study's (50
  crops).
- `aligned_56` equals the paper's 56 px crops on all 3,976 Color FERET crops of the
  ISO annex subset (for AI-Solutions-KK see the deviations below).
- Figures: `ds_kk_imgs`, `ds_kk_filesize`, `ds_kk_age`, `ds_kk_gender`, `ds_kk_mst`
  and `ds_kk_dims` (from the paper's statistics JSON) are byte-identical to the
  paper's PNGs with the paper's environment (matplotlib 3.11).

## Known deviations from the paper

- **Color FERET pose angles.** The paper's Color FERET labels carried per-image
  head-pose estimates in `yaw`/`pitch`/`roll`, which the NIST ground truth does not
  provide; `labels.csv` has the NIST nominal angles of the pose code (they correlate
  0.975 with the paper's yaw, but e.g. the half-profile images are at 67.5 degrees
  nominal against 35-38 degrees estimated on average, and pitch and roll are 0).
  Analyses of `|yaw|` or pitch (the sample-difficulty study) therefore give different
  numbers. All other label columns match.
- **AI-Solutions-KK age and gender.** The paper's estimates were computed on an
  earlier set of 112 px crops with a tighter framing than the released crops (the
  224 px crops of the two sets differ only in a few pixels: mean absolute difference
  0.0006 grey levels, maximum 11). On the released crops, 267/840 per-image ages and
  808/840 genders are identical (age MAE 1.7 years); per identity, the MST label
  agrees for 105/105, the gender mode for 104/105 and the median-age band for 95/105
  identities, so KK fairness subgroups by age can shift slightly.
- **Crop-tightness variants** are rendered from the 224 px crops instead of being
  warped from the source photographs (see [Derived crop sets](#derived-crop-sets)).
- **AI-Solutions-KK annex crops.** The paper's AI-Solutions-KK crops of the ISO annex
  (56 px and its 112/224 px columns) were warped separately from the released crops
  and differ from them (56 px: median PSNR about 41.6 dB against
  `aligned_112[::2, ::2]`), so the KK annex columns cannot be reproduced bit for bit
  from `aligned_56`. The 80 px annex crops cannot be derived exactly on either
  dataset (median 37.9 dB on Color FERET).
- **Source-dimension histogram.** `ds_kk_dims.png` samples 20 photos per identity from
  sorted file listings with a dedicated seeded generator; the paper's figure sampled
  in file-system order, so a fresh run from the raw download gives a slightly
  different histogram (same medians, 175 x 184 px). The file-size histogram and the
  per-identity counts are identical.
- **Example montages.** The paper's Color FERET panel shows source captures of the
  distribution; the public script builds montages of aligned crops only.
- **Availability wording.** The paper states that the aligned AI-Solutions-KK crops
  are freely downloadable; they are available on request from the authors.

## Licences and data-handling rules

- Color FERET: NIST licence, not redistributable. AI-Solutions-KK: MIT (raw photos).
- insightface pretrained models (`buffalo_l`): non-commercial research use only.
- `skin-tone-classifier` (`stone`): GPL-3.0; an optional dependency that is imported
  only for the MST estimate and is not distributed with face1kb.
- BiRefNet: MIT; MediaPipe selfie segmenter: Apache-2.0.
- Everything the data layer writes under `FACE1KB_DATA_ROOT` (crops, `index.csv`,
  `pairs.parquet`, `labels.csv`, `attributes.csv`) and the example montages are
  per-image data derived from the datasets: keep them local (the repository's
  `.gitignore` excludes `data/`, `work/`, `outputs/` and `models/`). The summary JSON
  files, histograms and the attribute table are aggregates.
