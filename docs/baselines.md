# Baseline codecs

The benchmark compares the two face1kb codecs with ten baseline codecs. Every
baseline is driven to the **largest setting whose output fits a byte budget**
(1024 B and 512 B in the paper; 768 B and 960 B in the budget sweep). The baselines
live in `face1kb.baselines`; the face1kb codecs are documented in
[codec.md](codec.md).

- [The ten codecs](#the-ten-codecs)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Budget search](#budget-search)
- [Classical codecs](#classical-codecs)
- [JPEG-FzT](#jpeg-fzt)
- [JPEG-AI](#jpeg-ai)
- [CompressAI: bmshj2018 and mbt2018](#compressai-bmshj2018-and-mbt2018)
- [Decoding stored files](#decoding-stored-files)
- [Checking an installation](#checking-an-installation)
- [Cost](#cost)
- [Reproducibility limits](#reproducibility-limits)
- [Differences between the paper text and the code](#differences-between-the-paper-text-and-the-code)
- [Licences](#licences)

## The ten codecs

| Name (result files) | Paper label | Backend (paper version) | Rate knob and grid | File | Device |
|---|---|---|---|---|---|
| `jpeg` | JPEG | Pillow 12.2.0 / libjpeg-turbo | quality 2, 4, ..., 94; `optimize=True`, 4:2:0 | `.jpg` | CPU |
| `jpeg2000` | JPEG 2000 | Pillow 12.2.0 / OpenJPEG | compression ratio 400, 396, ..., 12 (`quality_mode="rates"`, one layer) | `.jp2` | CPU |
| `webp` | WebP | Pillow 12.2.0 / libwebp | quality 2..94 (step 2), `method=6` | `.webp` | CPU |
| `jpeg_xl` | JPEG XL | pillow-jxl-plugin 1.3.7 / libjxl | quality 2..94 (step 2), default effort | `.jxl` | CPU |
| `avif` | AVIF | Pillow 12.2.0 native AVIF (libavif, aom) | quality 2..94 (step 2), default speed | `.avif` | CPU |
| `heif` | HEIF | pillow-heif 1.4.0 / libheif + x265 (HEVC) | quality 2..94 (step 2) | `.heic` | CPU |
| `jpeg_fzt` | JPEG-FzT | this package (NumPy) + Pillow JPEG | JPEG quality 1, 3, ..., 95 of the half-resolution image | `.fzt` | CPU |
| `jpeg_ai` | JPEG-AI | JPEG-AI reference software, commit `e9648f9`, HOP profile, tools off | target bpp x 100 in 2..50, analytic fit | `.jpegai` | CUDA |
| `neural_bmshj2018` | bmshj2018 | CompressAI 1.2.8, `bmshj2018-factorized` (MSE) | model quality 1..8 | `.ptci` | CUDA |
| `neural_mbt2018_mean` | mbt2018 | CompressAI 1.2.8, `mbt2018-mean` (MSE) | model quality 1..8 | `.ptci` | CUDA |

The two face1kb codecs are `ours_fast` / `ours_accurate` (`.bin`).

## Installation

```bash
pip install -e ".[codecs]"        # Pillow 12.2.0, pillow-heif 1.4.0, pillow-jxl-plugin 1.3.7
scripts/setup_jpegai.sh           # JPEG-AI reference software (see below); also installs the "jpegai" extra
```

CompressAI (and torch) are core dependencies of the package. The pretrained
CompressAI weights are downloaded by CompressAI into the torch hub cache on first use
(about 12 MB for bmshj2018 qualities 1-5 and 29 MB for 6-8; about 29 MB for mbt2018
qualities 1-4 and 71 MB for 5-8; about 0.65 GB for all 16 models).

The JPEG-FzT inverse-basis table `face1kb/baselines/data/jpeg_fzt_ibf.txt` is package
data; it must be present in the installed package (it is with an editable install
`pip install -e .`).

The `codecs` extra pins the paper's codec library versions (Pillow 12.2.0,
pillow-heif 1.4.0, pillow-jxl-plugin 1.3.7; compressai 1.2.8 is pinned in the core
dependencies), which byte-identical bitstreams need; `requirements/paper.txt` holds the
complete paper environment. See [Reproducibility limits](#reproducibility-limits).

## Quick start

```python
import numpy as np
from PIL import Image
from face1kb.baselines import get_codec, decode_file, CODECS

img = np.asarray(Image.open("face_112.png").convert("RGB"))  # aligned square crop

codec = get_codec("webp")
data, info = codec.encode_to_budget(img, 1024)   # info = {"fitted", "setting", "size"}
img_hat = codec.decode(data)                     # 112 x 112 x 3 uint8
print(codec.ext, codec.device, codec.knob)

img_hat = decode_file("crop.webp")               # any stored file, by extension
```

`encode_to_budget(img, budget, **options)` returns the stored bytes and
`{"fitted": bool, "setting": the selected knob value, "size": the size compared with
the budget}`. `img` is an `H x W x 3` `uint8` RGB array (a PIL image is accepted too).

Argument order: the package-level shorthand is
`face1kb.baselines.encode_to_budget(codec, img, budget)`, while the family functions
take the image first: `classical.encode_to_budget(img, codec, budget)`,
`compressai_codecs.encode_to_budget(img, codec, budget)`,
`jpeg_ai.encode_to_budget(img, budget)`, `jpeg_fzt.encode_to_budget(img, budget)`.
`get_codec` raises `TypeError` for a name that is not a string.

## Budget search

`face1kb.baselines.search.binary_search_fit(encode, settings, budget)` is the search
of the benchmark. `settings` is ordered so that the output size grows with the index
(ascending quality, descending JPEG 2000 ratio); `encode(setting)` returns
`(size, payload)`.

* `lo, hi = 0, n - 1`; the midpoint setting is encoded; if its size is `<= budget` it
  becomes the current best and `lo` moves up, otherwise `hi` moves down. The result is
  the fitting setting found last on that path (the largest one for a monotone size
  curve).
* **Fallback**: if no probed setting fits, `settings[0]` (the smallest output) is
  encoded and returned with `fitted=False`. The benchmark stores that over-budget file
  ("failure to compress") instead of dropping the crop; budget-compliance tables are
  computed from the sizes of the stored files.

The classical codecs and JPEG-FzT compare the whole file with the budget. CompressAI
compares the entropy-coded size (see below); JPEG-AI uses its own fit.

Side studies use other searches, which are available as options:

| Study | Search |
|---|---|
| Main grid, recompression, preprocessing through the codec | `binary_search_fit` (the default); JPEG-FzT with `qualities=jpeg_fzt.QUALITIES_EVEN` (2..94) in the recompression and preprocessing studies |
| Sample difficulty | exhaustive scan keeping the largest fitting output, encoder errors skipped, coarser grids: `search="scan", skip_errors=True, grid=classical.settings(codec, quality_step=4, ratio_step=12)` |
| Codec comparison | exhaustive scan (`search="scan", skip_errors=True`); JPEG-FzT with an early-stopping scan (`search="scan", stop_at_overflow=True, qualities=QUALITIES_EVEN`) |
| Speed benchmark | early-stopping scan that keeps the last fitting setting: `scan_fit(..., keep="last", stop_at_overflow=True)` |
| ISO/IEC 29794-5 annex | `hinted_binary_search_fit(encode, settings, budget, hint)`: a binary search on a +-4 window around the previous image's answer, with a fall-back to the full search |

All searches share the fallback rule. `scan_fit` keeps a later setting only if its
output is strictly larger (`keep="largest"`) unless `keep="last"`.

## Classical codecs

`face1kb.baselines.classical`: `encode_to_budget(img, codec, budget, grid=None,
search="binary", strip_metadata=False)`, `encode_setting(img, codec, setting)`,
`settings(codec, quality_step=2, ratio_step=4)`, `save_kwargs(codec, setting)`,
`decode(data)` and `decode_pil(data)`.

The encoder options are exactly the Pillow `save` arguments listed in the table above
(`save_kwargs`). The stored file is the container Pillow writes, and its full size
counts against the budget.

Metadata: a PIL input is encoded with its `info`, as in the benchmark. Aligned crops
carry no metadata, but decoded images can: an image decoded from JPEG XL carries an
ICC profile that the AVIF and HEIF encoders embed, and the comment of a decoded
JPEG 2000 file is written by the JPEG encoder. The recompression study chained codecs
this way (`decode_pil` returns such an image). `strip_metadata=True` encodes the pixels
only; NumPy inputs never carry metadata.

## JPEG-FzT

`face1kb.baselines.jpeg_fzt` implements JPEG-FzT (Perfilieva and Hurtik, "The
F-transform preprocessing for JPEG strong compression of high-resolution images",
Information Sciences 550, 2021) in NumPy:

1. **Encode**: a direct F-transform with a raised-cosine basis of radius `h = 2`
   filters the crop and samples it on a grid of step 2, giving a half-resolution
   image (56 x 56 for a 112 px crop). It is stored as a baseline JPEG with
   `optimize=True`; the `.fzt` file *is* that JPEG. The budget search runs over the
   JPEG quality 1, 3, ..., 95.
2. **Decode**: the JPEG is decoded and the inverse F-transform reconstructs the full
   resolution with the tabulated inverse basis (`face1kb/baselines/data/jpeg_fzt_ibf.txt`,
   4,990 values) and a neighbourhood radius `rad = 1`, which all results of the paper
   use.

`decode(data, size=None, rad=1)` reconstructs twice the stored size by default (the
original size for the even benchmark resolutions). `encode_stage` / `decode_stage`
expose the two transforms (also under their original names `_encode_stage` /
`_decode_stage`).

## JPEG-AI

JPEG-AI (Rec. ITU-T T.840.1 | ISO/IEC 6048-1) runs through its official reference
software, in-process on a CUDA GPU (`face1kb.baselines.jpeg_ai`).

### Setup

```bash
scripts/setup_jpegai.sh [--dir DIR] [--no-install] [--no-models] [--no-build] [--no-check]
export FACE1KB_JPEGAI_DIR=DIR     # only for a non-default location
```

The script (idempotent):

1. clones `https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git` and checks
   out commit `e9648f9a4bb98d8fb7333e97d4df4f0979520a7c`;
2. applies `third_party/jpeg-ai.patch` (below);
3. restores the upstream `cfg/betas` files and makes the checkout writable;
4. runs `pip install -e ".[jpegai]"` (and installs pybind11 if it is missing);
5. fetches the models with git-lfs (`models/` only, about 1 GB; the repository's LFS
   objects are public);
6. builds the two C++ entropy-coder libraries for the active Python;
7. encodes and decodes one synthetic image if a CUDA GPU is visible.

It needs git, git-lfs, make, g++ with OpenMP and the Python development headers. The
default location is `third_party/jpeg-ai-reference-software` in this repository
(`face1kb.config.JPEGAI_DIR`, environment variable `FACE1KB_JPEGAI_DIR`).

`third_party/jpeg-ai.patch` holds the local changes the benchmark ran with:

* `src/__init__.py`: `torch.load` defaults to `weights_only=False`, as before
  torch 2.6 (the reference software loads pickled checkpoints and caches);
* `src/codec/entropy_coding/binarizers.py`: the bypass decoder returns an array; the
  unary and Exp-Golomb header decoders use its first element (without it the
  bitstreams do not decode with the numpy / torch versions of the paper environment);
* `src/reco/coders/decoder.py`: the ptflops hooks are installed only with
  `--calc_ptflops`;
* `scripts/build_ec_lib.sh`, `scripts/build_test_libs.sh`: conda is activated only if
  it exists.

### Configuration of the benchmark

* Coding configuration `cfg/tools_off.json` plus the **high operating point** profile
  `cfg/profiles/high.json` (HOP), for encode and decode.
* Rate knob: the encoder's `--set_target_bpp` (target bits per pixel x 100), clamped
  to 2..50 (`BPP_RANGE`).
* Budget fit (`analytic_fit`): one encode at the analytic target
  `round(budget * 8 / pixels * 100)` clamped to 2..50; if the stream is over budget, at
  most two corrections to `round(bpp * budget / size * 0.92)` (clamped; a target
  already tried is lowered by 2, and the fit stops if that is still a repeat or below
  2). If nothing fits, the last encode is returned with `fitted=False`.
* A stream smaller than `MIN_PLAUSIBLE_FRAC = 0.25` of the budget is treated as a
  failed encode, never as a fit. (Such streams were truncated writes.) The genuine
  0.1 % size quantile at 64 px / 1024 B is 268 B, close to this 256 B floor.
* A failed encode counts as an infinitely large stream, so the correction after it
  targets the lowest rate (bpp x 100 = 2), and that stream is kept if it is above the
  floor (a warning is logged). This is the benchmark's behaviour.
* If the final attempt produced no plausible stream, `encode_to_budget` raises
  `JpegAIError`.

```python
from face1kb.baselines import jpeg_ai

data, info = jpeg_ai.encode_to_budget(img, 1024)            # HOP, tools off, analytic fit
data = jpeg_ai.encode_bpp(img, 16, profile="sop")           # one encode at a fixed target
data, recon = jpeg_ai.encode_bpp(img, 16, return_recon=True)
img_hat = jpeg_ai.decode(data)                              # HOP decoder
ref = jpeg_ai.ReferenceSoftware("/path/to/checkout")        # explicit checkout
```

`jpeg_ai.decode` returns `None` (and `face1kb.baselines.decode_file` /
`decode_bytes` raise `JpegAIError`) for a malformed input: before calling the reference decoder, `check_bitstream(data)` walks
the substream structure (SOC marker `0xFF80`, picture header first, each substream's
exp-Golomb size within the data, EOC marker `0xFF81`). The reference decoder itself
never returns on an empty, truncated or foreign file.

`profile` is `"sop"`, `"bop"` or `"hop"` (the operating-point study of the paper uses
all three at a fixed target of 16). `step_down_fit` and `hinted_fit` are the fits of
the codec-comparison / sample-difficulty and annex studies.

### Runtime notes

* The reference software resolves its configuration files relative to its checkout,
  so each call runs with the working directory set to the checkout (serialised by a
  lock) and the checkout's top-level `src` package on `sys.path`; no other module
  named `src` may be imported.
* With `--set_target_bpp` the reference encoder deletes and re-writes a per-image log
  `cfg/betas/<op>/betas_*.txt` inside the checkout (and never reads it back in this
  mode). face1kb points that log to a private temporary folder, which leaves the
  bitstreams unchanged (verified byte for byte) and the checkout untouched, so
  concurrent processes can share one checkout.
* The entropy coder writes `src/codec/entropy_coding/lib_wrappers/mans/cache.pt` on
  first use; run one encode (the setup check does) before starting parallel jobs.
* The models load on the first encode / decode of a process (about 20 s); a
  `ReferenceSoftware` instance holds GPU state and is not thread-safe: use one process
  per GPU and select it with `CUDA_VISIBLE_DEVICES`.
* Bitstreams and reconstructions go through a temporary folder (`TMPDIR`).
* The reference software switches PyTorch to cuDNN's deterministic algorithms and
  disables TF32 globally. face1kb applies these settings only for the duration of
  each call and restores the caller's settings afterwards, so other torch code in the
  same process keeps its numerics (the paper's scripts did not; see the CompressAI
  budget-sweep note below).

## CompressAI: bmshj2018 and mbt2018

`face1kb.baselines.compressai_codecs` uses the pretrained MSE models of CompressAI
1.2.8 (`bmshj2018-factorized`, `mbt2018-mean`) at quality 1..8.

* Input: RGB in [0, 1] (float32), edge-padded right/bottom to a multiple of 64 px.
* The quality is binary-searched on the **entropy-coded size**, the summed length of
  the latent strings.
* Container (`.ptci`): a pickle (protocol 4) of `{"strings", "shape", "size", "codec",
  "quality"}`. It adds 125 B (bmshj2018) or 135 B (mbt2018) to the entropy-coded
  size in almost all stored files. Pickle writes a byte string shorter than 256 B with a
  1-byte instead of a 4-byte length, so the overhead is 3 B less when the latent string
  `y` is shorter than 256 B (and would be 3 B more if an mbt2018 hyperprior string `z`
  reached 256 B). A stored file can therefore exceed the budget although the search
  fitted: at 112 px /
  1024 B, 59.3 % (bmshj2018) and 67.0 % (mbt2018) of the stored files are within the
  budget, as reported in the paper. The budget-compliance tables count file sizes.
* Reading: `unpack` uses a restricted unpickler that resolves no global except
  `torch.Size` and checks the structure, so a `.ptci` file cannot execute code.
* Decode: `decompress`, clamp to [0, 1], crop, `(x * 255).round()`. Decode on the
  device type the stream was encoded on: the paper streams (CUDA) decode to noise on
  the CPU, because the entropy-model tables differ; `decode` logs a warning for a
  non-CUDA device.
* Numerics: every encode and decode runs with fixed cuDNN / TF32 settings
  (`paper_numerics()`: no autotuning, no TF32, cuDNN's default algorithm selection),
  whatever the global torch settings are. `encode_to_budget(..., deterministic=True)`
  selects cuDNN's deterministic algorithms instead: the paper's 768 B / 960 B
  budget-sweep cells were encoded that way (they ran in the same process as JPEG-AI,
  whose reference software switches cuDNN to deterministic algorithms globally),
  the main grid with the defaults.
* mbt2018-mean decodes its latent with scale indexes that the decoder recomputes from
  the hyperprior. When a scale lies on a boundary of the scale table, a different
  kernel (one of cuDNN's non-deterministic default kernels, or the deterministic
  kernels a stream was encoded with) selects another index and the entropy decoder
  desynchronises: the reconstruction becomes noise. `decode` therefore verifies the
  decoded latent: re-encoding it with the decoder's indexes must reproduce the stored
  string, which holds only with the encoder's indexes. Otherwise the hyperprior is
  re-run with the default kernels (up to `attempts=4` times) and then with the
  deterministic kernels; `verify=False` decodes once, as the paper's code did. The
  encoder is affected in the same way: for a file with a boundary scale a re-encode
  can write another, equally valid stream of the same size.

The paper ran these two codecs on the first 300 crops of the Color FERET index at the
main-grid cells, and on all Color FERET crops at 112 px with 768 B and 960 B (budget
sweep); they were not run on AI-Solutions-KK.

## Decoding stored files

`decode_file(path, res=None, device="cuda", cache=False, **options)` and
`decode_bytes(data, ext, ...)` choose the decoder by the file extension:

| Extension | Decoder |
|---|---|
| `.jpg`, `.jp2`, `.webp`, `.jxl`, `.avif`, `.heic` | Pillow (with the HEIF and JPEG XL plugins) |
| `.fzt` | JPEG-FzT inverse F-transform (`res` sets the output size) |
| `.jpegai` | JPEG-AI reference decoder, HOP profile (`profile`, `tools_off`, `workdir`, `repo_dir`) |
| `.ptci` | CompressAI (`device`) |
| `.bin` | face1kb container, FAST or ACCURATE by its header (`face1kb.decode`, `device`, `weights_dir`) |

All return an `H x W x 3` `uint8` RGB array. Options a decoder does not take are
ignored.

Bitstreams are stored under `WORK_ROOT/<dataset>/compressed/<res>[<suffix>]px_<budget>B/<codec>/<subject>/<stem><ext>`
(`face1kb.config.compressed_dir`). JPEG-AI and face1kb decodes are slow enough to be
cached as PNG at `WORK_ROOT/<dataset>/decoded/<cell>/<codec>/<subject>/<stem>.png`
(`face1kb.config.decoded_dir`; `decoded_cache_path(bitstream)` maps one to the other).
`decode_file(path, cache=True)` returns the cached PNG when it exists.

## Checking an installation

```bash
python -m face1kb.baselines.verify [--codecs jpeg,webp,...] [--res 112] [--budget 1024] [--device cuda] [--strict]
python -m face1kb.baselines.verify --scan-jpegai WORK_ROOT/colorferet/compressed
```

The first form encodes a synthetic image to the budget with each codec, decodes it and
checks the output; a codec whose backend is missing is reported as `missing` (the exit
status is non-zero only for failures, or for missing codecs with `--strict`). The
second lists `.jpegai` files smaller than 25 % of their cell's budget (truncated
writes).

## Cost

Per crop, on one RTX 2080 Ti / one CPU core (indicative):

* classical codecs: 5-6 encodes of the binary search, milliseconds each (AVIF and
  HEIF are the slowest);
* JPEG-FzT: one F-transform plus 5-6 JPEG encodes; the decode (inverse F-transform)
  takes about 30 ms at 112 px on one core;
* JPEG-AI: 1-3 encodes of 2-4 s each (plus about 20 s to load the models once per
  process), decode about 0.5-1 s;
* CompressAI: 3-4 GPU encodes, well below a second.

## Reproducibility limits

* **Library versions change bytes.** The classical bitstreams are byte-identical to the
  paper's only with the paper's builds (Pillow 12.2.0, pillow-heif 1.4.0,
  pillow-jxl-plugin 1.3.7). For example, Pillow 10.1 selects the same JPEG 2000 ratio
  and size but writes a different OpenJPEG version comment; the AVIF, HEIF and JPEG XL
  encoders change between releases.
* **GPU codecs depend on the device and software stack.** JPEG-AI and CompressAI
  bitstreams and decodes were verified on RTX 2080 Ti (Turing) GPUs with torch 2.10.0
  / CUDA 12.8 and compressai 1.2.8; other GPU generations or versions can give
  different bitstreams. The stored CompressAI streams do not decode on the CPU at all
  (most reconstructions become noise, for bmshj2018 without any warning from
  CompressAI); use `device="cuda"`.
* **CompressAI decodes.** CUDA decodes repeat only up to 1 grey level in a few pixels
  (non-deterministic cuDNN kernels). The paper decoded mbt2018 files without checking
  the latent, so its mbt2018 metrics may include a few corrupt decodes in every cell.
  In the main-grid cells the files that desynchronise vary between processes and runs
  (observed: 1-3 per 300-file cell; a given file can decode correctly in one process
  and to noise on every attempt in another, e.g. 3 of 3,003 in one full scan). In the
  768 B / 960 B cells, whose streams were encoded with deterministic kernels, about
  1.6 % / 2.4 % of the files (187 / 272 of 11,335 in one scan) decode to noise with the
  default kernels on every attempt. The verified decode reconstructs all of them.
* **CompressAI encodes.** For files with a boundary scale an mbt2018 re-encode can
  write a stream of the same size with different bytes, and matches the stored bytes
  only intermittently (e.g. 2 of 720 re-encodes differed in one of four validation
  runs). Re-encoding the 768 B / 960 B cells needs `deterministic=True`.
* **JPEG-AI operating point.** The benchmark ran the high operating point (HOP,
  `cfg/profiles/high.json`) for all JPEG-AI results except the operating-point study,
  although the paper's codec-settings table names SOP.
* **Mixed provenance of the paper's Color FERET JPEG-AI files.** In every Color FERET
  cell, 31.6 % of the stored JPEG-AI streams (3,578-3,584 of 11,335) were written by an
  earlier run whose bpp search differed (same HOP / tools-off configuration, other
  targets; e.g. median 1007 B instead of 303 B at 64 px / 1024 B, 827 B instead of
  980 B at 224 px / 1024 B). They decode normally, but a re-run applies the final fit to
  all crops, so the Color FERET JPEG-AI results of a re-run differ from the paper's.
  3-4 Color FERET crops per cell have no stored JPEG-AI stream. The AI-Solutions-KK
  cells used the final fit throughout.
* **JPEG-AI lowest-rate fallbacks.** When the first encode of a crop failed (a
  transient error such as a full temporary disk), the fit's correction targets bpp x
  100 = 2 and keeps that stream if it is above the 25 % floor. At least 8 stored
  main-grid streams are such fallbacks (Color FERET 168 px / 512 B: 4, 224 px / 512 B:
  1; AI-Solutions-KK 224 px / 512 B: 3; 130-242 B, byte-identical to an encode at
  bpp x 100 = 2). A re-run, where the first encode succeeds, gives normal streams
  (about 430-490 B) for these crops.
* **Repaired JPEG-AI streams.** Truncated JPEG-AI streams (below 25 % of the budget)
  found in the store after the grid run were re-encoded in place; their manifest rows
  still show the truncated size and bpp x 100 = 2 while the stored file is the
  re-encoded stream. File sizes, not manifest rows, are authoritative.
* Three truncated JPEG-AI streams (below 25 % of the budget) remain in the paper's
  adversarial-sanitisation cells (112 px, HFC 6 px / 512 B: 2; HFC 10 px / 1024 B: 1);
  a re-run with the plausibility floor encodes them normally.

## Differences between the paper text and the code

The code is what produced the published numbers:

| Paper text | Code |
|---|---|
| JPEG-AI "profile SOP" (codec-settings table) | HOP (`cfg/profiles/high.json`) for encode and decode |
| JPEG 2000 compression ratio 8-400 | 12-400 in steps of 4 (`range(400, 8, -4)`) |
| AVIF via pillow-heif / libaom | Pillow's native AVIF plugin (libavif) |
| JPEG-FzT "reference C++/Qt implementation" | the NumPy implementation of this package (the same transforms) |
| JPEG-FzT radius study with a triangle-kernel variant | the package implements the raised-cosine basis with any `rad`; the triangle kernel is not implemented |
| JPEG-FzT default radius `rad=2` in the codec benchmark (radius ablation) | `rad=1` for every result (`jpeg_fzt.RAD`) |
| Quality search "capped at quality 94" (codec-settings caption) | true for the classical codecs; the JPEG-FzT search runs over JPEG quality 1, 3, ..., 95 |
| bmshj2018 / mbt2018 "pretrained hyperprior / joint models" | `bmshj2018-factorized` (factorized prior, no hyperprior) and `mbt2018-mean` (mean-scale hyperprior, no autoregressive context model) |

## Licences

* face1kb code, including the JPEG-FzT implementation and its inverse-basis table: MIT.
  The JPEG-FzT method is by Perfilieva and Hurtik (University of Ostrava;
  Information Sciences 550:221-238, 2021).
* Pillow (HPND), libjpeg-turbo, OpenJPEG, libwebp, libavif / aom and libjxl
  (BSD-style) are pip dependencies.
* pillow-heif (BSD-3-Clause) wheels bundle libheif (LGPL) and x265 (GPL); HEVC is
  patent-encumbered. They are installed from PyPI, never vendored.
* CompressAI: BSD-3-Clause-Clear, which grants no patent rights; the pretrained
  weights are downloaded from CompressAI's servers.
* JPEG-AI reference software: BSD-3-Clause (ITU/ISO/IEC); its licence grants no
  patent rights. It is fetched by `scripts/setup_jpegai.sh`, not distributed here;
  `third_party/jpeg-ai.patch` modifies it.
