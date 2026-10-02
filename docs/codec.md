# The face1kb codecs

face1kb ships two learned face codecs that compress an aligned face crop into a
self-describing container of **at most *B* bytes** (the paper evaluates *B* = 1024 and
512) while preserving the identity seen by face-recognition (FR) systems:

* **face1kb-FAST** -- a slim, attention-free codec (1.35 M parameters) with no FR model
  in the loop at inference time;
* **face1kb-ACCURATE** -- a wider codec with attention (18.7 M parameters, of which
  3.65 M are a frozen EdgeFace-S face-recognition anchor) and an identity side-stream
  that stores an entropy-coded identity code (with the released weights this code is
  the same 8 bytes for every image, see [Limitations](#limitations)).

They are called **Ours-FAST** and **Ours-ACCURATE** in the paper (arXiv:2608.22866,
Section "A Custom Identity-Preserving Codec") and in the result files. The weights are
in [`weights/`](../weights/README.md) (git LFS, CC BY-NC-SA 4.0); the code is MIT.

- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Container format](#container-format)
- [Hard byte budget](#hard-byte-budget)
- [API](#api)
- [Command line](#command-line)
- [Performance](#performance)
- [Reproducing the paper bitstreams](#reproducing-the-paper-bitstreams)
- [Limitations](#limitations)
- [Differences between the paper text and the code](#differences-between-the-paper-text-and-the-code)
- [Licence](#licence)

## Quick start

```bash
git clone https://github.com/innovatrics/1kb_face_ijcb && cd 1kb_face_ijcb
git lfs pull                      # the weights
pip install -e .                  # core dependencies: torch, compressai==1.2.8, timm==1.0.24, ...
```

The core dependencies bound torch to `< 2.11`, so a fresh install gets torch 2.10.0,
the CUDA 12.8 build on PyPI (it needs an NVIDIA driver that supports CUDA 12.8; the
PyPI wheels of torch 2.11 and newer are CUDA 13 builds that need a newer driver).
Exact paper bitstreams need this torch 2.10 / CUDA 12.8 stack with `compressai==1.2.8`
and `timm==1.0.24` (see [Reproducing the paper bitstreams](#reproducing-the-paper-bitstreams));
`requirements/paper.txt` pins the complete paper environment.

```python
import numpy as np
from PIL import Image
import face1kb

img = np.asarray(Image.open("face_112.png").convert("RGB"))  # aligned square crop
codec = face1kb.load("accurate", device="cuda")               # or "fast"
data = codec.encode(img, budget=1024)                         # bytes, len(data) <= 1024
img_hat = codec.decode(data)                                  # 112 x 112 x 3 uint8
```

Inputs are square RGB crops aligned to the ArcFace 5-point template (the 112 x 112
template, scaled for other sizes). The codecs were trained at 64-256 px and evaluated
at 64, 96, 112, 168 and 224 px.

## Architecture

Both variants subclass CompressAI's variable-rate mean-scale hyperprior
`MeanScaleHyperpriorVbr` (Kamisli, Racape and Choi, DCC 2024: gain vector,
`QuantABCD` quantisation-reconstruction offsets and a variable-rate hyperprior step;
built on the hyperpriors of Ballé et al. 2018 and Minnen et al. 2018 and on the gain
units of Cui et al. 2021) as `face1kb.codec.backbone.MeanScaleHyperpriorVBR3`:

* **Third hyper-downsample.** The hyper-analysis has a third stride-2 stage (mirrored by
  a third x2 upsample in the hyper-synthesis), so the hyperprior latent *z* is *x*/128
  instead of *x*/64. With a small hyperprior width *N<sub>z</sub>* this shrinks the
  *z* byte floor by roughly an order of magnitude. (The *z* quantisation step is
  derived from the gain, but for both released models it sits at its lower bound of
  0.5 for all 64 table gains, so the *z* stream is effectively gain-independent.) Inputs are
  replicate-padded (right/bottom) to a multiple of 128; 112 and 96 px are padded to
  128, 168 and 224 px to 256.
* **Gain-unit variable rate.** A fixed gain vector (8 training levels) scales the
  latent before quantisation, so one model spans all rates. The vector is not trained
  (CompressAI detaches it in the training forward): FAST keeps CompressAI's default
  0.1-1.0 and ACCURATE its log-spaced 0.04-1.0 initialisation, and the network is
  trained at those gains. At encode time the gain is taken from a frozen 64-entry
  log-spaced table spanning the training range.
* **Conditional synthesis.** Every x2 upsample of the synthesis is a *ResizeConv*
  (nearest-neighbour upsample + 3 x 3 convolution, no transposed convolution) and each
  of the three inverse-GDN stages is followed by FiLM on `[gain, res / 256]`.

| | face1kb-FAST | face1kb-ACCURATE |
|---|---|---|
| Class | `FaceCodecFast` | `FaceCodecAccurate` |
| `variant_id` in the container | 0 | 1 |
| Channels N / M / N<sub>z</sub> | 64 / 96 / 48 | 192 / 320 / 64 |
| Attention blocks | none | 2 in the analysis, 2 in the synthesis |
| Identity side-stream | none | frozen EdgeFace-S -> Linear(512, 128) + LayerNorm -> factorized entropy bottleneck -> FiLM (scale 1 +- 0.2, shift +- 0.2) on the 3 synthesis stages; with the released weights the stored code is constant (see [Limitations](#limitations)) |
| Refinement head | none | `clamp(x_hat + 0.1 * residual(x_hat, p_hat), 0, 1)`, conditioned on the decoded identity code |
| Gain range (64-entry table) | 0.1 - 1.0 | 0.04 - 1.0 |
| Training lambda range | 0.0018 - 0.18 | 0.0002 - 0.18 |
| Parameters | 1,352,021 | 18,712,400 (15,059,880 trained) |
| Weight file | 5.4 MB | 74.9 MB |

For ACCURATE the EdgeFace-S anchor is part of the **encoder**: every encode runs one
EdgeFace-S forward pass on the (unpadded) crop, resized to 112 px, and stores the
entropy-coded 128-D identity code in the container; the decoder entropy-decodes it
before the synthesis. With the released weights the projection has collapsed, so this
code (and therefore the side-channel) is the same for every image; the forward pass
does not change the output for any tested image. FAST never runs an FR model.

## Container format

Big-endian; `face1kb.codec.container` (`pack` / `unpack`):

| Bytes | Field | Meaning |
|---|---|---|
| 1 | flags | `(variant_id << 6) \| rate_index`; variant 0 FAST, 1 ACCURATE, 2 identity-only, 3 reserved; `rate_index` 0-63 into the model's gain table |
| 1 | res_bucket | 64 -> 0, 128 -> 1, 192 -> 2, 256 -> 3, 112 -> 4, 224 -> 5; 255 = raw geometry |
| 1 | sc_len | identity side-channel length (0 for FAST) |
| sc_len | side-channel | identity code (ACCURATE; with the released weights always the same 8 bytes, `f82a5c8500000000` on CUDA) |
| 2 | len(y) | length of the *y* rANS string (absent for identity-only) |
| len(y) | y string | spatial latent |
| 2 | len(z) | length of the *z* rANS string (absent for identity-only) |
| len(z) | z string | hyperprior latent |
| 4 | H, W | uint16 side lengths, only if res_bucket = 255 (e.g. 96 and 168 px) |

The fixed overhead excluding payloads is 7 bytes (paper Table `tab:codec-container`),
plus 4 bytes at resolutions without a bucket code. rANS strings are not
self-delimiting, hence the explicit length prefixes; there is no CRC. The
identity-only variant (no spatial latent) decodes to a black frame.

## Hard byte budget

A learned codec produces whatever size its gain yields, so the budget is enforced at
encode time (`face1kb.codec.budget.encode_to_budget`):

1. The crop is converted to float `[0, 1]` and padded to a multiple of 128. For
   ACCURATE the side-channel is encoded from the unpadded crop.
2. A **binary search** over the 64-entry gain table (larger gain = finer quantisation =
   more bytes) finds the largest gain whose *real* rANS output fits the target
   `budget - 7 - len(side-channel)` (about 6 real encodes). The entropy-coder tables of
   each gain are built once and cached on the model instance.
3. The chosen table index is stored in the header, so the decoder reconstructs the gain
   exactly.

**Accounting at 96/168 px.** The paper encoder ignored the 4-byte raw-geometry
trailer, so at 96 and 168 px some paper containers exceed the budget. The overshoot
is at most 4 bytes; because the rANS strings are whole 32-bit words it is exactly 3
bytes for budgets that are multiples of 4, which is the case for every such paper
container. `paper_compat=True` reproduces this accounting (and therefore the paper
bitstreams); the default `paper_compat=False` reserves the trailer. At bucketed
resolutions (64, 112, 128, 192, 224, 256 px) both settings run the same search, so
every stream that fits is byte-identical; where even the lowest gain overflows, the
default emits the identity-only container instead of the paper's over-budget floor
stream (`overflow="floor"` keeps the paper bytes).

**When nothing fits.** If even the lowest gain overflows, `overflow` decides:

| `overflow` | Output | Size |
|---|---|---|
| `"identity"` (default without `paper_compat`) | identity-only container (black frame on decode), with an `IdentityOnlyWarning` | `<= budget` (budgets below the identity-only size raise `ValueError`) |
| `"floor"` (default with `paper_compat=True`; the paper behaviour) | the lowest-gain stream, flagged `over_budget` | exceeds the budget |
| `"error"` | raises `face1kb.codec.budget.BudgetError` | -- |

With the defaults, `len(codec.encode(img, budget)) <= budget` holds for every budget
of at least the identity-only container size, `3 +` side-channel bytes, plus 4 at
raw-geometry resolutions (FAST 3 / 7 B, ACCURATE 11 / 15 B); smaller budgets raise
`ValueError`. An identity-only result issues a
`face1kb.codec.budget.IdentityOnlyWarning` (and `info["identity_only"]` is set).
With `paper_compat=True` a budget too small for the header and the side-channel
yields the identity-only container, as in the paper encoder.

Share of crops whose paper container holds the budget (container length <= budget,
%; paper Tables `tab:fit-cf`, `tab:fit-kk` and their 1024 B companions). These shares
are computed from the byte lengths of the stored containers. The encoder's
`info["fitted"]` flag cannot be used for them: it means that a spatial stream fit the
search target, which with `paper_compat=True` does not imply that the container holds
the budget at 96 and 168 px (use `info["within_budget"]` instead).

| Budget | Dataset | Codec | 64 px | 96 px | 112 px | 168 px | 224 px |
|---|---|---|---:|---:|---:|---:|---:|
| 1024 B | Color FERET | Ours-FAST | 100 | 87 | 100 | 88 | 100 |
| 1024 B | Color FERET | Ours-ACCURATE | 100 | 90 | 100 | 90 | 100 |
| 1024 B | AI-Solutions-KK | Ours-FAST | 100 | 87 | 100 | 87 | 100 |
| 1024 B | AI-Solutions-KK | Ours-ACCURATE | 100 | 90 | 100 | 90 | 100 |
| 512 B | Color FERET | Ours-FAST | 100 | 74 | 100 | 50 | 80 |
| 512 B | Color FERET | Ours-ACCURATE | 100 | 80 | 100 | 80 | 100 |
| 512 B | AI-Solutions-KK | Ours-FAST | 100 | 74 | 100 | 9 | 14 |
| 512 B | AI-Solutions-KK | Ours-ACCURATE | 100 | 79 | 100 | 79 | 100 |

A scan of all 577,380 stored paper containers shows two kinds of misses:

* **Trailer overruns** (exactly 3 bytes over, only at 96 and 168 px), which
  `paper_compat=False` removes. They are all misses of FAST at 96 px and at
  168 px / 1024 B, and 99.7 % of the misses of ACCURATE (17,558 of 17,614).
* **Genuine rate floors** (even the lowest gain does not fit; with the default
  `overflow` policy these crops become identity-only). FAST at 168 px / 512 B (31 %
  of Color FERET and 87 % of AI-Solutions-KK crops more than 4 bytes over; the rest
  of its misses are trailer overruns) and at 224 px / 512 B (20 % and 86 % over).
  ACCURATE has only 56 such containers: 13 and 35 of Color FERET at 96 px / 1024 B and
  512 B, 1 and 1 of AI-Solutions-KK at 96 px, and 2 and 4 of AI-Solutions-KK at
  224 px / 1024 B and 512 B.

Over all 54,030 misses at 96 and 168 px, 65 % are trailer overruns; the rest are
the rate floors of FAST at 168 px / 512 B and the 50 ACCURATE floors at 96 px.

## API

```python
import face1kb

codec = face1kb.load(variant="fast", device="cuda", weights_dir=None)
```

`face1kb.load(variant, device="cuda", weights_dir=None, *, weights_path=None,
reproducible=True)` builds `FaceCodecFast()` / `FaceCodecAccurate()` with the default
constructor arguments, loads `face1kb_<variant>.safetensors` from `weights_dir`
(default `FACE1KB_WEIGHTS_DIR`, i.e. `<repo>/weights`) with `strict=True`, and returns
a `Codec`. It raises if CUDA is requested but unavailable and warns when running on
CPU (see [Limitations](#limitations)). `reproducible=True` calls
`face1kb.codec.set_reproducible_backend()`, which disables TF32 in cuDNN convolutions
and matmuls and cuDNN autotuning (global `torch.backends` flags), so newer GPUs compute
in full fp32 like the GPUs the paper streams were produced on.

`Codec` methods and attributes:

| | |
|---|---|
| `encode(img, budget=1024, *, paper_compat=False, return_info=False, overflow=None)` | `img`: `H x W x 3` `uint8` RGB, `H == W`. Returns the container `bytes`, or `(bytes, info)` with `return_info=True`; `info` has `fitted` (a spatial stream fit the search target), `over_budget` (the lowest-gain stream was emitted although it does not fit), `identity_only`, `within_budget` (`len(bytes) <= budget`), `rate_index`, `gain`, `bytes`. With `paper_compat=True`, `fitted` does not imply `within_budget` (96/168 px trailer). |
| `decode(data)` | `H x W x 3` `uint8` RGB; the float output is sanitised (`nan_to_num`), clamped to `[0, 1]` and converted with `(x * 255).round()`. Identity-only containers give a black frame. |
| `decode_tensor(data)` | `(x_hat, header)`: `x_hat` is a `(1, 3, res, res)` float tensor in `[0, 1]`, or `None` for identity-only containers. |
| `Codec.info(data)` / `face1kb.codec.inspect(data)` | header fields and payload sizes, without decoding. |
| `net` | the underlying `torch.nn.Module` (eval mode). |
| `variant`, `device`, `metadata`, `num_parameters` | |

`face1kb.decode(data, device="cuda")` decodes any container, loading (and caching) the
variant named in its header.

A `Codec` is stateful (per-gain CDF cache, the decoded identity code of the last
ACCURATE decode) and not thread-safe; use one instance per thread or process.

## Command line

```bash
python -m face1kb.codec encode face.png --variant accurate --budget 1024   # -> face.f1k
python -m face1kb.codec info face.f1k
python -m face1kb.codec decode face.f1k                                    # -> face_decoded.png
```

The output path is an optional positional argument, before or after the options
(`encode face.png --budget 512 out.f1k`): `encode` defaults to `<input stem>.f1k`
and `decode` to `<input stem>_decoded.png`, so a round trip never replaces the
source crop, and neither command overwrites its own input. `encode` prints a JSON
record (`bytes`, `rate_index`, `fitted`, `within_budget`, `identity_only`,
`encode_ms`, ...), warns on stderr whenever the written container is larger than the
budget (with `--overflow floor`, or with `--paper-compat` at 96/168 px, where the
paper accounting does not reserve the 4-byte geometry trailer) or is the
identity-only fallback, and accepts `--paper-compat`,
`--overflow {identity,floor,error}`, `--device` and `--weights-dir`. `decode` reads
the variant from the header and reports the model loading time (`load_ms`)
separately from the decode time (`decode_ms`); identity-only containers are decoded
without loading a model. The CLI times one cold call per process (CUDA context
creation and the first entropy-coder table construction), so its `encode_ms` /
`decode_ms` are much larger than the warm per-crop times in
[Performance](#performance). Missing or unreadable files, an output path equal to
the input, malformed containers and impossible budgets end with an `error:` line and
exit code 1 (`-v` shows the traceback). The installed package also provides the same
interface as `face1kb-codec`.

## Performance

Numbers from the paper (arXiv v1); the table labels refer to the paper source.

**Identity and fidelity** (paper Tables `tab:codec-results` at 224 px and
`tab:codec-results-112` at 112 px): median identity cosine between each crop and its
reconstruction over *n* = 64 held-out crops, and native-resolution PSNR (dB). The
identity cosines of these two tables were measured with a proprietary face matcher that
is not distributed; they are reported as published and cannot be regenerated with the
public code.

| Res. | Dataset | Codec | id-cos 1024 B | PSNR 1024 B | id-cos 512 B | PSNR 512 B |
|---|---|---|---:|---:|---:|---:|
| 224 | Color FERET | Ours-ACCURATE | 0.947 | 28.6 | 0.797 | 26.6 |
| 224 | Color FERET | Ours-FAST | 0.942 | 29.6 | 0.823 | 27.7 |
| 224 | Color FERET | JPEG-AI (best codec) | 0.958 | 33.2 | 0.865 | 30.9 |
| 224 | AI-Solutions-KK | Ours-ACCURATE | 0.934 | 27.6 | 0.776 | 25.0 |
| 224 | AI-Solutions-KK | Ours-FAST | 0.941 | 28.5 | 0.852 | 26.6 |
| 112 | Color FERET | Ours-ACCURATE (best codec) | 0.948 | 28.6 | 0.910 | 27.6 |
| 112 | Color FERET | Ours-FAST | 0.859 | 27.2 | 0.783 | 26.5 |
| 112 | AI-Solutions-KK | Ours-ACCURATE (best codec) | 0.938 | 27.3 | 0.874 | 25.8 |
| 112 | AI-Solutions-KK | Ours-FAST | 0.844 | 26.0 | 0.757 | 24.8 |

**Verification** at 112 px on public matchers:

* EER (%) on the held-out `cvlface_ir101` matcher (paper Table `tab:heldout-cvlface`):
  Ours-ACCURATE 0.05 / 0.08 (Color FERET, 1024 / 512 B) and 0.51 / 0.92
  (AI-Solutions-KK), the best codec in three of the four columns; Ours-FAST 0.15 /
  0.32 and 1.21 / 2.40.
* Mean FNMR at FMR = 10<sup>-4</sup> over ArcFace and LVFace-L (%, paper Table
  `tab:frr-summary`), CF/1024, CF/512, KK/1024, KK/512: Ours-ACCURATE 1.26, 1.83,
  2.93, 6.93 (best at 512 B on both datasets); Ours-FAST 3.82, 6.65, 10.55, 24.30.
* Worst-case identity tail at 512 B (5th percentile of the per-image ArcFace cosine,
  paper Table `tab:idcos-tail`): Ours-ACCURATE 0.802 (Color FERET) / 0.763
  (AI-Solutions-KK), the tightest tail of the compared codecs; Ours-FAST 0.636 / 0.591.

**Speed** (paper Table `tab:speed`, 112 px / 1024 B, single-shot encode at the
budget-hitting gain, i.e. without the rate search, median per crop): Ours-FAST 37.9 ms
encode / 52.0 ms decode on one CPU core and 22.4 / 22.9 ms on an NVIDIA RTX 2080 Ti;
Ours-ACCURATE 262 / 365 ms on CPU and 54.0 / 63.1 ms on the GPU.

The full encode of `Codec.encode` includes the binary rate search (about 6 encodes).
Measured with this package on one RTX 2080 Ti (warm per-gain cache, median of 55
Color FERET crops per cell, all five resolutions and both budgets):

| | encode incl. rate search | decode |
|---|---:|---:|
| face1kb-FAST | 129-148 ms | 22-28 ms |
| face1kb-ACCURATE | 343-412 ms | 34-46 ms |

The first encodes of a new instance are slower while the per-gain entropy-coder tables
are built.

## Reproducing the paper bitstreams

Use CUDA and the pinned software stack (`requirements/paper.txt`, or at least
`compressai==1.2.8`, `timm==1.0.24`, torch 2.10) and `paper_compat=True`:

```python
data = codec.encode(img, budget, paper_compat=True)   # overflow="floor" implied
```

Verified for this release (NVIDIA RTX 2080 Ti, sm_75): re-encoding the first 24
images (canonical index order) of every cell -- both codecs, Color FERET and
AI-Solutions-KK, 64/96/112/168/224 px, 1024 and 512 B, 960 encodes -- with the
released safetensors gave **960/960 byte-identical** containers; decoding the stored
paper containers matched the decoded images of the paper pipeline exactly (maximum
absolute difference 0, 960/960). With the default `paper_compat=False` all 960
containers were within budget and the bytes at 64/112/224 px were unchanged except
where the paper stored an over-budget floor stream (then the default emits the
identity-only container; `overflow="floor"` restores the paper bytes).

`tests/codec/` contains the corresponding checks: `test_golden.py` (re-encode stored
paper bitstreams; needs the data; and fixed synthetic golden vectors recorded on
sm_75), `test_decode.py` (decode vs. cached decoded images), `test_budget_real.py`
(`len <= budget` at all five resolutions), `test_weights.py` (strict loading,
parameter counts, hashes).

## Limitations

* **Bitstreams are device-specific.** The entropy-coder CDF tables are rebuilt at run
  time from the learned parameters, and CPU and CUDA builds differ: CUDA-encoded
  streams (including all paper streams) do not decode correctly on CPU (the output is
  garbage), and CPU encodes differ from CUDA encodes. CPU-to-CPU round trips are
  self-consistent. Exchange streams only between CUDA machines with the pinned
  software stack; exactness was verified on Turing (sm_75) GPUs only. On other GPU
  generations run `pytest tests/codec/test_golden.py -k synthetic` first.
* **The identity side-channel carries no information.** With the released ACCURATE
  weights the gain of the side-stream projection's LayerNorm has collapsed (mean
  absolute value about 4e-4), so the projected code equals the LayerNorm bias up to
  about 1e-3 and rounds to the same integers for every input. Every stored ACCURATE
  container of the paper carries the same 8-byte side-channel (`f82a5c8500000000`),
  and fresh encodes of different subjects give the same bytes; 8 bytes is the minimum
  output of the rANS coder for this practically zero-entropy code. The decoded code
  acts as a constant learned conditioning of the FiLM stages and the refinement head.
  This is consistent with the side-stream ablation of the paper (`subsec:codec-side`),
  which found no identity gain on matchers independent of the EdgeFace anchor.
* **Privacy.** The stored side-channel bytes are image-independent and therefore not
  linkable. The spatial latent, however, encodes the face itself, so containers are
  biometric data and must be protected like face images.
* **Rate floors.** FAST cannot reach 512 B for most 168 and 224 px crops of
  in-the-wild faces (at 512 B, 9 % / 14 % of the AI-Solutions-KK and 50 % / 80 % of
  the Color FERET paper containers fit at 168 / 224 px; see the fit table). With the
  defaults those crops become identity-only containers of 3 (224 px) or 7 (168 px)
  bytes that decode to black frames, with an `IdentityOnlyWarning`;
  `overflow="floor"` returns the smallest (over-budget) spatial stream instead and
  `overflow="error"` raises. ACCURATE reaches such a floor for only 56 of its 288,690
  paper containers (at 96 px, and at 224 px on AI-Solutions-KK); with the defaults
  they become identity-only containers of 15 or 11 bytes.
* **Inputs.** Square, aligned face crops only; trained at 64-256 px. EdgeFace-based
  identity is only as good as the alignment.
* **Not thread-safe**, see [API](#api).
* **Training** reproduces the recipe, not the exact weights ([training.md](training.md)).

## Differences between the paper text and the code

The code is the reference; these statements of the paper (arXiv v1) differ from it:

* The paper describes the ACCURATE side-channel as about 90-175 B and derives a
  budget-reallocation estimate from it; every stored ACCURATE container has an 8-byte
  side-channel, and it is the same 8 bytes in all of them.
* The paper calls the side-channel a "hard identity floor" that survives
  spatial-latent collapse and "backs the identity-only fallback frame". With the
  released weights the stored code is image-independent, so it carries no identity,
  and an identity-only container decodes to a black frame (the decoder does not use
  the side-channel without a spatial latent).
* The paper's privacy paragraph ("The stored side-code is a linkable biometric")
  measured linkability on the unquantised 128-D projection, which is never stored.
  The stored, entropy-coded code is identical for all images and is not linkable.
* The paper attributes the 96/168 px budget misses to rate floors. The misses of
  ACCURATE (99.7 %) and those of FAST at 96 px and at 168 px / 1024 B are the 4-byte
  raw-geometry trailer that the paper encoder did not reserve (`paper_compat`); FAST
  at 168 px / 512 B is mostly a genuine rate floor. Over all 96/168 px misses, 65 %
  are trailer overruns (see [Budget](#hard-byte-budget)).
* The paper calls the rate gain a "learned gain vector". In the code the 8-level
  `Gain` vector is never trained: CompressAI's `MeanScaleHyperpriorVbr` detaches it
  in the training forward, so it stays at its initialisation (FAST: CompressAI's
  default 0.1-1.0; ACCURATE: log-spaced 0.04-1.0), which is what both weight files
  store. The network is trained at those fixed gains.

## Licence

Code: MIT (repository `LICENSE`). Weights: CC BY-NC-SA 4.0 ([`weights/LICENSE`](../weights/LICENSE));
the codecs were trained on WebFace42M, used for research purposes only, and the
ACCURATE file bundles the EdgeFace-S model of the Idiap Research Institute (CC BY-NC-SA
4.0). The vendored EdgeFace architecture code (`face1kb/third_party/edgeface`) is
BSD-3-Clause. See [`weights/README.md`](../weights/README.md) for attribution and the
model card.
