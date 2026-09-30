# Face-recognition evaluators (`face1kb.fr`)

The benchmark measures identity preservation with a roster of 14 publicly released
face-recognition (FR) models. `face1kb.fr` wraps all of them behind one interface.
Every evaluator takes aligned **112 x 112 RGB `uint8`** crops and returns an `(N, 512)`
`float32` array of **raw** embeddings. The embeddings are not L2-normalised, so
normalise them when you score. Each family applies its own preprocessing internally.

```python
from face1kb import fr

model = fr.load(fr.ARCFACE, device="cuda")   # or "cpu", "cuda:1"; default: cuda if available
emb = model.embed(crops)                     # list of (112, 112, 3) uint8 RGB, or an (N, 112, 112, 3) array
model.model                                  # the torch nn.Module (torch families only)
```

**Weights are not part of face1kb and are not redistributed.** They are downloaded
from the official sources into `FACE1KB_MODELS_ROOT` (default `<repo>/models`) and
verified against pinned SHA-256 values. Each set of weights is released by its
authors under their own terms, and most allow **non-commercial research use only**.
You are responsible for complying with those terms (see the licence column below).

## Roster

`fr.ROSTER` lists the models in the paper's display order. `fr.ANCHORS` holds the four
anchor matchers of the headline results, `(ARCFACE, LVFACE_L, TOPOFR_R100, EDGEFACE_XS)`.
`fr.HELDOUT = "cvlface_ir101"` is the held-out verification matcher: it is neither an
anchor nor used in codec training.

| name | label | family / runtime | official source (pinned) | training data | weights licence (as stated by the authors) | download |
|---|---|---|---|---|---|---|
| `arcface_antelopev2` | ArcFace R100 | insightface ONNX | [deepinsight/insightface](https://github.com/deepinsight/insightface) release v0.7, `antelopev2.zip` | Glint360K | non-commercial research purposes only | 344 MiB zip (249 MiB kept) |
| `topofr_r50` | TopoFR-R50 | torch | [DanJun6737/TopoFR](https://github.com/DanJun6737/TopoFR) code @ `5960a332`, weights on Google Drive | Glint360K | no licence stated (Glint360K: research only) | 870 MiB |
| `topofr_r100` | TopoFR-R100 | torch | same | Glint360K | no licence stated | 953 MiB |
| `topofr_r200` | TopoFR-R200 | torch | same | Glint360K | no licence stated | 1158 MiB |
| `lvface_t` | LVFace-T | ONNX | [bytedance-research/LVFace](https://huggingface.co/bytedance-research/LVFace) @ `b12702ab` | Glint360K | non-commercial research purposes only (LVFace README; the MIT tag covers the code) | 73 MiB |
| `lvface_s` | LVFace-S | ONNX | same | Glint360K | same | 290 MiB |
| `lvface_b` | LVFace-B | ONNX | same | Glint360K | same | 434 MiB |
| `lvface_l` | LVFace-L | ONNX | same | Glint360K | same | 976 MiB |
| `cvlface_vit_b` | CVLface ViT-B | torch (HF snapshot code) | [minchul/cvlface_adaface_vit_base_kprpe_webface12m](https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface12m) @ `daefd501` | WebFace12M | no explicit licence; the card asks users to follow the training-data licence (WebFace: non-commercial research) | 439 MiB |
| `cvlface_ir101` | CVLface IR-101 | torch (HF snapshot code) | [minchul/cvlface_adaface_ir101_webface12m](https://huggingface.co/minchul/cvlface_adaface_ir101_webface12m) @ `54f602a0` | WebFace12M | same | 249 MiB |
| `edgeface_xxs` | EdgeFace-XXS | torch | [Idiap/EdgeFace-XXS](https://huggingface.co/Idiap/EdgeFace-XXS) @ `e710f464` | WebFace260M subsets | CC BY-NC-SA 4.0 (Idiap) | 5 MiB |
| `edgeface_xs` | EdgeFace-XS | torch | [Idiap/EdgeFace-XS-GAMMA](https://huggingface.co/Idiap/EdgeFace-XS-GAMMA) @ `735c1b59` | WebFace260M subsets | CC BY-NC-SA 4.0 (Idiap) | 7 MiB |
| `edgeface_s` | EdgeFace-S | torch | [Idiap/EdgeFace-S-GAMMA](https://huggingface.co/Idiap/EdgeFace-S-GAMMA) @ `a1c17103` | WebFace260M subsets | CC BY-NC-SA 4.0 (Idiap) | 14 MiB |
| `edgeface_base` | EdgeFace-Base | torch | [Idiap/EdgeFace-Base](https://huggingface.co/Idiap/EdgeFace-Base) @ `2944a63d` | WebFace260M subsets | CC BY-NC-SA 4.0 (Idiap) | 70 MiB |

All 14 models together take about 5.7 GiB. The code licences are MIT for insightface,
LVFace and CVLface. EdgeFace is BSD-3-Clause, and its two architecture files are
vendored in `face1kb/third_party/edgeface` with their licence. The TopoFR repository has
no licence file: its four `backbones/*.py` files are downloaded from the pinned commit
at setup and never redistributed.

### Pinned weight files (SHA-256)

| file under `FACE1KB_MODELS_ROOT` | SHA-256 |
|---|---|
| `insightface/antelopev2/glintr100.onnx` (from `antelopev2.zip`, zip `8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5`) | `4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf` |
| `topofr/Glint360K_R50_TopoFR_9727.pt` | `15f07d919bd0b1443a481fc4e39f0689e773881836da7e959eca9ddb09e31310` |
| `topofr/Glint360K_R100_TopoFR_9760.pt` | `f92e5e61b326495d32803156ab60aa58abf7e17f84e83d38b5ff6bf1691bdbfe` |
| `topofr/Glint360K_R200_TopoFR_9784.pt` | `bf0fda461721273279f39688f834c81d4ec5e4b3465cd8946d324f555b025111` |
| `lvface/LVFace-T_Glint360K.onnx` | `bf8da0e1e93c432d9a1d874a9ba0990f5859f970e8864b3990f2f33d11f9cdb3` |
| `lvface/LVFace-S_Glint360K.onnx` | `cd09f27c82ce0a3633fb8b1966d779a7171b23aa4f14ca0de6edf9677573d119` |
| `lvface/LVFace-B_Glint360K.onnx` | `9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382` |
| `lvface/LVFace-L_Glint360K.onnx` | `49389036a4a5b69e0efcddfe34839ac72c7a71ce6b4dc1b6821e2ac368c87063` |
| `cvlface/cvlface_adaface_vit_base_kprpe_webface12m/pretrained_model/model.pt` | `04b4bee1de7cefa9e97900f8449fca906d8afbab2029bd39cc5049d33e927ed9` |
| `cvlface/cvlface_adaface_ir101_webface12m/pretrained_model/model.pt` | `e312d79222d28027f146ed495e182e48fe0dddf404cdbbdabcccdbdd07cc3758` |
| `edgeface/edgeface_xxs.pt` | `5b6bac48ea660aca185be8a8ac934db2093b51259461721da53c468230bc24c9` |
| `edgeface/edgeface_xs_gamma_06.pt` | `5ae7504cd9aee0a5d52c2115fd2eb66b0985dd1730f40134b5854e0cb658ce16` |
| `edgeface/edgeface_s_gamma_05.pt` | `dc59abda2e8580399fd115a1eeb07e1f21156196db604b884407bcf0f17efb07` |
| `edgeface/edgeface_base.pt` | `95861c09b22810136f43ec98845e7f09bfc3c43f5a804984a7bd2eac20abc30c` |

Every other file is also pinned: the CVLface snapshot code, configs and `README.md`, and
the TopoFR code. The full list is in `face1kb/fr/sources.py`. `python scripts/fetch_models.py --list`
prints it.

## Fetching the models

```bash
pip install -e ".[eval]"                          # onnxruntime-gpu, insightface, transformers, gdown, ...
python scripts/fetch_models.py                    # all 14 evaluators (about 5.7 GiB)
python scripts/fetch_models.py --only anchors     # the 4 anchors
python scripts/fetch_models.py --only training    # EdgeFace-XS and -S, used by the codec training
python scripts/fetch_models.py --only lvface_l,topofr   # model names and/or groups
python scripts/fetch_models.py --list             # files, sizes, hashes; no download
python scripts/fetch_models.py --check            # verify what is there; no download
python scripts/fetch_models.py --force            # re-download files that fail verification
```

The groups are `all`, `anchors`, `training`, `heldout` and the family names `lvface`,
`arcface`, `cvlface`, `edgeface` and `topofr`. Before a model's files are fetched, its
licence notice is printed. Each file is downloaded to a temporary file next to its
destination, checked against its SHA-256, and only then moved into place. Valid files
that are already present are kept, so the script can be re-run at any time. Pass
`--models-root DIR`, or set `FACE1KB_MODELS_ROOT`, to put the files elsewhere.

`fr.load()` also fetches a missing file on first use. Pass `download=False` to forbid
that, which raises `FileNotFoundError` instead. The SHA-256 of every file is checked
once per process when the model is loaded. Pass `verify=False` to skip the check.

**TopoFR weights (Google Drive).** The three checkpoints are fetched from Google Drive
with `gdown`. If that fails (download quota, network, `gdown` missing), the script
prints a manual-download instruction for each file and exits with status 1. Download
the file from the given `https://drive.google.com/file/d/<id>/view` page and save it
under its original name at the printed path, for example
`models/topofr/Glint360K_R100_TopoFR_9760.pt`. Then run the script again to verify it.

Resulting layout:

```text
models/
  lvface/LVFace-{T,S,B,L}_Glint360K.onnx
  insightface/antelopev2/glintr100.onnx
  cvlface/cvlface_adaface_ir101_webface12m/{wrapper.py,models/,pretrained_model/}
  cvlface/cvlface_adaface_vit_base_kprpe_webface12m/{wrapper.py,models/,pretrained_model/}
  edgeface/edgeface_{xxs,xs_gamma_06,s_gamma_05,base}.pt
  topofr/src-5960a3322096/backbones/{__init__,iresnet,iresnet2060,mobilefacenet}.py
  topofr/Glint360K_R{50,100,200}_TopoFR_*.pt
```

The EdgeFace files are shared with the codec training
(`face1kb.codec.identity_loss`), which pins and fetches them itself.

## Preprocessing per family

| family | preprocessing (input: aligned 112 px RGB `uint8`) |
|---|---|
| LVFace | RGB, `(x / 255 - 0.5) / 0.5`, NCHW, ONNX Runtime (CUDA EP) |
| ArcFace | RGB flipped to BGR, then insightface `ArcFaceONNX.get_feat` (swaps back to RGB and applies its own mean/std) |
| CVLface | RGB, `(x / 255 - 0.5) / 0.5`, NCHW. ViT-B also receives the fixed ArcFace 5-point template divided by 112 (`fr.ARCFACE_TEMPLATE_112`) as the keypoints of every crop |
| EdgeFace | RGB, `(x / 255 - 0.5) / 0.5`, NCHW |
| TopoFR | RGB, `(x / 255 - 0.5) / 0.5`, NCHW, `forward(x, phase="infer")` |

Crops at another resolution must be resized to 112 px first. The experiments use
bilinear resizing (`cv2.INTER_LINEAR`).

### Family-specific notes

* **ArcFace** is loaded by explicit path (`glintr100.onnx`, the ResNet-100 of the
  `antelopev2` pack) through `insightface.model_zoo.get_model`. There is no fallback to
  any other insightface model, such as `buffalo_l`/`w600k_r50`: a missing or modified
  file is an error.
* **CVLface** snapshots run their own `wrapper.py`, and each ships a top-level Python
  package named `models`. As a result, **only one CVLface model can be loaded per
  process**. Loading the other one raises `RuntimeError`, and so does loading either
  one when another package called `models` has already been imported. Load each in its
  own process, as the embedding stage does with one worker per model.
* **CVLface ViT-B keypoints.** The model is fed the fixed template keypoints instead of
  the snapshot's own landmark aligner (`aligner.pt`, which is not downloaded). All
  inputs are already aligned to that template. This is how the paper's numbers were
  produced, and it deviates from the model card's default pipeline.
* **CVLface ViT-B `rpe_index_cpp`.** KP-RPE optionally uses a compiled
  `rpe_index_cpp` op. When the op is not importable, the snapshot would try to build
  and install it (`python setup.py install --user`) on first import and then call
  `sys.exit()`. The loader disables that `sys.exit()` and the build attempt, and falls
  back to the snapshot's pure-PyTorch gather. The fallback gives bit-identical embeddings; it only costs
  speed. To build the op, run `python setup.py install` in
  `models/cvlface/cvlface_adaface_vit_base_kprpe_webface12m/models/vit_kprpe/RPE/rpe_ops`
  (this needs a C++ compiler, and the CUDA toolkit for the GPU kernel). Alternatively, construct
  `face1kb.fr.families.CVLfaceEmbedder(..., build_rpe_ops=True)`.
* **TopoFR** code is imported under the private module name `_face1kb_topofr_backbones`.
  TopoFR and the upstream EdgeFace repository both use the top-level name `backbones`,
  so a unique name lets the two coexist in one process.

## Your own matcher

The identity-cosine ("id-cos") columns of the paper's codec-comparison tables were
computed with a **proprietary matcher that is not distributed**. Those numbers are
shipped as published results but cannot be regenerated with public code. The public
default for identity-cosine scores is `fr.HELDOUT` (`cvlface_ir101`), or you can plug in
any model of your own:

```python
from face1kb import fr

# an ONNX model: preprocessing is declarative
fr.register_onnx(
    "my_matcher", "/path/to/model.onnx",
    input_name=None,          # default: the only input
    color="RGB",              # channel order the model expects: "RGB" | "BGR"
    normalize="[0,1]",        # "[-1,1]" | "[0,1]" | "none" | (mean, std) in pixels | callable
    size=112,                 # resized bilinearly if not 112
    sha256=None,              # optional pinned hash, checked at load
)
emb = fr.load("my_matcher").embed(crops)

# a torch network (the factory returns the nn.Module; it gets the device if it
# has a required positional parameter)
fr.register_torch("my_torch", lambda: build_my_net(), normalize="[-1,1]")

# anything with .embed(crops) -> (N, D) float32
fr.register_embedder("my_api", lambda device: MyEmbedder(device))
```

ONNX models whose batch dimension is fixed are run in chunks of that size. The last
chunk is padded, and the outputs of the padding are dropped. Built-in names cannot be
overwritten.

A registration lives in the process that made it. To use your model with the
experiment scripts, which may start worker processes, put the registration in a
plugin module and name it in `FACE1KB_FR_PLUGINS` (comma-separated module names or
`.py` paths). `fr.load()` and `fr.available()` import the plugins on first use, in
every process:

```python
# my_matchers.py
from face1kb import fr

fr.register_onnx("my_matcher", "/path/to/model.onnx", color="RGB", normalize="[0,1]")
```

```bash
export FACE1KB_FR_PLUGINS=/path/to/my_matchers.py
python -m face1kb.fr list            # my_matcher is listed as a user model
```

The model name can then be passed wherever the experiments accept a matcher name.

To use TopoFR in the codec training, as the identity loss or the side-stream anchor,
register its builder with the codec:

```python
from face1kb import fr
from face1kb.codec.identity_loss import register_torch_fr
register_torch_fr("topofr_r100", "topofr", fr.torch_builder("topofr_r100"))
```

## Command line

```bash
python -m face1kb.fr list                          # registered models, files present?
python -m face1kb.fr verify [MODEL ...]            # SHA-256 of the files
python -m face1kb.fr validate --model lvface_l --images a1.png a2.png b.png
```

`validate` embeds two crops of one subject and one crop of another subject. It prints
the self, mate and impostor cosines, and reports `OK` when the mate cosine is higher
than the impostor cosine.

## Numerical reproduction and cost

`face1kb.fr` reproduces the embedding arrays behind the paper's results. The 14
models downloaded by `scripts/fetch_models.py` from their official sources are
byte-identical to the files the paper used. Embedding the first 256 crops (index
order) of Color FERET and AI-Solutions-KK `aligned_112` and comparing them with the
paper's `aligned_112` arrays gives:

* in batches of 256 (the default batch size of the embedding stage), all 256/256
  rows are **bit-identical** (max abs diff 0) for every model on both datasets;
* in batches of 64 the first 64 rows differ only by float noise: max abs diff at most
  1.5e-5 for ArcFace and CVLface IR-101, whose raw embeddings have norms of about
  20-23, and at most 3e-6 for the other models, with every row cosine above
  0.99999999999. Repeated calls with the same batch are deterministic;
* the pure-PyTorch KP-RPE fallback of CVLface ViT-B gives the same bits as the
  compiled op.

The float noise comes from kernel selection in cuDNN, cuBLAS and ONNX Runtime, which
depends on the batch size, the GPU model and the library versions. A bit-exact
comparison therefore needs all three to match the run that produced an array. The
paper's other arrays (compressed and attacked images) were not all computed with the
same settings. Depending on the image set they were embedded in batches of 256, 128 or
64, and match bit for bit only when re-embedded with that batch size; a few were
computed with an older software stack or on a different GPU model, and no batch size
reproduces them exactly. Against those arrays the differences are float noise (at most
1.8e-5 abs, row cosine at least 0.9999998).

One exception is larger: the paper's EdgeFace-S arrays of the Ours-ACCURATE outputs
(tags `ours_accurate_*`) differ from a regeneration on an RTX 2080 Ti on 9-17% of the
rows by more than 1e-5 relative, at every batch size, with a maximum of about 8e-4
relative; every row cosine stays above 0.9999997. The Ours-FAST arrays (`ours_fast_*`)
reproduce like the other arrays (float noise, at most 1e-6 relative), and the other
evaluators reproduce the Ours-ACCURATE images as described above (EdgeFace-XS bit for
bit). The cause is not established: a live Ours-ACCURATE decode on an RTX 2080 Ti
equals the stored decoded images pixel for pixel, and embedding either gives the same
difference. The effect on EER is at float-noise level.

When you compare your embeddings with an existing array, use a tolerance, for example
a row cosine of at least 0.99999 (or a relative difference of at most 1e-5; 1e-3
for the EdgeFace-S arrays of the Ours-ACCURATE outputs), or probe the batch sizes 256,
128 and 64, rather than requiring exact equality at one batch size.

On machines with mixed GPU models, CUDA numbers devices fastest first by default,
which can differ from the `nvidia-smi` order. Set `CUDA_DEVICE_ORDER=PCI_BUS_ID`
before choosing a GPU with `CUDA_VISIBLE_DEVICES`.

The built-in loaders prepare the input tensors with the same memory layout as the
paper's embedding stage (EdgeFace and TopoFR receive a channels-last view, CVLface a
contiguous tensor), because the layout changes the cuDNN kernel selection: feeding
TopoFR-R100 a contiguous copy instead moves its embeddings by about 2e-7. Models
registered with `register_torch` receive contiguous tensors.

Setup: RTX 2080 Ti, CUDA 12.8, torch 2.10.0, onnxruntime-gpu 1.23.2, insightface 1.0.1,
timm 1.0.24, transformers 5.12.1. On other GPUs, or with other library versions,
expect float-level differences. `tests/fr/test_golden.py` checks an installation
against reference embeddings of two synthetic crops, with a tolerance of 1e-3. On the
same machine, CPU and GPU embeddings differ by at most 1.6e-5 in absolute value
(at most 4e-6 relative to the embedding norm).

Loading a model takes 6-21 s, most of it hashing and reading the weights. Every model
fits in 11 GB of GPU memory at batch 256. Embedding is cheap compared with decoding the
compressed images.

## Tests

```bash
pytest tests/fr        # unit tests: registry, hooks (tiny ONNX graphs), downloads, sources
FACE1KB_MODELS_ROOT=... pytest tests/fr/test_golden.py     # per-model reference check
FACE1KB_MODELS_ROOT=... FACE1KB_DATA_ROOT=... FACE1KB_WORK_ROOT=... \
    pytest tests/fr/test_stored_embeddings.py              # vs your embedding arrays
```

`test_golden.py` embeds two synthetic crops (no dataset faces) with every fetched
model, each in its own process, and compares the result with reference values from
the paper environment. `test_stored_embeddings.py` (markers `gpu`, `data`) re-embeds
the first 64 aligned 112 px crops of each dataset and compares them with rows 0..63 of
`embeddings/<model>/aligned_112.npy` under `FACE1KB_WORK_ROOT` (row cosine at least
0.9999). Tests that need model files, crops or arrays skip when those are absent.
