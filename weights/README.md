# Model card: face1kb-FAST, face1kb-ACCURATE and the Li-AE proxy

This folder holds the released weights of the paper *Toward Sub-1 kB
Identity-Preserving Face Compression* (arXiv:2608.22866): the two learned face codecs,
called **Ours-FAST** and **Ours-ACCURATE** in the paper, and the small Li-AE proxy
auto-encoder of the adversarial study (see
[Li-AE proxy](#li-ae-proxy-adversarial-study)). Both codecs compress an aligned face
crop into a self-describing container of at most *B* bytes (the paper evaluates
*B* = 1024 and 512) while trying to preserve face-recognition identity.

The files are stored with git LFS. After cloning, run `git lfs pull` (or
`scripts/setup_env.sh`), otherwise the `.safetensors` files are small pointer files.

| File | Variant | Parameters | Training steps | Size (bytes) | SHA-256 |
|---|---|---:|---:|---:|---|
| `face1kb_fast.safetensors` | FAST (`variant_id` 0) | 1,352,021 | 1,000,000 | 5,418,964 | `244cb42ac0bb9f0469a28b0c975e0678671c1ad2d82f5d238595080878880e5c` |
| `face1kb_accurate.safetensors` | ACCURATE (`variant_id` 1) | 18,712,400 (15,059,880 trained + 3,652,520 frozen EdgeFace-S) | 3,000,000 | 74,910,352 | `577525fd07625c29a88b055c94f8fa867763d6cfd20c10a34e9bdc1211adda98` |
| `liae_proxy.safetensors` | Li-AE proxy (not a codec) | 1,381,443 | 60 epochs | 5,536,012 | `eac885eaf6088a5352f8a8a4350a9bbe66e15fb41666fa1d46d97a20943e7afa` |

## How to load

```python
import face1kb

codec = face1kb.load("accurate", device="cuda")    # or "fast"
data = codec.encode(img, budget=1024)              # img: HxWx3 uint8 RGB aligned crop
img_hat = codec.decode(data)                       # HxWx3 uint8 RGB
```

`face1kb.load` builds the network with its default constructor and loads the file with
`strict=True`. The files are net-only state dicts (fp32 and int32 tensors, safetensors
format) with this header metadata:

| Key | FAST | ACCURATE |
|---|---|---|
| `face1kb.format_version` | `1` | `1` |
| `face1kb.variant` | `fast` | `accurate` |
| `face1kb.architecture` | `FaceCodecFast` | `FaceCodecAccurate` |
| `face1kb.step` | `1000000` | `3000000` |
| `face1kb.parameters` | `1352021` | `18712400` |
| `face1kb.license` | `CC-BY-NC-SA-4.0` | `CC-BY-NC-SA-4.0` |
| `face1kb.requires` | `compressai==1.2.8; timm==1.0.24 (EdgeFace anchor keys)` | same |
| `face1kb.bundled` | -- | EdgeFace-S anchor attribution |
| `format` | `pt` | `pt` |

The `face1kb.requires` string is the same in both files; its timm pin matters only
for the EdgeFace anchor keys of ACCURATE (FAST contains no anchor and does not use
timm).

The entropy-coder tables (`*._quantized_cdf`, `*._cdf_length`, `*._offset`,
`gaussian_conditional.scale_table`) are stored empty and rebuilt at run time from the
learned parameters, which is why the bitstreams depend on the device and on the
`compressai` version (see *Limitations*).

## Architecture

Both variants are mean-scale hyperprior codecs (Ballé et al. 2018; Minnen et al. 2018)
with the variable-rate extension of Kamisli, Racape and Choi (DCC 2024; gain units
after Cui et al. 2021), built on CompressAI 1.2.8. The 8-level gain vector is not
trained: CompressAI detaches it in the training forward, so both files store its
initial values (FAST: CompressAI's default 0.1-1.0; ACCURATE: log-spaced 0.04-1.0),
and the network is trained at those gains. The codecs add:

* a third stride-2 stage in the hyper-analysis (mirrored in the hyper-synthesis), so
  the hyperprior latent is *x*/128 and its byte floor is small; inputs are
  edge-padded to a multiple of 128;
* a synthesis transform whose x2 upsamplings are nearest-neighbour + convolution
  (no transposed convolutions) and whose stages are FiLM-modulated by the gain and the
  crop resolution;
* a frozen, log-spaced 64-entry gain table spanning the training gain range (FAST
  0.1-1.0, ACCURATE 0.04-1.0); the encoder binary-searches it (about 6 real rANS
  encodes) for the largest gain whose container fits the budget.

| | FAST | ACCURATE |
|---|---|---|
| Channels N / M / N<sub>z</sub> | 64 / 96 / 48 | 192 / 320 / 64 |
| Attention blocks | none | 2 in analysis, 2 in synthesis |
| Identity side-stream | none | frozen EdgeFace-S anchor -> learned 128-D projection -> factorized entropy bottleneck -> FiLM of the synthesis (the stored code is constant with these weights, see *Limitations*) |
| Refinement head | none | `clamp(x_hat + 0.1 * residual)`, conditioned on the decoded identity code |
| Training identity loss | EdgeFace-XS | EdgeFace-S |

Container layout: 1 flags byte (variant id, 6-bit rate index), 1 resolution-bucket
byte, 1 side-channel length byte, the side-channel, two length-prefixed rANS strings,
and a 4-byte geometry trailer for resolutions without a bucket code (7 bytes of fixed
overhead). See `docs/codec.md`.

## Training data

The codecs were trained on **WebFace42M** (Zhu et al., CVPR 2021; 112 x 112 aligned
crops), used for research purposes only. Because of the terms of that dataset, the
weights are released under **CC BY-NC-SA 4.0 (non-commercial)**. Training draws
uniformly random crops (no augmentation), random gain levels and random resolution
buckets (FAST: 64, 128, 192, 224, 256 px with weights 1:4:3:3:5; ACCURATE: 64, 128,
192, 256 px uniform). The evaluation datasets of the paper (Color FERET,
AI-Solutions-KK) were never used for training.

The released weights come from warm-start chains: FAST continues a 300k-step run for
1M steps with the re-balanced resolution buckets; ACCURATE continues a 60k-step run
on a 1M-step schedule, stopped at 990k steps, and is then extended to 3M steps on a
rebuilt cosine schedule. Some
early links of these chains ran on earlier revisions of the training code, so
`python -m face1kb.codec.train` reproduces the recipe, not these exact weights (see
`docs/training.md`).

## Bundled third-party weights: EdgeFace-S

`face1kb_accurate.safetensors` contains, under the `side.anchor.*` keys (257 tensors,
3,652,520 parameters), a bit-identical copy of the EdgeFace-S (gamma = 0.5) face
recognition model `edgeface_s_gamma_05.pt` (SHA-256
`dc59abda2e8580399fd115a1eeb07e1f21156196db604b884407bcf0f17efb07`), which is the
frozen identity anchor of the ACCURATE encoder:

* (c) 2024 Anjith George, Christophe Ecabert, Hatef Otroshi Shahreza, Ketan Kotwal,
  Sébastien Marcel, Idiap Research Institute, Martigny, Switzerland;
* licence: CC BY-NC-SA 4.0 (<https://huggingface.co/Idiap/EdgeFace-S-GAMMA>);
* the EdgeFace architecture code is vendored in `face1kb/third_party/edgeface/`
  (BSD-3-Clause).

```bibtex
@article{edgeface,
  title={Edgeface: Efficient face recognition model for edge devices},
  author={George, Anjith and Ecabert, Christophe and Shahreza, Hatef Otroshi and Kotwal, Ketan and Marcel, Sebastien},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2024}
}
```

## Intended use

* Research on compact face storage, e.g. face images embedded in 2D barcodes of
  identity documents, and benchmarking of codecs under a hard byte budget.
* Reproduction of the paper results.

Inputs are square RGB face crops aligned to the ArcFace 5-point template; the codecs
were trained at 64-256 px and evaluated at 64, 96, 112, 168 and 224 px. Other inputs
(unaligned photos, non-face images) are out of scope. The weights are not licensed
for commercial use.

## Limitations

* **Bitstreams are device-specific.** The CDF tables of the entropy coder are rebuilt
  at run time and differ between CPU and CUDA; CUDA-encoded streams (including every
  stream of the paper) do not decode correctly on CPU, and CPU-encoded streams differ
  from CUDA-encoded ones. Byte-exact reproduction was verified on CUDA (NVIDIA Turing,
  torch 2.10, compressai 1.2.8, timm 1.0.24); other GPU generations should be checked
  against the golden vectors in `tests/codec/` before exchanging streams.
* **The identity side-channel is constant.** In the ACCURATE weights the gain of the
  side-stream projection's LayerNorm has collapsed (mean absolute value about 4e-4),
  so the projected identity code equals the LayerNorm bias up to about 1e-3 and is
  entropy-coded to the same 8 bytes (`f82a5c8500000000` on CUDA) for every image; every
  stored ACCURATE container of the paper carries these bytes. The decoded code acts as
  a constant learned conditioning of the FiLM stages and the refinement head. On
  matchers independent of the EdgeFace anchor the side-stream brings no measurable
  identity gain (paper, `subsec:codec-side`).
* **Budget floors.** FAST cannot reach 512 B at 168 or 224 px for most in-the-wild
  crops (of the paper containers, 9 % / 14 % of AI-Solutions-KK and 50 % / 80 % of
  Color FERET crops fit at 168 / 224 px; `tab:fit-kk`, `tab:fit-cf`). With the
  default settings the encoder then emits an identity-only container that decodes to
  a black frame and issues an `IdentityOnlyWarning`; the paper instead stored the
  smallest (over-budget) stream (`paper_compat=True`, or `overflow="floor"`).
  ACCURATE hits such a floor for only 56 of its 288,690 paper containers (96 px, and
  224 px on AI-Solutions-KK).
* **Paper budget accounting.** The paper's encoder did not reserve the 4-byte geometry
  trailer at 96 and 168 px, so some paper containers at those resolutions exceed the
  budget: by at most 4 bytes in general and by exactly 3 bytes in every such paper
  container (the rANS strings are whole 32-bit words and the budgets are multiples of
  4). `paper_compat=False` (default) reserves the trailer.
* **Privacy.** The stored side-channel bytes are the same for every image and are not
  linkable (the paper's linkability measurement used the unquantised projection,
  which is never stored). The spatial latent encodes the face itself, so treat
  containers as biometric data.
* **Demographic effects.** Verification accuracy after compression differs across
  demographic subgroups; see the fairness section of the paper.

## Li-AE proxy (adversarial study)

`liae_proxy.safetensors` is the surrogate model of the Li-AE no-box attack (Li, Guo
and Chen, NeurIPS 2020) used in the paper section "Adversarial Robustness and
Sanitization". It is a small convolutional auto-encoder: the attack crafts
perturbations against it and never queries the evaluated face matchers. It is neither
a face codec nor a face-recognition model. The attack and the proxy are described in
[`docs/adversarial.md`](../docs/adversarial.md#the-li-ae-proxy).

| File | Parameters | Size (bytes) | SHA-256 |
|---|---:|---:|---|
| `liae_proxy.safetensors` | 1,381,443 | 5,536,012 | `eac885eaf6088a5352f8a8a4350a9bbe66e15fb41666fa1d46d97a20943e7afa` |

Load it with `face1kb.adversarial.load_proxy()` (`strict=True`);
`face1kb.adversarial.li_ae_attack` uses it by default.

* **Architecture** (`face1kb.adversarial.ProxyAE`, width 32).
  * Encoder: four `Conv2d(k4, s2, p1) -> BatchNorm2d -> LeakyReLU(0.2)` stages with
    32/64/128/256 channels (112 -> 7 px).
  * Decoder: three `ConvTranspose2d(k4, s2, p1) -> BatchNorm2d -> ReLU` stages and a
    final `ConvTranspose2d` to 3 channels with a sigmoid.
  * Input is 112 x 112 RGB in [0, 1]. The attack reads the 64 x 28 x 28 output of the
    second encoder stage and the 256 x 7 x 7 bottleneck.
  * Independent implementation; no code of github.com/qizhangli/nobox-attacks is
    used.
* **Format.** Net-only safetensors with 51 tensors: fp32 weights and batch-norm
  statistics, and int64 batch-norm step counters. Header metadata:
  `face1kb.format_version` `1`, `face1kb.architecture` `ProxyAE`, `face1kb.width`
  `32`, `face1kb.parameters` `1381443`, `face1kb.n_ids` `400`, `face1kb.n_imgs`
  `3854`, `face1kb.epochs` `60`, `face1kb.objective`, `face1kb.training_data`,
  `face1kb.license` `CC-BY-NC-SA-4.0`, `format` `pt`.
* **Training data.** 3,854 aligned 112 x 112 crops of 400 identities (at most 12 per
  identity) from a small sample of **WebFace42M** (Zhu et al., CVPR 2021), used for
  research purposes only.
  * The evaluation datasets (Color FERET, AI-Solutions-KK) were not used.
  * Whether any of the 400 identities also appear in the evaluation sets was not
    checked.
* **Training.** Prototypical objective: the decoder reconstructs the mean crop of the
  input's identity (MSE loss).
  * 60 epochs, batch 128, Adam lr 2e-3 with cosine annealing, seed 0
    (`python -m face1kb.adversarial.train`).
  * Final prototype MSE on the training crops in eval mode: 0.0064.
  * Retraining reproduces the recipe, not these weights: the sample came from an
    identity-sorted re-packing of WebFace42M, and GPU training is not
    bit-deterministic.
* **Intended use.** Reproducing the paper's Li-AE attack conditions, and research on
  the adversarial robustness of face-compression pipelines. It is not suitable as a
  face encoder or recognizer.
* **Licence.** CC BY-NC-SA 4.0, the same as the codec weights and for the same reason:
  the WebFace42M training data is research-only.

## Licence

Copyright (c) 2026 Innovatrics (Petr Hurtik, Jakub Sochor).

All weights in this folder (both codecs and the Li-AE proxy) are licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International licence; the
full legal code is in [`LICENSE`](LICENSE). The EdgeFace-S tensors bundled in
`face1kb_accurate.safetensors` remain (c) Idiap Research Institute under the same
licence (see above). The code of the repository is MIT-licensed (top-level
`LICENSE`).

## Citation

```bibtex
@article{hurtik2026sub1kb,
  title   = {Toward Sub-1 kB Identity-Preserving Face Compression: A Benchmark of Codecs,
             a Custom Learned Codec, and Studies of Resolution, Demographic Fairness,
             Recompression, and Adversarial Robustness},
  author  = {Hurtik, Petr and Sochor, Jakub},
  journal = {arXiv preprint arXiv:2608.22866},
  year    = {2026}
}
```
