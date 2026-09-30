# Adversarial robustness and sanitization

The paper's section "Adversarial Robustness and Sanitization" (arXiv:2608.22866) asks
whether compressing a face crop to 512-1024 bytes also removes an adversarial
perturbation that was added before compression. In this setting the codec acts as a
fixed input-purification step in front of the face matcher. `face1kb.adversarial`
contains the three attacks of that study, the Li-AE proxy model and its training
recipe, and the sanitization metric. The drivers that run the full study (crafting,
compression, embedding, tables) are in `experiments/adversarial/`.

- [Install](#install)
- [Threat model](#threat-model)
- [The three attacks](#the-three-attacks)
- [The Li-AE proxy](#the-li-ae-proxy)
- [Compression-as-defence protocol](#compression-as-defence-protocol)
- [Sanitization metric](#sanitization-metric)
- [Determinism and reproducibility](#determinism-and-reproducibility)
- [Cost](#cost)
- [Differences between the paper text and the code](#differences-between-the-paper-text-and-the-code)
- [Licences and provenance](#licences-and-provenance)

## Install

```bash
pip install -e ".[adversarial]"   # OpenAI CLIP (openai-clip 1.0.1), OpenCV
git lfs pull                      # weights/liae_proxy.safetensors
```

HFC needs only NumPy and OpenCV. The CLIP attack downloads the OpenAI ViT-B/32
checkpoint (about 350 MB) on first use to `FACE1KB_MODELS_ROOT/clip`. The `clip`
package checks the file's SHA-256. The Li-AE attack loads the released proxy from
`FACE1KB_WEIGHTS_DIR/liae_proxy.safetensors`.

## Threat model

The attacker is **no-box**. They have no access to the architecture, weights,
training data or query interface of the face matcher, and they never query it. A
no-box attack therefore has to exploit weaknesses that many recognition models share,
chiefly their reliance on high-frequency cues and on non-robust features.

* The attacker perturbs the **aligned 112 x 112 crop** directly.
* The perturbation is bounded by an **l-inf budget** `eps` on the [0, 1] scale. The
  paper uses `eps` in {0.03, 0.06, 0.10}, which is at most `ceil(eps * 255)` = 8, 16
  or 26 grey levels.
* The defender's matchers are the four anchor models `arcface_antelopev2`,
  `lvface_l`, `topofr_r100` and `edgeface_xs`. None of the attacks uses them.

## The three attacks

```python
from face1kb import adversarial as adv

crops = ...                                   # (N, 112, 112, 3) uint8 RGB, index order
x_hfc = adv.hfc_attack(crops, eps=0.06)                        # CPU
x_clip = adv.clip_surrogate_attack(crops, eps=0.06, device="cuda")
x_liae = adv.li_ae_attack(crops, eps=0.06, device="cuda")      # released proxy
x = adv.craft("liae", crops, 0.06, device="cuda")              # the same, by name

# reproduce a paper crop set: its random-start stream, on any device
x_liae = adv.li_ae_attack(crops, 0.06, cuda_blocks=adv.paper_cuda_blocks("kk", "liae"))
```

The attacks take `(N, H, W, 3)` uint8 RGB crops and return uint8 crops of the same
shape. The defaults are the paper settings, including `seed=0`. `craft(attack, ...)`
forwards its keyword arguments and drops the ones an attack does not use: `model`
(a loaded CLIP model) is passed only to the CLIP attack, and for HFC it also ignores
`device`, `batch_size` and `cuda_blocks=None`. One call such as
`craft(attack, crops, eps, device="cuda", model=clip_model, cuda_blocks=paper_cuda_blocks(dataset, attack))`
therefore works for all three attacks.

**HFC (training-free high-frequency attack)**, `hfc_attack(imgs, eps, tile=4,
suppress=0.5, seed=0)`. This is a simplified re-implementation, in the spatial domain,
of the high-frequency-component idea of Zhang et al. 2022 (arXiv:2203.04607). For each
crop it:

1. draws standard-normal noise on a `(H/4+1) x (W/4+1) x 3` grid and tiles each value
   into a 4 x 4 block, which gives a regionally homogeneous, repeating pattern;
2. high-passes the pattern by subtracting its Gaussian blur (`sigma = 4`) and keeps
   its sign;
3. moves the crop halfway towards its own Gaussian low-pass (`suppress = 0.5`), which
   attenuates the crop's high frequencies;
4. adds `eps * sign(pattern)`, projects the result into the `eps`-ball and onto
   [0, 1], and rounds it to uint8.

It uses no surrogate model and no data, and it runs on the CPU. The noise comes from
one `numpy.random.default_rng(seed)` generator and is drawn crop by crop, so a crop's
result depends only on its position in the input.

**CLIP surrogate (I-FGSM in CLIP feature space)**, `clip_surrogate_attack(imgs, eps,
steps=10, seed=0, batch_size=64, cuda_blocks=None)`. This is untargeted iterative
FGSM against the image encoder of OpenAI CLIP ViT-B/32 (Radford et al. 2021). It
follows the CLIP-surrogate idea of MF-CLIP (arXiv:2307.06608), without MF-CLIP's
margin fine-tuning.

* The crop is resized bilinearly to 224 px inside the differentiable graph and
  normalised with the CLIP constants.
* The loss is `1 - cos(f(adv), f(clean))`, which pushes the features away from those
  of the clean crop.
* The attack starts from a uniform random point of the `eps`-ball, because the
  gradient is exactly zero at `adv = clean`.
* It then takes 10 signed-gradient steps of `max(eps/4, 1/255)`, each followed by
  projection.

`clip.load` keeps CLIP in **fp16 on CUDA** and in fp32 on the CPU. Pass
`model=adv.load_clip(device="cuda")` to reuse one loaded model across several `eps`.

**Li-AE (prototypical auto-encoder with ILA)**, `li_ae_attack(imgs, eps, proxy=None,
steps=10, ila_steps=10, seed=0, batch_size=64, cuda_blocks=None)`. This follows the
no-box attack of Li, Guo and Chen 2020 (arXiv:2012.02525). It crafts against a small
auto-encoder proxy that the attacker trained on their own faces
([below](#the-li-ae-proxy)). Each batch goes through two stages:

1. **I-FGSM feature push.** From a random start, 10 steps maximise the squared L2
   distance between the proxy's bottleneck code (`256 x 7 x 7`) and that of the clean
   crop. The resulting displacement of the proxy's mid-layer features (`64 x 28 x 28`),
   normalised per crop, becomes the guide direction.
2. **Intermediate-Level Attack refinement** (ILA, Huang et al. 2019). From a new
   random start, 10 steps maximise the projection of the mid-layer displacement onto
   the guide.

Both stages use the step `max(eps/4, 1/255)` and project into the `eps`-ball after
every step. `LiAEAttack(proxy, device)` keeps a loaded proxy for repeated calls.

**Random starts.** CLIP and Li-AE draw one random start per batch of `batch_size`
crops (Li-AE: two, one per stage), seeded with `seed`; the global torch RNG is not
touched.

* `cuda_blocks=None` (default): a private `torch.Generator` on the attack device. It
  produces the same numbers as `torch.manual_seed(seed)` followed by the device's
  default generator, which is how the paper crops were drawn.
* `cuda_blocks=<int>`: the starts are recomputed with NumPy exactly as torch's CUDA
  `uniform_` writes them on a GPU whose random kernels launch that many blocks
  (streaming multiprocessors x max threads per SM / 256; `adv.cuda_launch_blocks()`
  gives the value of the current GPU). The values are the same on every device,
  including the CPU. `adv.paper_cuda_blocks(dataset, attack)` returns the value of
  each paper crop set ([below](#determinism-and-reproducibility)).

## The Li-AE proxy

`face1kb.adversarial.ProxyAE` is a plain convolutional auto-encoder for 112 px RGB
crops in [0, 1]:

* the encoder has four stages of `Conv2d(k4, s2, p1) -> BatchNorm -> LeakyReLU(0.2)`
  with 32, 64, 128 and 256 channels;
* the decoder mirrors it with `ConvTranspose2d(k4, s2, p1) -> BatchNorm -> ReLU` and
  ends in a sigmoid.

It has 1,381,443 parameters. The attack reads the output of the second encoder stage
(`mid`) and the bottleneck (`code`).

| File | Parameters | Size (bytes) | SHA-256 |
|---|---:|---:|---|
| `weights/liae_proxy.safetensors` | 1,381,443 | 5,536,012 | `eac885eaf6088a5352f8a8a4350a9bbe66e15fb41666fa1d46d97a20943e7afa` |

The file is a net-only safetensors state dict with 51 tensors. It holds fp32
parameters and batch-norm statistics, plus int64 batch-norm step counters.
`load_proxy()` loads it with `strict=True`. The header metadata records the
architecture, width, parameter count, training-set size (400 identities, 3,854 crops),
epochs (60) and the licence. The model card is in [`weights/README.md`](../weights/README.md).

**Training recipe** (`face1kb.adversarial.train`, the defaults):

* **Objective.** The prototypical objective in the spirit of Li et al.: every crop is
  mapped to the **mean crop of its identity**, and the loss is the MSE of the sigmoid
  output against that mean.
* **Data.** 400 identities with up to 12 crops each; identities with fewer than 4
  crops are skipped. Records are read in order, and reading stops once 400 identities
  have 12 crops. The result keeps the first 400 identities with at least 4 crops, in
  order of first appearance.
* **Optimisation.** 60 epochs of batch 128 over a seeded permutation, Adam with
  lr 2e-3 and cosine annealing over the epochs, seed 0. The whole set stays on the
  device.

The released proxy was trained on a small sample of **WebFace42M** (3,854 crops of
400 identities), used for research purposes only. The Color FERET and AI-Solutions-KK
images were not used.

```bash
# folder-per-identity dataset of aligned 112 px crops (e.g. the official WebFace42M layout)
python -m face1kb.adversarial.train --images-root /path/to/webface42m \
    --out runs/liae_proxy.safetensors [--subjects ids.txt] [--device cuda]
```

```python
from face1kb.adversarial.train import iter_image_folder, select_identities, train_proxy
from face1kb.adversarial.proxy import save_proxy
images, labels, subjects = select_identities(iter_image_folder(root))
net, history = train_proxy(images, labels)          # history: proto-MSE per epoch
save_proxy(net, "liae_proxy.safetensors", {"n_ids": len(subjects)})
```

The released proxy was selected from an identity-sorted re-packing of WebFace42M
whose record order cannot be recreated from the official release. In addition, GPU
training is not bit-deterministic with the default cuDNN settings. A retrained proxy
therefore reproduces the recipe, not the released weights, so use the released file
to reproduce the paper's Li-AE numbers. On the same 3,854 crops, the retrained and
released proxies reach the same training loss (prototype MSE 0.00636 to 0.00639 in
eval mode), and their weights differ by 13-14 % in relative L2 norm.

## Compression-as-defence protocol

1. **Craft.** Take the 112 px aligned crops in `index.csv` order. For each attack and
   `eps`, write the adversarial crops as an ordinary aligned-crop variant:
   `aligned_dir(dataset, 112, adv_suffix(attack, eps))`, for example
   `aligned_112_adv_hfc_006/<subject>/<stem>.png`. The 3-digit tag is
   `eps_tag(eps) = round(100 * eps)`. Because the variant looks like any other crop
   set, the compression and embedding pipelines consume it unchanged. To reproduce
   the paper sets, craft from the first crop of the index with `seed=0`,
   `batch_size=64` and `cuda_blocks=paper_cuda_blocks(dataset, attack)`.
2. **Compress.** Encode both the clean and the adversarial crops with every codec at
   1024 and 512 B. Ours-FAST and Ours-ACCURATE use `paper_compat=True`.
3. **Embed.** Embed the clean crops, the adversarial crops and all their decoded
   versions with the four anchor matchers. The embedding tags are `aligned_112`,
   `aligned_112_adv_<attack>_<tag>`, `<codec>_112_<B>` and
   `<codec>_112_adv_<attack>_<tag>_<B>`.
4. **Score.** Compute `sanitization_record` for each (dataset, matcher, attack,
   `eps`, codec, budget) cell, then average the residual over the four matchers.

Scope of the paper run (subsets are the first *n* crops in index order):

| Dataset | Attack | Crops | Codecs | Matchers |
|---|---|---:|---|---|
| Color FERET | HFC | 11,335 | JPEG, JPEG 2000, WebP, JPEG XL, AVIF, HEIF, Ours-FAST, Ours-ACCURATE | 4 |
| Color FERET | HFC | 300 | JPEG-FzT | 4 |
| Color FERET | HFC | 300 | JPEG-AI, bmshj2018, mbt2018 | 2 (`arcface_antelopev2`, `lvface_l`) |
| Color FERET | Li-AE | 11,335 | the 7 classical codecs incl. JPEG-FzT; Ours-FAST and Ours-ACCURATE at `eps` = 0.06 only | 4 |
| Color FERET | CLIP | 2,000 | the 7 classical codecs incl. JPEG-FzT, Ours-FAST, Ours-ACCURATE | 4 |
| AI-Solutions-KK | HFC, CLIP, Li-AE | 2,000 | the 7 classical codecs incl. JPEG-FzT, Ours-FAST, Ours-ACCURATE | 4 |

On AI-Solutions-KK, JPEG XL did not produce one of the 2,000 crops in some cells, so
those cells have `n` = 1,999.

## Sanitization metric

For one cell (matcher, attack, `eps`, codec, budget), each crop gives three cosines
to the embedding of its clean, uncompressed crop:

* `a = cos(clean, adv)` is the attack strength before compression (lower means a
  stronger attack);
* `b = cos(clean, codec(clean))` is the codec's own identity cost;
* `c = cos(clean, codec(adv))` is the identity that survives both the attack and the
  codec.

`sanitization_record(clean, adv, clean_comp, adv_comp)` returns their means
(`cos_adv`, `cos_clean_comp`, `cos_adv_comp`) and two derived values:

* `residual = b - c`, the paper's headline metric. It is about 0 when the codec
  removed the perturbation and large when the perturbation survived.
* `sanitization = (c - a) / (b - a)`, the fraction of the attack's gap that the codec
  recovers. It is NaN when `b = a`.

All three means are taken over the crops for which all four embeddings exist, and
`n` counts those crops. An all-zero or non-finite embedding row marks a crop that a
codec did not produce. This keeps a codec measured on a subset comparable with
itself. The paper tables report the four-matcher mean of `residual`, or the
two-matcher mean where only two matchers were run. A double dagger marks reduced
coverage: JPEG-AI, bmshj2018 and mbt2018 were measured with two matchers, and these
three and JPEG-FzT were attacked on a 300-crop subset of Color FERET (HFC only).
Recomputed from the study's embeddings (which are not distributed), the metric
reproduces all 1,300 populated cells of the shipped `sanitization.csv` (four anchors,
both datasets, three attacks, three `eps`, 12 codecs, two budgets): `n` is
identical, and the cosines and the residual differ by at most 1.1e-16 (the last
binary digit).

## Determinism and reproducibility

* **HFC is bit-exact.** It is deterministic on the CPU. The crops it produces with
  OpenCV 4.13 and NumPy 2.4 are byte-identical to the paper crops: 48 of 48 per
  dataset and `eps` on Color FERET and AI-Solutions-KK. Another OpenCV build may
  change a few pixels through its Gaussian-blur implementation. A resolved
  `pip install -e ".[adversarial]"` gets NumPy 1.26 and OpenCV 4.11 (compressai
  declares `numpy<2`), on which bit-exactness is not established; to reproduce the
  paper crops byte for byte, install the paper environment
  (`scripts/setup_env.sh --paper`, which installs `requirements/paper.txt` with
  `--no-deps`: `opencv-python-headless==4.13.0.90`, `numpy==2.4.6`).
* **The random starts depend on the batch and on the GPU model.** CLIP and Li-AE
  draw their starts per batch, so a crop is reproduced only with the same `seed=0`,
  the same `batch_size=64` and the same batch position (crafting from the first crop
  of the index). On CUDA, which random number lands on which pixel also depends on
  the number of multiprocessors of the GPU, and the CPU generator is a different
  stream altogether. The paper sets were drawn on two GPU models:

  | Crop sets | `cuda_blocks` | GPU of the paper run |
  |---|---:|---|
  | Color FERET CLIP and Li-AE, AI-Solutions-KK Li-AE | 272 | GeForce RTX 2080 Ti (68 SMs) |
  | AI-Solutions-KK CLIP | 288 | Quadro RTX 6000 (72 SMs) |

  With `cuda_blocks=paper_cuda_blocks(dataset, attack)` the paper's starts are
  recomputed on any device. The emulation equals torch's CUDA `uniform_` bit for bit
  (several shapes, layouts, seeds and successive draws on an RTX 2080 Ti), and with
  deterministic kernels the crops it gives equal those of the default generator on
  that GPU.
* **Li-AE on CUDA is not bit-deterministic.** Some cuDNN gradient kernels are
  non-deterministic. A near-zero gradient element can flip sign, and the sign steps
  carry that flip through the rest of that crop's optimisation.
  * On an RTX 2080 Ti with the default settings, 250 to 253 of the first 256 Color
    FERET crops (`eps = 0.06`) matched the paper crops bit for bit in six runs, and
    two runs differed from each other in 1 or 2 crops. Of the first 32 crops of each
    dataset and `eps` (192 crops), 189, 191 and 192 matched in three runs, and of the
    first 64 (384 crops), 377 to 381 in six runs.
  * `torch.use_deterministic_algorithms(True)` (with
    `CUBLAS_WORKSPACE_CONFIG=:4096:8`) makes reruns bit-identical. It selects
    different kernels, however, and matches only 237 of 256 paper crops.
  * On the CPU, with `cuda_blocks=272`, 98-99 % of the pixels and 32-56 of the first
    64 crops per dataset and `eps` equal the paper crops (with the CPU generator: 55-60
    % of the pixels and no crop).
* **CLIP on CUDA is not reproducible bit for bit.** CLIP runs in fp16 on CUDA, and
  the backward pass of the bilinear resize accumulates with atomics, so each run
  follows a slightly different trajectory from the same start.
  * Two runs from the same start on the same GPU agree on 25-38 % of the pixel
    values, depending on `eps`; a different start gives about 10 %. With the paper's
    starts the attack reproduces 25-39 % of the pixels of the paper crops on both
    datasets (with the RTX 2080 Ti's own stream the AI-Solutions-KK CLIP crops fall
    to 8-15 %). No crop is reproduced exactly.
  * The attack strength is reproduced. On the first 64 Color FERET crops at
    `eps = 0.06`, the CLIP loss `1 - cos` was 0.939 for the paper crops and 0.947
    and 0.953 for two reruns (the random start alone gives 0.070).
  * On the CPU the attack runs in fp32 and is deterministic: reruns are
    bit-identical. Its results differ from CUDA results.
* **The retrained proxy is not the released one.** It reproduces the recipe, not the
  weights ([above](#the-li-ae-proxy)).

Because of this, the paper's CLIP and Li-AE crops, and every table built on them,
can be regenerated only up to this run-to-run variation. The HFC crops are
reproduced exactly.

## Cost

These are the crafting costs measured on one RTX 2080 Ti:

| Attack | Cost per crop | Paper run |
|---|---|---|
| HFC | about 3 ms (CPU) | about 35 s per `eps` for 11,335 crops |
| Li-AE | about 3 ms (GPU, batch 64) | about 35 s per `eps` for 11,335 crops |
| CLIP | about 12 ms (GPU, batch 64, about 1.5 GB) | about 25 s per `eps` for 2,000 crops |

Li-AE on the CPU (16 threads) takes about 50 ms per crop. The emulated random starts
(`cuda_blocks`) are computed on the CPU and add about 0.25 s per random start of a
batch of 64 crops (Li-AE draws two per batch), about 1.5 min per `eps` for the 11,335
Color FERET Li-AE crops. Proxy training takes about one minute on the same GPU. The
dominant cost of the study is compressing and embedding every adversarial crop set.

## Differences between the paper text and the code

The code is the reference; these are the points where the paper's description is
looser.

* **HFC runs in the spatial domain.** The paper describes HFC as suppressing
  high-frequency components and injecting crafted noise "directly in the frequency
  domain". The implementation uses no transform. It high-passes a tiled block-noise
  pattern with a Gaussian blur (`sigma = 4`), keeps its sign, and attenuates the
  crop's own high frequencies by blending halfway towards its Gaussian low-pass.
* **CLIP and Li-AE start from a random point.** The paper calls both "I-FGSM". They
  start from a uniform random point of the `eps`-ball, as PGD does, and use the step
  `max(eps/4, 1/255)`: 0.0075, 0.015 and 0.025 for the three budgets. The ILA stage of
  Li-AE starts from a fresh random point rather than from the stage-1 result.
* **The Li-AE objective is a simplification.** The paper's "prototypical objective"
  is implemented as reconstruction of the per-identity mean crop with a single
  decoder. Li et al. use several decoders and prototype pairs. The stage-1 loss is
  the squared L2 distance of the bottleneck codes.
* **Identity overlap between the proxy's identities and the evaluation sets was not
  checked.** The paper calls the 400 WebFace42M identities "disjoint from the
  evaluation sets". No check was run for these identities. The paper's own overlap
  scan of a WebFace42M sample found 22 AI-Solutions-KK celebrities, so disjointness
  from AI-Solutions-KK is not guaranteed. The Color FERET and AI-Solutions-KK images
  were not used.
* **The AI-Solutions-KK CLIP set was crafted on a different GPU model** than the
  other CLIP and Li-AE sets (see the table above); the paper does not say so. This
  affects only which random start each crop received.

## Licences and provenance

* **Code** (`face1kb/adversarial/`): MIT. The attacks, the proxy architecture and its
  training recipe are independent implementations written for this project. No code
  of the reference implementation of Li et al.
  (github.com/qizhangli/nobox-attacks, commit 6306205, which carries no licence) is
  used, copied, vendored or downloaded. That repository uses a different
  auto-encoder (ResNet blocks, a 7 x 7 stem, several decoder heads) and a
  cross-entropy prototype loss. The ILA objective follows the formula of Huang et al.
  2019. HFC is a re-implementation from the paper's description. The random-start
  emulation implements the published Philox-4x32-10 generator (Salmon et al. 2011)
  and reproduces the launch geometry of torch's CUDA random kernels.
* **Proxy weights** (`weights/liae_proxy.safetensors`): CC BY-NC-SA 4.0. They were
  trained on a WebFace42M sample used for research purposes only, which is why the
  weights are non-commercial.
* **OpenAI CLIP**: MIT. It is installed from PyPI as `openai-clip==1.0.1`, whose
  ViT-B/32 code path is identical to github.com/openai/CLIP at commit `ded190a`. The
  checkpoint is downloaded from OpenAI at run time and is never redistributed here.
