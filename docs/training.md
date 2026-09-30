# Training the face1kb codecs

`python -m face1kb.codec.train` trains face1kb-FAST or face1kb-ACCURATE on a single
CUDA GPU. Its defaults are the recipe of the released weights (paper Section
"Variable-rate gain and training"), except the schedule length: `--steps` defaults to
600k, while the released runs used 1M-step schedules, so pass `--steps 1000000`. This page describes the recipe, the data it
expects, the commands that follow the lineage of the released weights, and the
compute cost.

> The released weights were produced by warm-start chains that include runs on
> earlier revisions of the training code (see [Lineage](#lineage-of-the-released-weights)).
> Running the trainer reproduces the **recipe**, not the exact released weights.

Install the training extra first: `pip install -e ".[train]"` (or
`scripts/setup_env.sh`).

## Data

The shipped codecs were trained on **WebFace42M** (Zhu et al., "WebFace260M: A
Benchmark Unveiling the Power of Million-Scale Deep Face Recognition", CVPR 2021),
42.5 M aligned 112 x 112 face crops of 2 M identities, used for research purposes
only. WebFace42M is not distributed with this repository; obtain it from its authors
under their terms. Any other collection of aligned 112 x 112 RGB face crops can be
used.

`--data` accepts either

* an **image folder** `<root>/<subject>/<image>.{jpg,jpeg,png}` (the layout of the
  official WebFace42M release); the subject id is the folder path relative to
  `<root>`. The file list is scanned once and sorted; pass `--data-list list.txt` to
  cache it (scanning 42 M files takes a while). A cached list is checked against the
  folder when it is read (its first entries must exist); delete it to rescan. Or
* a **HuggingFace `datasets` folder** written by `Dataset.save_to_disk`, with an
  `image` column (`datasets.Image` or `{"bytes", "path"}`) and a string `subject_id`
  column.

Crops that are not 112 x 112 are resized (bicubic) to 112 x 112.

**Sampling.** Training is step-based: every sample is a uniformly random index into
the whole source, drawn with replacement; there are no epochs, no shuffle buffer and
no augmentation. Each data-loader worker has its own fixed random stream
(`seed + 104729 * worker + 1`). Unreadable rows are skipped with a (rate-limited)
warning; 1000 consecutive unreadable rows, or 100,000 consecutive draws that all
belong to the other split, stop the run with an error instead of stalling it.

**Held-out split.** Subjects with `crc32(subject_id) % 64 == 0` (`--val-mod`) are
excluded from training (about 1/64 of the identities) and form a disjoint-identity
validation split (`face1kb.codec.data.collect_crops(root, n, split="val")`). The split
depends on the subject-id strings: the released weights were trained on a
`save_to_disk` copy of WebFace42M whose `subject_id` values are consecutive integers
`"0"`, `"1"`, ..., so an image folder with the original identity folder names gives a
different (equally disjoint) split.

## Recipe

| | FAST | ACCURATE |
|---|---|---|
| Network | `FaceCodecFast()` (N=64, M=96, Nz=48) | `FaceCodecAccurate()` (N=192, M=320, Nz=64, side-stream, refine head) |
| Resolution buckets (`--buckets`) | 64, 128, 192, 224, 256 | 64, 128, 192, 256 |
| Bucket weights (`--res-weights`) | 1 : 4 : 3 : 3 : 5 | uniform |
| Identity-loss model | EdgeFace-XS (gamma 0.6) | EdgeFace-S (gamma 0.5) |
| Loss weights R / D (MSE) / p (LPIPS) / id / ms (MS-SSIM) / floor | 1 / 0.3 / 1 / 0.4 / 0.3 / 1 | 1 / 2.0 / 1 / 0.5 / 0.3 / 1 |
| Gain levels (lambda range) | 8 (0.0018 - 0.18) | 8 (0.0002 - 0.18, log-spaced) |

Common to both variants (all defaults of the trainer):

* **Per step**: one resolution bucket is drawn with the bucket weights, the batch of
  112 px crops is resized to it (area when shrinking, antialiased bilinear when
  enlarging) and edge-padded to a multiple of 128; one of the 8 gain levels `s` is
  drawn uniformly.
* **Objective**: `R + lmbda[s] * DSCALE * (w_D MSE + w_p LPIPS + w_id (1 - cos) +
  w_ms (1 - MS-SSIM)) + DSCALE * w_floor * relu(0.6 - MS-SSIM)`, with rate `R` in bits
  per pixel (including the ACCURATE side-stream), `DSCALE = 256` (`--dscale`), LPIPS
  with the AlexNet backbone and MS-SSIM with a 7-pixel window. The identity term uses
  a frozen EdgeFace model at 112 px.
* **Identity warm-up** (three phases over `step / --steps`): the identity weight is 0
  for the first 2 % of the steps, ramps linearly to its full value between 2 % and
  20 %, and stays at full weight afterwards. All other terms are active from step 0.
* **Optimisation**: Adam, lr 1e-4 (`--lr`), batch 32 (`--batch`), linear warm-up over
  the first 1 % of the steps from 1 % of the lr, then cosine annealing to 1e-5;
  gradient-norm clipping at 1.0; fp16 autocast for the forward pass (`--no-amp`
  disables it); TF32 enabled. A separate Adam (lr 1e-3, `--aux-lr`) trains the
  hyperprior entropy-bottleneck quantiles.
* **ACCURATE side-stream**: the EdgeFace-S anchor is loaded with its official
  pretrained weights and stays frozen; the projection, the side entropy bottleneck,
  the FiLM heads and the refinement head are trained.
* **Default length**: 600k steps (`--steps`); the released weights were trained
  longer (below).

EdgeFace weights (CC BY-NC-SA 4.0, Idiap Research Institute) are downloaded on first
use from the official Idiap repositories on Hugging Face into
`$FACE1KB_MODELS_ROOT/edgeface/` and checked against their SHA-256. LPIPS downloads its
AlexNet weights through torchvision on first use.

## Outputs, checkpoints and resume

Everything is written to `--ckpt <dir>`:

* `last.pt` -- network, both optimisers, LR scheduler (and which kind of schedule it
  is), step and variant, saved every
  `--ckpt-every` steps (default 2000) and at the end (written atomically);
* `train_log.jsonl` -- one JSON record every `--log-every` steps (default 50): step,
  bucket, gain level, lambda, lr, auxiliary loss, iterations per second and the loss
  terms (`rate_bpp`, `mse`, `lpips`, `ms_ssim`, `id_cos_dist`, `total`);
* `config.json` -- the command-line arguments.

`--resume` continues from `last.pt` with its network, optimiser and scheduler state
and its step counter. It does not restore the sampling state: the data order and the
bucket / gain-level draws restart from `--seed` (so a resumed run replays the sample,
bucket and gain sequence of the run's first steps), and the fp16 loss scale restarts
from its default. A resumed run is therefore not step-for-step equivalent to an
uninterrupted one. The released ACCURATE extension was trained this way (resumed
from the 990k-step checkpoint with the default `--seed 0`); pass a different `--seed`
for fresh draws.

`--resume --rebuild-sched --extend-from S` extends a finished run to a larger
`--steps`: the learning rate is reset to `--lr` at absolute step `S` and a fresh
cosine anneals it to 1e-5 at `--steps`. The trainer records the rebuilt schedule and
`S` in `last.pt`, so a later plain `--resume` of the extended run (e.g. after a crash)
stays on the same curve; passing `--rebuild-sched --extend-from S` again is
equivalent. (A checkpoint of an extended run without this record, e.g. from an older
trainer version, stops with a message asking for the flags.) `--stop-at N` ends a run
at absolute step `N` while keeping the LR and identity schedule of `--steps`.

Checkpoints (`--resume`, and `.pt` files given to `--warmstart-from`) are read with
`torch.load(..., weights_only=True)`, which accepts the trainer's `last.pt` files but
refuses pickles that would execute code when loaded.

`--warmstart-from` initialises only the network from another checkpoint, either a
trainer `last.pt` or a net-only `.safetensors` file such as the released weights
(`--warmstart-reset-gain` keeps the freshly initialised gain vector).
`--export runs/face1kb_<variant>.safetensors` additionally writes a net-only weight
file at every checkpoint. For the default FAST and ACCURATE configurations it loads
with `face1kb.load(variant, weights_path="runs/face1kb_<variant>.safetensors")`;
exports of the ablations (`--no-side-stream`, a non-default `--anchor`) record their
architecture in the metadata and are refused by `face1kb.load`; build those with
`face1kb.codec.variants.build_model(variant, no_side, anchor)` and load the state
dict from `face1kb.codec.api.load_state_dict_file`.

Monitor the `total` field of `train_log.jsonl`. An isolated `NaN` record is not a
failure: with fp16 autocast the gradient scaler skips a step whose gradients are not
finite and lowers the loss scale, and training recovers (the run behind the released
ACCURATE weights logged one such step at about 2.14M steps and finished at 3M). A
persistent `NaN` (every record `NaN` from some step on) means the run diverged: one
3M-step extension of FAST logged a first transient `NaN` at about 2.43M steps and
became permanently `NaN` from about 2.56M steps, so FAST ships at 1M steps. `last.pt`
is overwritten at every checkpoint, so keep periodic copies of it to be able to go
back to a step before a divergence.

## Commands

The examples write everything under `runs/` (ignored by git). From scratch (recipe
of the respective variant):

```bash
python -m face1kb.codec.train --variant fast --data /path/to/webface42m \
    --data-list runs/webface42m_files.txt --steps 1000000 --ckpt runs/fast
python -m face1kb.codec.train --variant accurate --data /path/to/webface42m \
    --data-list runs/webface42m_files.txt --steps 1000000 --ckpt runs/accurate
```

A chain that mirrors the lineage of the released weights with the current code (the
original chains also contained earlier runs, see below):

```bash
D=/path/to/webface42m
L=runs/webface42m_files.txt
# FAST: a 300k-step run with a 224/256-heavy bucket weighting, then 1M steps with
# the final 1:4:3:3:5 weighting, warm-started from it
python -m face1kb.codec.train --variant fast --data $D --data-list $L --steps 300000 \
    --res-weights 1,1.5,3,5,5 --ckpt runs/fast_300k
python -m face1kb.codec.train --variant fast --data $D --data-list $L --steps 1000000 \
    --warmstart-from runs/fast_300k/last.pt --ckpt runs/fast_1M \
    --export runs/face1kb_fast.safetensors

# ACCURATE: a 60k-step run; a warm start on a 1M-step schedule, stopped at 990k
# steps; then an extension to 3M steps on a rebuilt cosine schedule
python -m face1kb.codec.train --variant accurate --data $D --data-list $L \
    --steps 60000 --ckpt runs/acc_60k
python -m face1kb.codec.train --variant accurate --data $D --data-list $L \
    --steps 1000000 --stop-at 990000 --warmstart-from runs/acc_60k/last.pt \
    --ckpt runs/acc_3M
python -m face1kb.codec.train --variant accurate --data $D --data-list $L \
    --steps 3000000 --resume --rebuild-sched --extend-from 990000 --ckpt runs/acc_3M \
    --export runs/face1kb_accurate.safetensors
```

The bucket weights of the FAST 300k-step run were not recorded; `1,1.5,3,5,5` is
estimated from the bucket counts in its training log (64 / 128 / 192 / 224 / 256 px
drawn 356 / 589 / 1162 / 1912 / 1981 times over 6000 logged steps).

Ablations of the paper: `--no-side-stream` trains ACCURATE without the identity
side-stream and refinement head; `--anchor <name>` replaces the EdgeFace-S anchor by
another torch FR model. From the command line only the built-in EdgeFace names
(`edgeface_xxs`, `edgeface_xs`, `edgeface_s`, `edgeface_base`) are available; other
models must be registered with `face1kb.codec.identity_loss.register_torch_fr` in the
same process before the trainer builds the network, e.g. with a small wrapper script:

```python
# train_my_anchor.py
from face1kb.codec.identity_loss import register_torch_fr
from face1kb.codec.train import main

def build_my_fr(pretrained: bool):
    ...  # return a torch module mapping 112 x 112 RGB in [-1, 1] to 512-D embeddings

register_torch_fr("my_fr", "edgeface", build_my_fr)
raise SystemExit(main(["--variant", "accurate", "--anchor", "my_fr", "--data", "...",
                       "--steps", "1000000", "--ckpt", "runs/acc_my_fr"]))
```

The `family` argument selects the embedding call (see `register_torch_fr`).

The paper's TopoFR-R100 de-confound arm (`--anchor topofr_r100`) uses the evaluator
roster's TopoFR model. Its weights have no redistribution licence and are not shipped;
fetch them from the authors' official source with
`python scripts/fetch_models.py --only topofr_r100` (see [models.md](models.md)), then
register the built-in builder in the wrapper script above instead of `build_my_fr`:

```python
from face1kb import fr

register_torch_fr("topofr_r100", "topofr", fr.torch_builder("topofr_r100"))
```

This builder gives the same normalised embeddings as `fr.load("topofr_r100")`, and
`pretrained=False` builds the same 360,232-class architecture. The EdgeFace anchors
used for training (EdgeFace-XS/S) are fetched with
`python scripts/fetch_models.py --only training`.

By default the trainer keeps the frozen anchor in eval mode. The trainer of the paper
runs left it in training mode (`net.train()` also switched the anchor), which makes
no difference for EdgeFace (no batch-norm, no active dropout) but does for anchors
with batch-norm such as TopoFR: in training mode they normalise with the statistics
of each batch and keep updating their running statistics.
Pass `--anchor-train-mode` to restore that behaviour for a faithful rerun of such an
ablation arm; without it, a batch-norm anchor trained with this code differs from the
paper's arm.

## Lineage of the released weights

| Weights | Chain | Buckets |
|---|---|---|
| face1kb-FAST | earlier runs -> 300k-step run (bucket weights about 1:1.5:3:5:5) -> warm start -> 1,000,000 steps | 64/128/192/224/256, weights 1:4:3:3:5 in the final run |
| face1kb-ACCURATE | earlier runs -> 60k-step run (warm-started with a reset gain vector, `--warmstart-reset-gain`, so that the gain range is 0.04-1.0) -> warm start on a 1,000,000-step schedule, stopped at 990k steps -> resumed and extended to 3,000,000 steps on a rebuilt cosine (`--extend-from 990000`) | 64/128/192/256, uniform |

The early links of both chains were trained with earlier revisions of the codec code
that were fixed afterwards, so a from-scratch run with this trainer follows the same
recipe but does not reproduce the released weights bit for bit. Weight initialisation
is not seeded by default (as in the released runs); `--init-seed` seeds it. The
bucket / gain-level sampling (`--seed`) and the data order are seeded.

## Compute cost

Measured throughput of the released runs at batch 32 on one NVIDIA Turing GPU
(fp16 autocast): **FAST about 5.0 it/s** (1M steps in about 2.3 days) and
**ACCURATE about 2.35 it/s** (the 2M-step extension from 990k to 3M took about 9.9
days; 3M steps from scratch correspond to about 15 days). The data loader needs 8
workers (`--workers`) with fast random access to the training images.

## Notes

* compressai is pinned to 1.2.8. In that version the variable-rate entropy bottleneck
  of the hyperprior passes its quantisation step positionally into the
  `stop_gradient` argument of `_likelihood_variable`, so the z density model stays at
  its initialisation during training. The released weights were trained this way,
  and the trainer keeps it.
* In the released ACCURATE weights the gain of the side-stream projection's
  LayerNorm collapsed during training (mean absolute value about 4e-4), so the
  projected identity code equals the LayerNorm bias for every input and is stored as
  the same 8 bytes for every image (see [codec.md](codec.md#limitations)); the
  side-stream acts as a constant learned conditioning of FiLM and the refinement
  head. Separately, the auxiliary loss covers only the hyperprior entropy bottleneck,
  so the quantiles of the side-stream entropy bottleneck stay at their initial values;
  this does not cause the constant code. The trainer keeps both properties of the
  recipe; a new run may or may not collapse the same way.
