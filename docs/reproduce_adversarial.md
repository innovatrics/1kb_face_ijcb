# Reproducing the adversarial study

This page covers the drivers in `experiments/adversarial/`. They reproduce the paper
section "Adversarial Robustness and Sanitization": the tables
`tab:attack-strength`, `tab:sanitization-hfc`, `tab:sanitization-clip`,
`tab:sanitization-liae`, `tab:ours-defense` and `tab:sanitization-hfc-kk`, and the
data behind `fig:sanitization`. The attacks, the Li-AE proxy and the sanitization
metric are in `face1kb.adversarial` and are described in
[adversarial.md](adversarial.md). That page also covers the threat model,
determinism and licences.

- [Quick start: the paper tables from the shipped results](#quick-start-the-paper-tables-from-the-shipped-results)
- [Full run](#full-run)
- [Scripts](#scripts)
- [Outputs](#outputs)
- [Cost](#cost)
- [Reproducibility](#reproducibility)

## Quick start: the paper tables from the shipped results

`results/adversarial/sanitization.csv` holds the per-cell metrics of the paper run.
To render the six tables from it:

```bash
python experiments/adversarial/generate_sanitization_table.py --from-results
# -> outputs/adversarial/{sanitization_hfc,sanitization_clip,sanitization_liae,
#    sanitization_hfc_kk,attack_strength,ours_defense}.tex
```

The four `sanitization_*.tex` files are byte-identical to the paper's tables. The
paper prints `attack_strength` and `ours_defense` inline in the text; the generated
tables have the same cells. This step takes seconds on a CPU.

## Full run

```bash
bash experiments/adversarial/run.sh          # GPU=0 SKIP="..." DATASETS="..."
```

The script runs five steps. Every step skips outputs that already exist, so an
interrupted run can be restarted.

1. **Craft** the adversarial crop sets with `craft.py`.
2. **Compress** the adversarial crop sets with `experiments/compress/compress.py
   --align-suffix _adv_<attack>_<tag>`. Ours-FAST and Ours-ACCURATE use the paper's
   budget accounting (`paper_compat=True`).
3. **Embed** the adversarial crops and their decoded versions with
   `experiments/embed/compute_embeddings.py --suffixes ...`.
4. **Score** every cell with `analyze.py`, which writes
   `outputs/adversarial/sanitization.csv`.
5. **Render** the tables with `generate_sanitization_table.py`.

Prerequisites:

* the aligned crops and `index.csv` of both datasets
  ([datasets.md](datasets.md), `experiments/prepare/run.sh`);
* the clean 112 px benchmark: the bitstreams and the embeddings `aligned_112` and
  `<codec>_112_<budget>` of the four anchors (`experiments/compress/run.sh`,
  `experiments/embed/run.sh`). Step 3 also computes any missing clean arrays of the
  full-coverage codecs;
* `pip install -e ".[adversarial,codecs,eval]"`, `git lfs pull` (the codec weights
  and `weights/liae_proxy.safetensors`) and
  `python scripts/fetch_models.py --only anchors`. The JPEG-AI cells also need the
  `jpegai` extra and `scripts/setup_jpegai.sh` ([baselines.md](baselines.md)).

The paper scope is fixed in `run.sh`. Subsets are the first *n* crops in `index.csv`
order; eps is 0.03, 0.06 and 0.10 and the budgets are 1024 and 512 B throughout.

| Dataset | Attack | Crops | Codecs | Matchers |
|---|---|---:|---|---|
| Color FERET | HFC | 11,335 | JPEG, JPEG 2000, WebP, JPEG XL, AVIF, HEIF, Ours-FAST, Ours-ACCURATE | 4 anchors |
| Color FERET | HFC | 300 | JPEG-FzT | 4 anchors |
| Color FERET | HFC | 300 | JPEG-AI, bmshj2018, mbt2018 | `arcface_antelopev2`, `lvface_l` |
| Color FERET | Li-AE | 11,335 | the 6 classical codecs, JPEG-FzT; Ours-FAST and Ours-ACCURATE at eps 0.06 only | 4 anchors |
| Color FERET | CLIP | 2,000 | the 6 classical codecs, JPEG-FzT, Ours-FAST, Ours-ACCURATE | 4 anchors |
| AI-Solutions-KK | HFC, Li-AE, CLIP | 2,000 | the 6 classical codecs, JPEG-FzT, Ours-FAST, Ours-ACCURATE | 4 anchors |

The four anchors are `arcface_antelopev2`, `lvface_l`, `topofr_r100` and
`edgeface_xs`.

## Scripts

**`craft.py`** writes one crop set per eps:
`aligned_dir(dataset, 112, "_adv_<attack>_<tag>")/<subject>/<stem>.png`, for example
`aligned_112_adv_liae_006`. Here `<tag>` is `round(100 * eps)` with three digits.

* It reads the clean 112 px crops in `index.csv` order and skips a crop whose file is
  missing. `--limit N` keeps the first N index rows.
* The defaults are the paper settings: seed 0, batches of 64, 10 I-FGSM steps (and 10
  ILA steps for Li-AE) and the released proxy.
* `--random-starts paper` is the default. It recomputes the random starts of the
  paper's crop sets on any device (see [adversarial.md](adversarial.md#determinism-and-reproducibility)).
  `--random-starts native` uses the attack device's own generator.
* `--device` selects the GPU for CLIP and Li-AE; HFC runs on the CPU.
* `--out-root DIR` writes the crop sets under `DIR` instead of the dataset folder.
* Existing files are kept unless you pass `--overwrite`.

```bash
python experiments/adversarial/craft.py --dataset colorferet --attack hfc
python experiments/adversarial/craft.py --dataset kk --attack clip --limit 2000 --device cuda
```

**`analyze.py`** computes `face1kb.adversarial.sanitization_record` for every
(dataset, matcher, attack, eps, codec, budget) cell. It reads four embedding arrays
per cell:

* `aligned_112`
* `aligned_112_adv_<attack>_<tag>`
* `<codec>_112_<budget>`
* `<codec>_112_adv_<attack>_<tag>_<budget>`

A cell whose arrays are missing is written with `n = 0` and NaN metrics. The defaults
cover both datasets, the four anchors, the three attacks, the three eps values, all
twelve codecs and both budgets. Other selections are available with `--datasets`,
`--models`, `--attacks`, `--eps`, `--codecs`, `--budgets` and `--out`.

**`generate_sanitization_table.py`** renders the tables from `sanitization.csv`. It
reads `--input FILE`, the shipped results (`--from-results`) or, by default,
`outputs/adversarial/sanitization.csv`. `--tables` selects a subset of the tables.
Every number is the mean over the four anchors.

* **Residual tables.** Each cell is the mean `residual`; rows are ordered by their
  mean over the columns.
  * The best cell of each column is shaded green and bold, and the worst red and
    underlined.
  * A codec with fewer matchers or fewer crops than the best-covered codec is marked
    `$^\ddagger$` and is not shaded. In the paper these are JPEG-FzT, JPEG-AI and the
    CompressAI codecs on the 300-crop HFC subset.
  * A cell that was never measured reads `---`.
* **`attack_strength`.** Each cell is the mean `cos_adv`, taken from the cells of full
  crop coverage (`cos_adv` is measured on the crops of each cell).
* **`ours_defense`.** It lists the residuals of JPEG 2000, WebP, Ours-FAST and
  Ours-ACCURATE at eps 0.06 under the three attacks.

## Outputs

| File | Content |
|---|---|
| `DATA_ROOT/<dataset>/aligned_112_adv_<attack>_<tag>/` | adversarial crops (per-image data; keep them local) |
| `WORK_ROOT/<dataset>/compressed/112_adv_<attack>_<tag>px_<budget>B/<codec>/` | bitstreams of the adversarial crops |
| `WORK_ROOT/<dataset>/embeddings/<model>/*_adv_*.npy` | embeddings of the adversarial crops |
| `OUTPUT_ROOT/adversarial/sanitization.csv` | `dataset, model, attack, eps, codec, budget, n, cos_adv, cos_clean_comp, cos_adv_comp, residual, sanitization` |
| `OUTPUT_ROOT/adversarial/*.tex` | the six tables |

`results/adversarial/sanitization.csv` has the same columns. It also lists the ten
matchers outside the anchors, with empty (NaN) rows, because the paper run scored
every matcher folder present. The tables use the four anchors only.

## Cost

These are estimates for one RTX 2080 Ti, based on the per-crop timings of the
codecs, the attacks and the matchers:

| Step | Cost |
|---|---|
| craft | HFC about 3 min on the CPU; Li-AE about 5 min on the GPU, plus about 5 min on the CPU for the paper's random starts; CLIP about 3 min |
| compress | about 20 GPU-hours, dominated by Ours-ACCURATE (about 0.4 s per encode, 139k encodes per face1kb variant); JPEG-AI about 1.5 GPU-hours (1,800 encodes); the classical codecs several CPU-hours |
| embed | about 3 GPU-hours (about 1.5 M decoded crops x 4 matchers, including the Ours decode) |
| analyze | minutes on the CPU (reading the embedding arrays) |
| tables | seconds |

## Reproducibility

* **HFC crops** are bit-identical to the paper crops. This was checked on the first
  48 crops of both datasets at all three eps (288/288).
* **Li-AE crops** reproduce up to cuDNN nondeterminism. With the paper random starts
  on an RTX 2080 Ti, 64/64 of the first Color FERET crops at eps 0.06 were identical,
  and 62-63/64 of the first AI-Solutions-KK crops at eps 0.03 over two runs (99.98 %
  of the pixels).
* **CLIP crops** reproduce only statistically. The attack runs in fp16 with atomic
  gradient accumulation. With the paper random starts, 28-30 % of the pixels of the
  first 64 crops at eps 0.06 were equal on both datasets, and no crop was identical.
  The Li-AE and CLIP tables therefore agree with the paper within run-to-run noise,
  not bit for bit ([adversarial.md](adversarial.md#determinism-and-reproducibility)).
* **Sanitization metric.** Recomputed from the study's embedding arrays, it reproduces
  the paper's `sanitization.csv` exactly: all 1,728 anchor rows, `n` and every metric
  (1,300 rows carry data).
* **Tables.** Rendered from the shipped `sanitization.csv`, the four residual tables
  are byte-identical to the paper's. `attack_strength` and `ours_defense` equal the
  inline tables of the paper, apart from whitespace.
* **Codec streams.** The compression step inherits the portability limits of the
  codecs ([codec.md](codec.md), [baselines.md](baselines.md)).
