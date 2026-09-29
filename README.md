# Face Recognition at One Kilobyte

**Evaluating Image Compression Algorithms for Barcode-Constrained Biometric
Verification**

This repository is the supplementary material for our paper **accepted at
the IEEE International Joint Conference on Biometrics (IJCB) 2026**. It
contains everything needed to replicate the paper's experiments: the
evaluation pipeline (data preparation, compression, embedding extraction,
metric computation, statistical tests) and a link to the aligned dataset.

> 📄 **Paper:** arXiv link coming soon.

## Abstract

> Barcode-based biometric credentials, such as ICAO Digital Travel
> Credentials, require facial photographs to be compressed to approximately
> one kilobyte, posing significant challenges for automated face
> recognition. This paper systematically evaluates six lossy compression
> algorithms (JPEG, JPEG2000, JPEG-AI, JPEG-FzT, JPEG-XL, and WebP) on ten
> face recognition models (seven open-source and three proprietary
> configurations) at two resolutions (112×112 and 224×224 pixels). The
> results indicate that JPEG-AI and WebP consistently yield the lowest
> degradation of recognition across models and operating points. In
> particular, JPEG-AI and JPEG-FzT are the only algorithms capable of
> compressing all 224×224 images to the 1 kB target, making them uniquely
> suitable for deployments requiring strict per-image size guarantee.
> JPEG2000 exhibits the poorest performance under these conditions.

<p align="center">
  <img src="assets/verification_scheme.png" width="720"
       alt="Barcode-based biometric verification scheme"><br>
  <em>The studied scenario: a facial photograph compressed to ≤ 1 kB is
  signed into a QR code at enrollment (Phase 1), printed on the boarding
  pass (Phase 2) and compared at the gate against a live capture
  (Phase 3). This repository covers the "Compress", "1:1 Compare" and
  "Decision" blocks.</em>
</p>

## What is in this repository

```text
face1kb/                      Evaluation pipeline (Python package)
├── config.py                 Central paths & constants (env-overridable)
├── data/                     1) Data preparation
│   ├── download_dataset.py   Fetch aligned crops from HuggingFace
│   ├── pairs.py              Canonical verification-pair protocol
│   └── define_pairs.py       Pair statistics / optional CSV export
├── compression/              2) Compression to the 1 kB budget
│   ├── codecs.py             Target-size search for all six codecs
│   ├── jpeg_fzt.py           JPEG-FzT codec (F-transform + JPEG)
│   ├── jpeg_ai.py            JPEG-AI reference-software wrapper
│   ├── compress_dataset.py   Compress every image with every codec
│   ├── decode.py             Bitstream → RGB for any codec
│   └── decompress_dataset.py Decode all bitstreams to PNG once
├── embeddings/               3) Face-embedding extraction
│   ├── compute_embeddings.py             7 open-source models (DeepFace)
│   ├── compute_embeddings_proprietary.py 3 proprietary models / mockup
│   └── make_mockup_model.py              Deterministic mockup ONNX per model
└── metrics/                  4) Metrics of the paper
    ├── verification.py       Pair scores, EER, FRR@FAR core
    ├── compute_accuracy.py   Accuracy per model × codec (Sec. 3.3)
    ├── compute_image_quality.py  SSIM + LPIPS (Table 4)
    ├── collect_file_sizes.py     Size stats & compliance (Figs. 4-5)
    ├── measure_speed.py          Codec speed (Tables 1-2)
    └── statistical_tests.py      Friedman, Wilcoxon-Holm, Cliff's δ

tools/build_hf_dataset.py     Script that built the HuggingFace dataset
docs/JPEG_AI.md               How to obtain & configure JPEG-AI
setup_env.sh                  Environment preparation
run_all.sh                    Complete pipeline, end to end
```

## Dataset

The evaluation uses the publicly available, MIT-licensed
[AI-Solutions-KK/face_recognition_dataset](https://huggingface.co/datasets/AI-Solutions-KK/face_recognition_dataset)
(105 identities, 17,534 images). The images were preprocessed with a
proprietary face detector and ArcFace-style five-landmark alignment; since
that step is not reproducible without proprietary tooling, we publish the
**aligned crops** directly:

- **HuggingFace dataset:** [`Cha53c/1kb-face-aligned`](https://huggingface.co/datasets/Cha53c/1kb-face-aligned)
  — a single split holding both crop resolutions (image, identity,
  file name, resolution); the `resolution` column (112 or 224)
  selects a resolution.

`face1kb.data.download_dataset` fetches it automatically. Verification uses
all non-redundant image pairs — 1.52 M mated and 152.2 M non-mated — formed
directly from the identity labels (see `face1kb/data/pairs.py`).

## Getting started

```bash
git clone https://github.com/innovatrics/1kb_face_ijcb.git
cd 1kb_face_ijcb
./setup_env.sh              # venv + Python deps; add --with-jpeg-ai to
                            # also clone the JPEG-AI reference software
source .venv/bin/activate
```

External requirements:

- `cjxl`/`djxl` for JPEG XL (`sudo apt install libjxl-tools`),
- the JPEG-AI reference software for the `jpeg_ai` codec — **see
  [docs/JPEG_AI.md](docs/JPEG_AI.md)**; without it the pipeline runs with
  the remaining five codecs,
- a CUDA GPU is recommended (JPEG-AI and embedding extraction).

### Run everything

```bash
./run_all.sh
```

Runs the nine stages listed in the script header for both resolutions and
writes all tables to `outputs/metrics/`. Expect the full run to take on the
order of a day on a single-GPU machine (JPEG-AI encoding and 10 × 7
embedding extractions dominate). Every stage is resumable and can be run
individually:

| Stage | Command |
| --- | --- |
| Download data | `python -m face1kb.data.download_dataset` |
| Pair statistics | `python -m face1kb.data.define_pairs` |
| Compress | `python -m face1kb.compression.compress_dataset --resolution 112` |
| Decode | `python -m face1kb.compression.decompress_dataset --resolution 112` |
| Embeddings (open-source) | `python -m face1kb.embeddings.compute_embeddings --resolution 112 --source jpeg_ai` |
| Embeddings (proprietary) | `python -m face1kb.embeddings.compute_embeddings_proprietary --resolution 112 --source original` |
| Accuracy | `python -m face1kb.metrics.compute_accuracy --resolution 112` |
| Image quality | `python -m face1kb.metrics.compute_image_quality --resolution 112` |
| File sizes | `python -m face1kb.metrics.collect_file_sizes --resolution 112` |
| Speed | `python -m face1kb.metrics.measure_speed --resolution 112` |
| Statistics | `python -m face1kb.metrics.statistical_tests --resolution 112` |

All paths default to `data/` and `outputs/` inside the checkout and can be
redirected with environment variables (`FACE1KB_DATA_ROOT`,
`FACE1KB_OUTPUT_ROOT`, `JPEGAI_REPO_DIR`, ... — see `face1kb/config.py`).

## ⚠️ Proprietary models are not included

Three of the ten evaluated recognition models (`inno-fast`,
`inno-balanced`, `inno-accurate`) are proprietary Innovatrics products and
**are not distributed** with this repository. The pipeline expects them as

```text
models/proprietary/inno-fast.onnx
models/proprietary/inno-balanced.onnx
models/proprietary/inno-accurate.onnx
```

(the directory can be changed with `FACE1KB_PROPRIETARY_MODELS`); each
takes a 112×112 BGR face crop scaled to [-1, 1] as input `input.1` of
shape (1, 3, 112, 112) and returns a 512-D embedding. For every missing
file the pipeline prints a prominent warning and substitutes a small
deterministic **mockup ONNX model** with the identical interface, one per
model with its own seed (`models/mockup/<name>_mockup_512d.onnx`, see
`face1kb/embeddings/make_mockup_model.py`), so every stage runs end to end.
Mockup embeddings carry **no biometric meaning**, so without the real
models:

- the three proprietary rows of every result table are placeholders and
  do not match the paper;
- every statistic that includes them does not match the paper either: the
  ten-model and proprietary-only means (Figs. 4-5, Table 3) and the
  Friedman / Wilcoxon-Holm / Cliff's δ tests of `statistical_tests`, which
  treat each model as a block;
- the seven open-source models are unaffected and reproduce the
  corresponding paper results.

To evaluate the real proprietary models, obtain them from
[Innovatrics](https://www.innovatrics.com) and place the ONNX files at the
paths above.

## Visual comparison

All six codecs at the 1 kB budget, 224×224 input (Figure 10 of the paper;
the label above each image gives the achieved file size and quality
parameter):

![Visual comparison of the six codecs at 1 kB](assets/visual_comparison_224.png)

## Citation

```bibtex
@inproceedings{face1kb2026,
  title     = {Face Recognition at One Kilobyte: Evaluating Image
               Compression Algorithms for Barcode-Constrained Biometric
               Verification},
  booktitle = {IEEE International Joint Conference on Biometrics (IJCB)},
  year      = {2026},
  note      = {arXiv preprint: link coming soon}
}
```

## License

Released under the [MIT License](LICENSE). The JPEG-AI reference software
and the evaluated third-party models retain their own licenses.
