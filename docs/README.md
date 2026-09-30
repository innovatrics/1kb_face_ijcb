# Documentation

Start with the [repository README](../README.md) for an overview and a quick start.
To regenerate a table or figure of the paper, go to [reproduce.md](reproduce.md).

## The face1kb codecs

| Page | Content |
|---|---|
| [codec.md](codec.md) | face1kb-FAST and face1kb-ACCURATE: architecture, container format, hard byte budget, API, command line, performance, bitstream reproduction, limitations |
| [training.md](training.md) | Training recipe, data layout, commands of the released lineage, resume and extension, compute cost |
| [../weights/README.md](../weights/README.md) | Model card of the released weights (codecs and Li-AE proxy), hashes, bundled EdgeFace-S, licence |

## Library reference

| Page | Package | Content |
|---|---|---|
| [baselines.md](baselines.md) | `face1kb.baselines` | The ten baseline codecs, their budget searches, JPEG-AI setup, CompressAI, decoding stored files |
| [models.md](models.md) | `face1kb.fr` | The 14 face-recognition evaluators, fetching and pinning, preprocessing, plugging in your own matcher |
| [datasets.md](datasets.md) | `face1kb.data`, `experiments/prepare/` | Obtaining the datasets, data layout, index and pairs, Color FERET labels, KK attributes, crop variants, preprocessing operators |
| [metrics.md](metrics.md) | `face1kb.eval` | Verification metrics, significance tests, fairness, image quality, CSV schemas |
| [adversarial.md](adversarial.md) | `face1kb.adversarial` | Threat model, the HFC / CLIP / Li-AE attacks, the Li-AE proxy, sanitization metric |

## Reproducing the paper

| Page | Paper sections |
|---|---|
| [reproduce.md](reproduce.md) | Overview: every section, table and figure mapped to commands, inputs, cost and whether it renders from `results/` |
| [reproduce_compress.md](reproduce_compress.md) | Compressed grid, decoded caches, budget compliance (5.7), codec properties (2), speed and the JPEG-AI side studies (6) |
| [reproduce_accuracy.md](reproduce_accuracy.md) | Embeddings, the verification grid (5), held-out matcher and identity tail (7), significance (14) |
| [reproduce_quality.md](reproduce_quality.md) | Image quality and face image quality (6), codec comparison (5-7), fairness (11) |
| [reproduce_studies.md](reproduce_studies.md) | Sample difficulty (8), resolution and preprocessing (9), recompression (12) |
| [reproduce_adversarial.md](reproduce_adversarial.md) | Adversarial robustness and sanitization (13) |
| [reproduce_annex.md](reproduce_annex.md) | ISO/IEC 29794-5 Annex E/F study (10) |
| [results.md](results.md) | The shipped `results/` folder file by file, the summary figures and the known deviations of the shipped files |
| [errata.md](errata.md) | Errata of the arXiv report 2608.22866 v1 (not the IJCB paper), to be fixed in its next version, and reproducibility limits |
