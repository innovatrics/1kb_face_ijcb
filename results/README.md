# Paper results

Aggregate result files of the extended study (arXiv 2608.22866): the numbers behind
its tables and figures. The layout is the same as `FACE1KB_OUTPUT_ROOT`, so every
table and figure generator of `experiments/` can read this folder instead of a fresh
run (`--from-results` / `--input-root results`).

| Folder | Content |
|---|---|
| `accuracy/` | verification grid (`metrics.csv`), preprocessing-through-codec, paired significance tests, rank statistics |
| `ablation/` | side-stream ablation of the ACCURATE codec, clean preprocessing and crop-tightness ablation |
| `adversarial/` | attack strength and sanitization per codec and budget |
| `annex/` | ISO/IEC 29794-5 Annex E/F study |
| `codec_comparison/` | achieved file sizes, JPEG-AI operation points, per-codec identity cosine and quality on held-out crops |
| `codec_properties.csv` | provenance, licensing and implementation of the baseline codecs |
| `contamination/` | dataset-level train/test overlap summary |
| `difficulty/` | image-level difficulty correlations, easiest vs. hardest decile contrast |
| `fairness/` | subgroup EERs and disparities, differential FMR, disparity CIs |
| `quality/` | full-reference quality, face image quality, identity-cosine tail, reconstruction checks, encode/decode speed |
| `recompression/` | recompression chain EERs |
| `report_summary/` | findings register, figure-derived scalars, side-channel leakage test |
| `resolution_information/` | EER per resolution, resize cosine, spectral retention |

All files are aggregates over cells or subgroups: no images, per-image values, file
names or embeddings. The file-by-file description, the paper item each file backs and
the known deviations are in [docs/results.md](../docs/results.md); column definitions
of the accuracy and fairness files are in [docs/metrics.md](../docs/metrics.md).

The identity-cosine values in `codec_comparison/res*/comparison.json` were measured
with a proprietary face matcher that is not part of this release and cannot be
re-measured with the public code.
