# Errata: arXiv report 2608.22866 (v1)

This page is the errata list of the extended report
**"Toward Sub-1 kB Identity-Preserving Face Compression"**
([arXiv:2608.22866](https://arxiv.org/abs/2608.22866), version 1). It does **not**
concern the IJCB 2026 conference paper *Face Recognition at One Kilobyte*, whose code
is preserved under the git tag `ijcb2026`.

Each entry is a confirmed difference between the text of the report and the code and
result files of this repository. **The code is the reference**: it produced the
published numbers and has not been changed to match the text. These corrections will
be incorporated into the next version of the report on arXiv. Labels such as
`tab:speed` refer to the LaTeX labels of the report source; table and figure numbers
are those of the v1 PDF. The area pages linked in each section give the details.

- [The face1kb codecs (Section 7)](#the-face1kb-codecs-section-7)
- [Baseline codecs and settings (Sections 4-6)](#baseline-codecs-and-settings-sections-4-6)
- [Datasets and attributes (Section 3)](#datasets-and-attributes-section-3)
- [Matchers and scoring protocol (Sections 4, 5, 14)](#matchers-and-scoring-protocol-sections-4-5-14)
- [Quality, codec comparison and fairness (Sections 5-7, 11)](#quality-codec-comparison-and-fairness-sections-5-7-11)
- [Side studies (Sections 8, 9, 12)](#side-studies-sections-8-9-12)
- [Adversarial study and annex study (Sections 10, 13)](#adversarial-study-and-annex-study-sections-10-13)
- [Hand-maintained tables, findings and artifacts](#hand-maintained-tables-findings-and-artifacts)
- [Items that cannot be regenerated publicly](#items-that-cannot-be-regenerated-publicly)
- [Reproducibility limits](#reproducibility-limits)

## The face1kb codecs (Section 7)

Details: [codec.md](codec.md), [training.md](training.md).

- **Side-channel size.** The paper gives the ACCURATE identity side-channel as about
  90-175 B (`fig:codec-arch`, `subsec:codec-side`). It derives from this a share of
  9-17 % of a 1024 B container and 18-34 % of a 512 B one, which it uses in the
  budget-reallocation paragraph. In fact, all 288,690 stored ACCURATE containers have
  an **8-byte side-channel**, and it is the same 8 bytes in all of them
  (`f82a5c8500000000` on CUDA). In the released weights, the gain of the LayerNorm in
  the side-stream projection has collapsed (mean absolute value about 4e-4). The
  projected code therefore equals the LayerNorm bias and quantises to the same
  integers for every input. The code acts as a constant learned conditioning of FiLM
  and the refine head. Independently, the quantiles of the side entropy bottleneck are
  never trained (the auxiliary loss covers only the hyperprior bottleneck); this is
  not what makes the code constant.
- **"Hard identity floor".** The paper says the side-channel survives spatial-latent
  collapse and backs the identity-only fallback. The stored code is image-independent,
  so it carries no identity. An identity-only container decodes to a black frame: the
  decoder does not use the side-channel without a spatial latent.
- **Privacy paragraph.** The paper measured linkability (EER 17.0 %, AUC 0.914) on the
  unquantised 128-D projection. That projection is never stored. The stored,
  entropy-coded code is identical for all images and cannot link them. The spatial
  latent still encodes the face, so containers remain biometric data.
- **Budget misses at 96/168 px.** The paper attributes them to rate floors (Known
  limitations). Of the 54,030 misses at 96 and 168 px, 65 % are trailer overruns of
  exactly 3 bytes. The search target reserves 7 + `sc_len` bytes, but the container
  also appends a 4-byte raw-geometry trailer at resolutions without a bucket code. The
  overruns cover all misses of FAST at 96 px and at 168 px / 1024 B, and 99.7 % of
  the ACCURATE misses (17,558 of 17,614). The genuine floors are FAST at 168 px /
  512 B (mostly), all FAST misses at 224 px / 512 B, and 56 ACCURATE containers.
  `paper_compat=True` keeps the paper accounting. The default reserves the trailer and
  emits an identity-only container when nothing fits. The fit tables
  (`tab:fit-cf`, `tab:fit-kk`) match the stored files.
- **Gain vector.** The paper says "a learned gain vector sets the quantisation step".
  The gain vector is never trained: CompressAI 1.2.8 detaches it in the training
  forward pass. Both weight files store it at its initialisation (FAST: 0.1-1.0;
  ACCURATE: `0.04 * 25**(i/7)`). The network is trained at these fixed gains.
- **Identity warm-up.** The paper says the identity term is "warmed up from zero". In
  the code it is off for the first 2 % of the steps, ramps linearly until 20 %, and is
  at full weight afterwards.
- **Speed (`tab:speed`).** The face1kb rows time a single encode at the gain that hits
  the budget, without the rate search. The full `Codec.encode`, including the binary
  search of about 6 encodes, takes 129-148 ms (FAST) and 343-412 ms (ACCURATE) per crop
  on an RTX 2080 Ti with a warm cache. The published rows were measured before the
  weights were final, so their rate indexes and sizes cannot be regenerated. With the
  released weights, FAST at 112 px / 1024 B selects rate index 54 and 1011 B instead
  of 55 and 1007 B, and ACCURATE selects 60 and 1007 B instead of 58 and 1001 B. For
  FAST the cause is not established. The speed benchmark takes the ACCURATE rate index
  from the release encode path, which computes the side code on the unpadded crop.
  The original benchmark used a padded-crop side code; both give 60 at 112 px / 1024 B.
- **Anchor mode in training.** The original trainer's `net.train()` also put the
  frozen anchor into training mode. This makes no difference for EdgeFace (no
  batch-norm), but it changes a batch-norm anchor such as the TopoFR arm of the
  side-stream ablation. The trainer keeps the anchor in eval mode by default;
  `--anchor-train-mode` restores the original behaviour.
- **Codec-properties table (`tab:codec-properties`).** The Ours rows print "MIT (source
  to be released)". In this release the code is MIT and the weights are
  CC BY-NC-SA 4.0. The generator keeps the paper wording so that the table stays
  byte-identical.

## Baseline codecs and settings (Sections 4-6)

Details: [baselines.md](baselines.md), [reproduce_compress.md](reproduce_compress.md).

- **JPEG-AI profile.** `tab:codec-settings` says "profile SOP". Every JPEG-AI result
  except the operation-point study used `cfg/tools_off.json` with
  `cfg/profiles/high.json` (HOP) for encode and decode. HOP is the default here.
- **JPEG 2000 rate grid.** The compression-ratio grid runs from 400 down to 12 in
  steps of 4 (`range(400, 8, -4)`), not 8-400. The annex study's JPEG 2000 ladder ends
  at 400 × 0.97<sup>175</sup> = 1.94, not "ratio 2".
- **AVIF backend.** AVIF ran through Pillow 12's native plugin (libavif with aom), not
  through pillow-heif / libaom.
- **JPEG-FzT.** The benchmark ran this repository's Python (NumPy) rewrite, not the
  reference C++/Qt implementation. Every result uses radius 1, while Section 9 states
  radius 2 as the benchmark default. The "triangle kernel" variant of
  `fig:fzt-ablation` is not implemented. `tab:codec-settings` says the quality is
  "capped at quality 94", which does not hold for JPEG-FzT: its grid is 1-95.
- **JPEG-FzT timing (`tab:speed`).** The decode time of 0.16 ms (shaded fastest)
  covers only the JPEG decode of the half-resolution image. The full decode, which
  includes the inverse F-transform, takes about 30 ms at 112 px on one core. The
  original encode also wrote a JPEG to a temporary file, so the ported encode is
  slightly faster.
- **CompressAI description.** `tab:codec-settings` calls them "pretrained hyperprior /
  joint models". bmshj2018-factorized has no hyperprior, and mbt2018-mean has no
  autoregressive context model.
- **Color FERET JPEG-AI cells.** In every Color FERET cell, 31.6 % of the stored
  streams come from an interrupted earlier run with another bpp search. They use the
  same HOP configuration, but their sizes differ strongly from the final rule (e.g.
  median 1007 vs 303 B at 64 px / 1024 B). A public re-run encodes every crop with the
  final rule, so the Color FERET JPEG-AI numbers change. The stored cells have further
  gaps: 3-4 crops per cell have no stream, at least 8 main-grid streams are
  lowest-rate (bpp 2) fallbacks, and three truncated streams feed the JPEG-AI row of
  the HFC sanitization table. The KK cells used the final rule throughout.
- **mbt2018 decodes.** The CUDA decode of mbt2018-mean is not deterministic and can
  silently desynchronise the rANS decoder. In the 112 px 768 / 960 B cells, 1.6 % and
  2.4 % of the files decode to noise with default kernels. Those files were encoded
  after the JPEG-AI reference software had left cuDNN in deterministic mode. The
  paper's mbt2018 reconstructions of those crops were therefore probably corrupt. The
  decoder here verifies the latent and reconstructs all stored files within one grey
  level, so regenerated mbt2018 quality medians differ slightly. Encodes of files at a
  scale boundary can also differ (about 2 in 720).
- **CompressAI subset.** The CompressAI cells use the first 300 rows of the canonical
  index. Color FERET 224 px / 1024 B holds 303 stored files, the subset plus 3
  spot-checks.
- **JPEG-AI side studies.** The JPEG-AI speed benchmark allows bpp × 100 up to 65 at
  64-112 px, above the benchmark's fit cap of 50. The published GPU speed row
  (1128 B, 38 crops) re-runs at 1131 B with stride sampling. `tab:jpegai-kk` shades
  the ArcFace EER column (SOP best, BOP worst) although the three values are
  statistically flat. Its fixed target of bpp × 100 = 16 realises about 1111 B, 8.5 %
  over 1024 B, so it is not a budget fit. No script for the measurement behind
  `tab:jpegai-speed` exists; the table is rendered from the shipped CSV.

## Datasets and attributes (Section 3)

Details: [datasets.md](datasets.md).

- **Availability.** Section 3 says the aligned AI-Solutions-KK crops are "freely
  downloadable". They are **available on request from the authors**.
- **Color FERET pose angles.** The yaw, pitch and roll of the attribute table are
  per-image head-pose estimates, not NIST ground truth. For example, the half-left
  mean yaw is 37.6°, while the NIST nominal angle is 67.5°. NIST provides only the
  nominal angle of each pose code. All other attribute columns are NIST ground truth
  and are reproduced exactly (11,338 of 11,338).
- **Color FERET landmarks.** The paper mentions 23-point landmark annotations. The
  NIST distribution contains only eye, nose and mouth coordinates.
- **KK age and gender.** The estimates behind the fairness analysis and
  `fig:ds-kk-demo` were computed on an earlier set of 112 px crops with tighter
  framing. On that set the CPU default reproduces all 840 outputs. On the benchmark
  crops, the per-identity gender mode agrees for 104 of 105 identities and the median
  age band for 95 of 105. The skin-tone labels (from the 224 px crops) are unaffected.
  On CUDA, one age on a rounding boundary moves by a year.
- **Resolutions.** The 112 px crops are exactly `aligned_224[::2, ::2]`, and the annex
  study's 56 px crops are exactly `aligned_112[::2, ::2]`. The 64, 96 and 168 px crops
  are separate warps.
- **Dataset figures.** `fig:ds-cf` shows source images of the original dataset copy
  and cannot be regenerated; the public script builds a montage of your own crops. The
  size histogram of `fig:ds-kk-orig` used an unseeded file sample. The port uses a
  seeded sample with the same medians (175 × 184 px) and a slightly different
  histogram.

## Matchers and scoring protocol (Sections 4, 5, 14)

Details: [models.md](models.md), [metrics.md](metrics.md),
[reproduce_accuracy.md](reproduce_accuracy.md).

- **Matcher input size.** Section 4.1 says every model consumes the aligned crop at
  the working resolution. Every matcher consumes 112 × 112 px: crops at 64, 96, 168
  and 224 px are resized bilinearly to 112 px before embedding.
- **CVLface ViT-B keypoints.** CVLface ViT-B (KP-RPE), also the matcher of the annex
  study, receives the fixed ArcFace five-point template as the keypoints of every
  crop, not per-image landmarks.
- **Symmetric scoring.** Section 4.2 says results are compressed-vs-original unless
  stated otherwise. The code scores every accuracy, significance, fairness,
  recompression, preprocessing and resolution cell compressed-vs-compressed: both
  members of a pair come from the same embedding array.
  Only the annex study also reports an asymmetric protocol.
- **McNemar.** The paper compares decisions "at a fixed operating threshold". In the
  code, each source decides at its own EER threshold on the shared trial set.
- **Cliff's delta.** The paper describes a paired "randomly chosen matcher"
  probability. The code computes the unpaired two-sample delta over all matcher
  combinations.
- **BH-FDR grouping.** The shipped significance CSVs, and the paper's 55 tests per
  cell, use per-cell Benjamini-Hochberg correction. This is the default here. Grouping
  per (dataset, model) instead, as one original script did, shifts the KK adjusted
  p-values by at most 0.0099 and flips no decision.
- **Section 14.2, independent-matcher claim.** The text says that under CVLface
  IR-101 the Ours-ACCURATE vs AVIF pair at Color FERET 112 px / 1024 B is not
  significant (p_adj = 0.34). No artifact exists for this claim. The published
  protocol gives chi² = 51.8 and p_adj = 6.7e-13, which is significant. The only
  non-significant pair of that cell is JPEG-AI vs WebP (p_adj = 0.88). The
  |ΔAUC| ≤ 2e-4 part holds. The command is in [reproduce.md](reproduce.md#section-14-statistical-significance).
- **Scorer precision.** The paper does not say which scorer each result used.
  `metrics.csv` and all fairness CSVs were scored with the CUDA (torch float32)
  scorer, and the significance CSVs with the NumPy scorer. They differ in the last
  float32 bit. With the NumPy scorer, 4 of 49 checked `metrics.csv` cells move by one
  impostor trial.
- **`metrics.csv` contents.** Section 4.4 says the `median_bytes` column records the
  median encoded size. It is NaN in every row; the sizes are in
  `codec_comparison/file_size_summary.csv`. The file has 4,618 rows (not 4,632),
  including 28 empty KK neural 224 px / 1024 B rows. It has no rows for KK CVLface
  IR-101 Ours-ACCURATE at 168 px, although the arrays exist, so a full run adds these
  two rows.

## Quality, codec comparison and fairness (Sections 5-7, 11)

Details: [reproduce_quality.md](reproduce_quality.md), [metrics.md](metrics.md).

- **Quality aggregation.** Section 4.3 says the pipeline "averages over the cell". The
  published quality tables are per-cell **medians**.
- **Identity cosine from a proprietary matcher.** The id-cos columns of `comparison.json`,
  Tables 7, 8, 40 and 41 and key finding F02 were measured with a proprietary matcher
  that is not part of this release. The public default (`cvlface_ir101`, or a matcher
  registered with `face1kb.fr.register_onnx`) gives different values. The study also
  embedded float reconstructions, while the public path rounds them to uint8. With
  the same matcher, this moves the median id-cos by at most 1.1e-3 (112 px) and 5.4e-3
  (224 px); all other fields reproduce exactly.
- **224 px / 512 B tables.** Several codecs cannot reach the budget and fall back to
  their smallest setting. For example, the WebP median is 775 B on Color FERET and
  1,155 B on KK. Tables 7, 8, 40 and 41 do not show these sizes.
- **FIQ grid (Table 34).** The published shading ranks unrounded means
  (`--shading raw`). The Ours-ACCURATE rows and the KK JPEG-AI rows were computed in
  the `--codecs` mode on a different 250-crop sample. A full run gives, for example,
  0.954 instead of 0.952 for Ours-ACCURATE at Color FERET 112 px / 1024 B.
- **`quality_kk.csv`.** The Ours-ACCURATE rows at 64, 96 and 168 px were measured on
  bitstreams of an earlier ACCURATE checkpoint. No paper table uses them.
- **Disparity CIs (Table 63, `tab:disparity-ci`).** The code assigns an impostor pair
  to a skin-tone subgroup by its first image only, so cross-tone impostors are
  included. Point values therefore differ from the within-subgroup disparities
  (ArcFace, aligned: 0.35 vs 0.37 pp). The paper attributes the gap only to the 400k
  impostor sample. The aligned and Ours rows match `disparity_ci_kk.csv`, which the
  code reproduces 40 of 40. The WebP and JPEG 2000 rows do not match. The file gives
  (ratio [95 % CI], ArcFace / LVFace-L / TopoFR-R100 / EdgeFace-XS):
  - WebP: 1.6 [1.1, 2.9], 1.1 [0.7, 2.3], 1.1 [0.8, 2.5], 1.1 [0.9, 1.6].
  - JPEG 2000: 10.7 [4.5, 29.4], 8.7 [3.9, 31.3], 12.6 [4.9, 33.9], 4.3 [2.1, 9.5].

  The text's "JPEG 2000 lower bounds 1.5-4.5×" are 2.1-4.9× in the file. "WebP's CI
  includes 1×" does not hold for ArcFace: its CI, [1.07, 2.86], excludes 1.
- **Differential FMR (Table 65).** The caption says JPEG-AI and CompressAI are omitted
  because every subgroup FMR quantises to zero. The actual cause is NaN propagation.
  A few crops have no embedding, so the impostor pool contains NaN, the threshold
  becomes NaN and every FMR evaluates to 0. With `nan_policy="omit"`, the Color FERET
  JPEG-AI dFMR is 2.62 / 1.05 / 2.10 / 1.62 pp (ArcFace / LVFace-L / TopoFR /
  EdgeFace), inside the band. The default keeps `propagate` to reproduce the table.

## Side studies (Sections 8, 9, 12)

Details: [reproduce_studies.md](reproduce_studies.md).

- **Recompression, Ours-ACCURATE.** The paper says the full matrix was run. The 15
  Ours-ACCURATE cells per table are scored on a 120-crop sample taken at a fixed
  subject stride, with up to 10 crops per subject. On Color FERET this is 15
  subjects, 456 mated and 6,684 non-mated pairs. Their delta-EER is relative to a
  single-pass reference on the same sample.
- **Recompression, KK.** The classical block uses the first 2,500 index crops, while
  the single-pass references use all 17,534.
- **Recompression, JPEG XL → AVIF / HEIF.** The paper blames an interaction of codec
  artefacts for these failures. The decoded first pass keeps its 536-byte ICC profile,
  and AVIF and HEIF embed it, so about half of a 1 kB budget goes to metadata.
  `--strip-metadata` exists but is not the paper setting.
- **Preprocessing through the codec.** The sample is a uniform seeded draw of 800
  crops, not "subject-stratified".
- **`fig:preproc-impact`.** The labels are WebP qualities; the caption says "JPEG
  quality factor". The table sample is the first four index crops (one subject), and
  the figure shows another subject.
- **`fig:align-check`.** The dots are the fixed ArcFace template, not detected
  interest points. The axis label now reads "template overlay".
- **Crop-tightness presets (`tab:ablation-crop`).** The paper warped the presets from
  the source photographs. The public presets are sub-windows of the 224 px crop. The
  EERs move by at most 0.33 pp (LVFace-L fill 10.44 → 10.11), with unchanged ordering
  and shading. The shipped CSV holds the paper values.
- **`tab:res-decomp`.** This table was typed by hand. Its EdgeFace-XS 96 px EER reads
  1.45; the CSV gives 1.444.
- **`fig:res-summary` (Figure 34).** The arXiv figure has legends inside panels 1 and
  3 and draws a KK TopoFR-R200 line. The code draws one legend panel, fixes the
  per-panel colour cycling, and has no data for that line.
  `embedding_decomposition.csv` holds only an empty 64 px row for TopoFR-R200 on KK,
  because the paper run read the 112 px array while it was still being written. A
  fresh run gives 0.954. The table does not use the row.
- **Sample difficulty.** The paper's id-cos used the proprietary matcher; the public
  default (`--id-model cvlface_ir101`) gives different values. The Color FERET |yaw|
  and pitch descriptors were per-image head-pose estimates. The public NIST labels
  have only nominal pose-code angles, so those rows of `tab:difficulty-predictors`
  and `tab:difficulty-contrast` cannot be regenerated. The KK age-bucket means cover
  12 images, and MST has no sampled crop, so its cell prints "--".

## Adversarial study and annex study (Sections 10, 13)

Details: [adversarial.md](adversarial.md), [reproduce_adversarial.md](reproduce_adversarial.md),
[reproduce_annex.md](reproduce_annex.md).

- **HFC.** The paper says HFC works "directly in the frequency domain". HFC runs in
  the spatial domain. It tiles 4 × 4 blocks of Gaussian noise, high-passes them with
  a Gaussian blur and takes the sign. It blends the crop halfway towards its own
  low-pass, adds eps × sign and projects. No transform is used.
- **I-FGSM details.** The paper does not state these details. The CLIP and Li-AE
  attacks start from a uniform random point in the eps-ball, with step
  max(eps/4, 1/255). The Li-AE ILA stage restarts from a fresh random point, not from
  the stage-1 result.
- **Li-AE.** The implementation simplifies Li et al. (2020) in ways the paper does not
  state:
  - it uses a single decoder that reconstructs the per-identity mean crop;
  - stage 1 maximises the squared L2 distance of the 256 × 7 × 7 bottleneck codes;
  - ILA uses the 64 × 28 × 28 second encoder stage.
- **Proxy identities.** The paper calls the 400 WebFace42M proxy identities disjoint
  from the evaluation sets. No overlap check was run for them. The paper's own
  WebFace42M overlap scan found 22 KK celebrities, so disjointness from KK is not
  established.
- **Attack scope.** On Color FERET, HFC and Li-AE ran on all 11,335 crops and CLIP on
  2,000. On KK, all three attacks ran on 2,000 crops. The Ours codecs were tested
  against Li-AE on Color FERET at eps 0.06 only. JPEG-FzT, JPEG-AI and the CompressAI
  baselines were attacked with HFC only, on 300 Color FERET crops. JPEG-FzT was
  scored with 4 anchors, the others with 2. Some KK JPEG XL cells have n = 1999.
- **Inline tables.** `tab:attack-strength` and `tab:ours-defense` are typed inline in
  the paper. They are now generated from `sanitization.csv`, and every cell equals the
  paper.
- **Annex 80 px crops.** The paper warped every annex resolution from the source
  photographs. Here 80 px is resampled from the 224 px crop, with a median PSNR of
  about 38 dB against a direct warp. This affects the Annex-F WebP cell of
  `tab:annex-exp1`, the JPEG XL winner (r80) and 48 grid cells. The 56 px crops are
  exact.
- **Annex KK crops.** The KK annex crops were aligned separately for the study, with a
  different warp and interpolation from the released KK crops (about 43.6 dB). The KK
  columns of `tab:annex-exp1` and `tab:annex-ci` do not reproduce bit for bit from the
  released crops.
- **Annex Stage B.** The original selection step failed on the JPEG-AI grid cells.
  Selection is restricted to the six classical codecs, which regenerates exactly the
  62 recorded Stage-B cells and the 6 winners. `scores_pop2000.csv` must be scored
  after Stage B and before the confirmation run, as `run.sh` does. The recorded
  `bytes_fixed_q` of a cwebp (sns) cell is the fitted size; `tab:annex-bytes` uses only
  cells without flags and is unaffected.

## Hand-maintained tables, findings and artifacts

Details: [results.md](results.md).

- **Table 24 (`tab:best-config`).** The JPEG-AI 1024 B cell prints `0.73 @224`. The
  shipped `metrics.csv` gives a mean FNMR at 1e-4 of 0.71 % (ArcFace 0.287 %,
  LVFace-L 1.126 %) at 224 px. 0.73 is the value from before the JPEG-AI cells were
  re-scored. The best resolution and the shading are unchanged.
- **`tab:fit-kk` (1024 B).** The arXiv table has two all-"--" rows for bmshj2018 and
  mbt2018. The generator drops codecs without a cell of at least 30 files, as in the
  512 B table.
- **Key finding F16.** The statement says "only two" modern-codec pairs are
  non-significant on Color FERET and KK has a "single" non-significant row. Its own
  value column, `significance_*.csv` and the Section 14 text give four Color FERET
  pairs in the EdgeFace-XS cell (AVIF/WebP, HEIF/JPEG XL, HEIF/Ours-FAST, JPEG
  XL/Ours-FAST) and 7 of 440 KK tests. The main-findings box and the Figure 44 caption
  name only AVIF/WebP and HEIF/JPEG XL; the figure marks all four.
- **Figure 44.** It was drawn from a per-cell copy. The merged
  `significance_colorferet.csv` differs from it only in the last printed digit of
  five p-values, and the figure renders identically from it.
- **Table 80 (`tab:artifacts`).** It lists per-cell significance files, contamination
  CSVs and a per-sample difficulty file. They are not shipped: they are per-image or
  per-identity, or they come from partial runs. The fairness files are listed under
  `accuracy/` but ship under `fairness/`.

## Items that cannot be regenerated publicly

These items need data or models that are not public. Their shipped aggregates in
`results/` are the only source:

- the identity cosines of Tables 7, 8, 40, 41 (`comparison.json`), key finding F02
  and the sample-difficulty id-cos, which came from a proprietary matcher;
- the side-stream ablation (`subsec:codec-side`, `results/ablation/s7_ablation.csv`),
  whose four checkpoints are not released. Its TopoFR arm would also bundle weights
  without a licence;
- the train/test overlap scan and the side-code linkability test (dataset-level
  summaries only), and the reconstruction check `quality/codec_verify.csv`;
- the Color FERET |yaw| and pitch difficulty descriptors, which need head-pose
  estimates beyond the NIST labels;
- the Color FERET JPEG-AI streams of the earlier run (31.6 % per cell);
- the KK annex crops, and the exact 80 px annex crops;
- the published source-image montage of Figure 3 (the public script builds a montage
  of your own crops instead), and Figure 39, which has no generator.

The other face montages and dataset figures (Figures 1, 4-6, 8, 24-28, 32, 35-38) can
be regenerated from crops you obtain (see [reproduce.md](reproduce.md#paper-map)).
They are never committed because they show dataset faces.

Public users align NIST Color FERET themselves. Their index, pair universe, impostor
sample and 300-crop subsets therefore differ from the paper's, and their Color FERET
numbers are comparable, not identical.

## Reproducibility limits

- **GPU and software stack.** All GPU bitstreams (face1kb, JPEG-AI, CompressAI) and
  embeddings were verified byte- or bit-exact only on an NVIDIA RTX 2080 Ti (Turing,
  sm_75) with torch 2.10.0 / CUDA 12.8, compressai 1.2.8, timm 1.0.24 and
  onnxruntime 1.23.2. `face1kb.load` disables TF32 by default. Other GPU generations
  are unverified; `pytest tests/codec/test_golden.py -k synthetic` checks a GPU
  against golden vectors. The paper fleet also included a Quadro RTX 6000.
- **CPU vs CUDA.** Entropy-coder tables differ between CPU and CUDA builds.
  CUDA-encoded face1kb streams do not decode on a CPU, and CPU encodes differ from
  CUDA encodes.
- **Classical codecs.** Byte-identical classical bitstreams need the pinned wheels
  Pillow 12.2.0, pillow-heif 1.4.0 and pillow-jxl-plugin 1.3.7, whose bundled codec
  libraries determine the bytes.
- **Package pins.** `requirements/paper.txt` pins the full paper environment
  (Python 3.11, numpy 2.4.6, matplotlib 3.11.0). compressai 1.2.8 declares
  `numpy<2`, so a plain `pip install` resolves numpy 1.26. The published numbers
  depend on NumPy 2 float32 semantics in `np.linspace` (EER grid) and `np.quantile`.
  `face1kb.eval` evaluates both explicitly in float32, so its results do not depend on
  the NumPy version. The paper environment used CLIP from the PyPI distribution
  `openai-clip==1.0.1`, whose ViT-B/32 code path is identical to openai/CLIP commit
  ded190a. Hugging Face revisions (LVFace, CVLface), the TopoFR commit and the
  BiRefNet revision were not pinned in the original code. They are pinned here, to
  files byte-identical to those behind the paper's arrays.
- **Embeddings.** The stored arrays were computed in chunks of 256, 128 or 64 crops,
  depending on the matcher and cell. Each array is bit-exact only at its own batch
  size on the same GPU and software stack; otherwise the differences are float noise
  (at most 1.8e-5 absolute). Some stored Ours-ACCURATE arrays (ArcFace, CVLface ViT-B,
  EdgeFace-S) differ from a re-embedding by up to about 1e-3 relative in 6-20 % of the
  rows, at every batch size. The cause is not established; use a tolerance of 1e-3
  relative when comparing.
- **Scorer and metrics.** Exact `metrics.csv` and fairness values need the CUDA
  scorer. CPU-only runs give one-trial EER differences in a few cells. LPIPS
  reproduces to the last bits (6e-8) only on the GPU that produced the study's
  records.
- **CLIP attack.** CLIP runs in fp16, and the upsample backward pass uses atomics.
  Runs from the same start agree on only 25-39 % of pixels, so the CLIP crop sets and
  the CLIP rows of the sanitization tables reproduce statistically, not bit for bit.
  The KK CLIP crop set was crafted on a GPU with another SM count, which changes only
  its random starts.
- **Li-AE attack and proxy.** cuDNN makes Li-AE crafting non-deterministic: about
  1-3 % of the crops differ between runs, even on the paper GPU model. Retraining the
  proxy is not bit-reproducible, because the 400 identities came from a re-packed
  copy of WebFace42M whose mapping to the official folder names is unknown. The
  released proxy weights are the reproduction path. HFC is bit-exact with OpenCV
  4.13.0 and NumPy 2.4.6.
- **Other GPU steps.** BiRefNet preprocessing differs in one pixel by one level for 2
  of 400 crops between runs. The Ours-ACCURATE through-codec cells re-encode within
  8e-6 EER of the paper values. The annex study's warm-started budget search depends
  on how crops are scheduled across workers when a codec's size curve is not
  monotone (seen for 1 of 256 crops in one WebP cell).
- **Codec training.** The trainer reproduces the recipe, not the released weights:
  the early links of both lineages ran on earlier code revisions, and weight
  initialisation was not seeded. The held-out WebFace42M split depends on the
  subject-id strings of the training copy, so the official folder release gives a
  different split, equally identity-disjoint.
- **Figures.** Byte-identical PNGs need matplotlib 3.11.0 and the paper environment's
  fonts. Other versions change pixels only.
