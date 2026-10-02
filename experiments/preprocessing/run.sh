#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Preprocessing and crop-tightness ablations (paper Section "Ablations and
# Preprocessing"):
#
#   1. clean-crop ablation: EER of the crop-tightness presets and preprocessing
#      operators on Color FERET 112 px (4 anchors)           CPU, ~10 min, ~6 GB RAM
#   2. preprocessing through the codec on AI-Solutions-KK 224 px / 1024 B
#      (std, A1-A4, B2 x WebP, AVIF, Ours-ACCURATE; ArcFace + EdgeFace-XS)
#        encode, WebP/AVIF                                   CPU, ~40 min
#        encode, Ours-ACCURATE (4,800 encodes)                1 GPU, ~45 min
#        aggregate (embedding + EER)                          1 GPU, ~5 min
#   3. the four LaTeX tables                                  CPU, seconds
#   4. figures (show dataset faces; keep them local):
#        preprocessing_impact.png (WebP headroom)             1 GPU or CPU, ~1 min
#        alignment_check.png, alignment_variants.png          CPU, ~10 s
#        preproc_through_codec_kk.png (MONTAGE_SAMPLES)       1 GPU, ~1 min
#
# Needs for step 1: the crop sets aligned_112_{tight,mid,fill} and
# aligned_112_{A1..A4,B1,B2,C1,C2} (experiments/prepare/run.sh steps 4 and 5) and
# their embeddings for the four anchors:
#   python experiments/embed/compute_embeddings.py --datasets colorferet \
#     --models anchor --kind aligned --resolutions 112 \
#     --suffixes ,_tight,_mid,_fill,_A1,_A2,_A3,_A4,_B1,_B2,_C1,_C2
# Step 2 needs the AI-Solutions-KK aligned_224 crops, index and pairs, the face1kb
# weights and the MediaPipe selfie segmenter (downloaded on first use).
#
# Environment: PYTHON (default python), DEVICE (default cuda), SKIP (space-separated
# step numbers), MONTAGE_SAMPLES ('subject/stem:label,...' KK crops of the
# through-codec figure; the figure is skipped when unset).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
SKIP="${SKIP:-}"
E=experiments/preprocessing

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}

if ! skip 1; then
	run "$E/ablation.py"
fi

if ! skip 2; then
	run "$E/through_codec.py" --phase encode --codecs webp,avif --device "$DEVICE"
	for op in std A1 A2 A3 A4 B2; do
		run "$E/through_codec.py" --phase encode --codecs ours_accurate \
			--operators "$op" --device "$DEVICE"
	done
	run "$E/through_codec.py" --phase aggregate --device "$DEVICE"
fi

if ! skip 3; then
	run "$E/make_tables.py"
fi

if ! skip 4; then
	run "$E/impact_figure.py" --device "$DEVICE"
	run "$E/montages.py"
	if [ -n "${MONTAGE_SAMPLES:-}" ]; then
		run "$E/through_codec.py" --montage-only --montage-samples "$MONTAGE_SAMPLES" \
			--device "$DEVICE"
	fi
fi
