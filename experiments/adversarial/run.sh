#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Adversarial robustness and sanitization (paper section "Adversarial Robustness and
# Sanitization"), end to end:
#
#   1. craft     the adversarial 112 px crop sets aligned_112_adv_<attack>_<eps>
#   2. compress  clean-crop codecs on the adversarial sets, Ours-FAST/ACCURATE as
#                a defence (paper_compat budget accounting)
#   3. embed     the adversarial crops and their decoded versions, 4 anchor matchers
#   4. analyze   OUTPUT_ROOT/adversarial/sanitization.csv
#   5. tables    sanitization_{hfc,clip,liae,hfc_kk}.tex, attack_strength.tex,
#                ours_defense.tex
#
# Paper scope (subsets are the first N crops in index.csv order):
#   Color FERET      HFC    all 11,335   6 classical codecs, Ours-FAST, Ours-ACCURATE
#                    HFC    300          JPEG-FzT (4 matchers); JPEG-AI, bmshj2018,
#                                        mbt2018 (arcface_antelopev2, lvface_l)
#                    Li-AE  all 11,335   6 classical codecs, JPEG-FzT; Ours at eps 0.06
#                    CLIP   2,000        6 classical codecs, JPEG-FzT, Ours
#   AI-Solutions-KK  HFC, CLIP, Li-AE    2,000   6 classical codecs, JPEG-FzT, Ours
# eps is 0.03, 0.06 and 0.10 throughout; budgets 1024 and 512 B.
#
# Cost (estimates for one RTX 2080 Ti from the per-crop timings of the codecs and
# attacks; the run is resumable, every step skips outputs that exist):
#   1 craft     HFC CPU ~3 min; Li-AE GPU ~5 min (+ ~5 min CPU for the paper's random
#               starts); CLIP GPU ~3 min
#   2 compress  ~20 GPU-hours, dominated by Ours-ACCURATE (~0.4 s per encode, 139k
#               encodes per face1kb variant); JPEG-AI ~1.5 GPU-hours (1,800 encodes);
#               the classical codecs several CPU-hours
#   3 embed     ~3 GPU-hours (about 1.5 M decoded crops x 4 matchers, Ours decode)
#   4 analyze   CPU, minutes (reads the embedding arrays)
#   5 tables    CPU, seconds
#
# Prerequisites:
#   * pip install -e ".[adversarial,codecs,eval]" (plus ".[jpegai]" and
#     scripts/setup_jpegai.sh for the JPEG-AI cells), git lfs pull (codec and Li-AE
#     proxy weights), python scripts/fetch_models.py --only anchors
#   * the clean aligned crops and index.csv of both datasets
#     (experiments/prepare/run.sh)
#   * the clean 112 px benchmark: bitstreams and embeddings aligned_112 and
#     <codec>_112_<budget> of the four anchors (experiments/compress/run.sh,
#     experiments/embed/run.sh); step 3 computes missing clean arrays of the
#     full-coverage codecs as well
#
# Environment:
#   PYTHON    interpreter (default: python)
#   GPU       physical GPU id as nvidia-smi numbers it (default: 0; exported as
#             CUDA_VISIBLE_DEVICES with CUDA_DEVICE_ORDER=PCI_BUS_ID, and passed to
#             compute_embeddings.py --gpus)
#   WORKERS   processes of the CPU codecs (default: compress.py's default)
#   SKIP      space-separated step numbers to skip, e.g. SKIP="1 2"
#   DATASETS  datasets (default: "colorferet kk")

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
SKIP="${SKIP:-}"
DATASETS="${DATASETS:-colorferet kk}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${GPU:-0}"

HERE=experiments/adversarial
COMPRESS=experiments/compress/compress.py
EMBED=experiments/embed/compute_embeddings.py
EPS=(003 006 010)
CLASSICAL=jpeg,jpeg2000,webp,jpeg_xl,avif,heif
OURS=ours_fast,ours_accurate
ANCHORS=arcface_antelopev2,lvface_l,topofr_r100,edgeface_xs
TWO_ANCHORS=arcface_antelopev2,lvface_l

skip() { [[ " $SKIP " == *" $1 "* ]]; }
selected() { [[ " $DATASETS " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}
# suffixes ATTACK [EPS_TAGS...]: comma-joined crop-set suffixes
suffixes() {
	local attack=$1 out="" t
	shift
	for t in "${@:-${EPS[@]}}"; do out="${out:+$out,}_adv_${attack}_${t}"; done
	echo "$out"
}
# compress DATASET ATTACK LIMIT CODECS [EPS_TAGS...]
compress() {
	local ds=$1 attack=$2 limit=$3 codecs=$4 sfx
	shift 4
	for sfx in $(suffixes "$attack" "$@" | tr ',' ' '); do
		run $COMPRESS --dataset "$ds" --codecs "$codecs" --resolutions 112 \
			--budgets 1024,512 --align-suffix "$sfx" ${limit:+--limit "$limit"} \
			${WORKERS:+--workers "$WORKERS"}
	done
}
# embed DATASET SUFFIXES LIMIT CODECS MODELS [EXTRA...]: aligned and compressed
# sources of the suffixes ("" = the clean crops)
embed() {
	local ds=$1 sfx=$2 limit=$3 codecs=$4 models=$5
	shift 5
	run $EMBED --datasets "$ds" --models "$models" --resolutions 112 \
		--budgets 1024,512 --codecs "$codecs" --suffixes "$sfx" --gpus "${GPU:-0}" \
		${limit:+--limit "$limit"} "$@"
}

# ---------------------------------------------------------------- 1. craft
if ! skip 1; then
	if selected colorferet; then
		run $HERE/craft.py --dataset colorferet --attack hfc
		run $HERE/craft.py --dataset colorferet --attack liae --device cuda
		run $HERE/craft.py --dataset colorferet --attack clip --limit 2000 \
			--device cuda
	fi
	if selected kk; then
		for attack in hfc liae clip; do
			run $HERE/craft.py --dataset kk --attack "$attack" --limit 2000 \
				--device cuda
		done
	fi
fi

# ---------------------------------------------------------------- 2. compress
if ! skip 2; then
	if selected colorferet; then
		compress colorferet hfc "" "$CLASSICAL,$OURS"
		compress colorferet hfc 300 jpeg_fzt,jpeg_ai,neural_bmshj2018,neural_mbt2018_mean
		compress colorferet liae "" "$CLASSICAL,jpeg_fzt"
		compress colorferet liae "" "$OURS" 006
		compress colorferet clip 2000 "$CLASSICAL,jpeg_fzt,$OURS"
	fi
	if selected kk; then
		for attack in hfc liae clip; do
			compress kk "$attack" 2000 "$CLASSICAL,jpeg_fzt,$OURS"
		done
	fi
fi

# ---------------------------------------------------------------- 3. embed
# The clean arrays (suffix "") are those of the benchmark; the first call of each
# dataset computes the ones that are missing, over all crops. The clean JPEG-AI and
# CompressAI arrays of Color FERET come from the benchmark run.
if ! skip 3; then
	for ds in colorferet kk; do
		selected "$ds" || continue
		embed "$ds" "" "" "$CLASSICAL,jpeg_fzt,$OURS" "$ANCHORS"
	done
	if selected colorferet; then
		embed colorferet "$(suffixes hfc)" "" "$CLASSICAL,$OURS" "$ANCHORS"
		embed colorferet "$(suffixes hfc)" 300 jpeg_fzt "$ANCHORS" --kind compressed
		embed colorferet "$(suffixes hfc)" 300 \
			jpeg_ai,neural_bmshj2018,neural_mbt2018_mean "$TWO_ANCHORS" \
			--kind compressed --jpegai-live
		embed colorferet "$(suffixes liae)" "" "$CLASSICAL,jpeg_fzt" "$ANCHORS"
		embed colorferet "$(suffixes liae 006)" "" "$OURS" "$ANCHORS" --kind compressed
		embed colorferet "$(suffixes clip)" 2000 "$CLASSICAL,jpeg_fzt,$OURS" "$ANCHORS"
	fi
	if selected kk; then
		embed kk "$(suffixes hfc),$(suffixes liae),$(suffixes clip)" 2000 \
			"$CLASSICAL,jpeg_fzt,$OURS" "$ANCHORS"
	fi
fi

# ---------------------------------------------------------------- 4. analyze
if ! skip 4; then
	run $HERE/analyze.py --datasets "$(echo "$DATASETS" | tr ' ' ',')"
fi

# ---------------------------------------------------------------- 5. tables
if ! skip 5; then
	run $HERE/generate_sanitization_table.py
fi
echo "==> done"
