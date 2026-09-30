#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Compression benchmark of the paper (Sections 4-6): every codec at every
# (dataset, resolution, budget) cell, the decoded caches, budget compliance, the
# codec tables, and the JPEG-AI side studies. Starts from the aligned crops, index
# and pairs of experiments/prepare/run.sh. Costs are for one RTX 2080 Ti and one
# CPU core unless stated; both datasets, full grid.
#
#   1. classical codecs + JPEG-FzT, all cells           CPU, ~200 core-hours
#      (Color FERET and AI-Solutions-KK, 64-224 px x 1024/512 B, and the Color FERET
#      112 px 768/960 B budget sweep)
#   2. JPEG-AI: Color FERET all cells + 112 px 768/960 B, AI-Solutions-KK 112/224 px
#                                                        GPU, ~230 GPU-hours
#   3. CompressAI bmshj2018 / mbt2018: Color FERET, first 300 index rows of every
#      cell, all crops at 112 px 768/960 B               GPU, ~4 GPU-hours
#   4. face1kb FAST + ACCURATE (paper_compat), both datasets, all cells, and the
#      Color FERET 112 px 768/960 B sweep                GPU, ~45 GPU-hours
#   5. decoded PNG caches of JPEG-AI and face1kb         GPU, ~35 GPU-hours
#   6. file-size summary, fit-rate tables, compliance and size figures
#                                                        CPU, ~20 min (file stat)
#   7. codec-properties table (fit column from the stage-6 summary; FROM_RESULTS=1
#      uses the shipped summary) and the JPEG-AI operation-point table (from the
#      shipped results/codec_comparison/jpegai_decoders.csv)    CPU, seconds
#   8. JPEG-AI operation points on AI-Solutions-KK (n=600, SOP/BOP/HOP; needs the
#      ArcFace and EdgeFace-XS matchers, scripts/fetch_models.py)   GPU, ~1.5 h
#   9. scan of the stored JPEG-AI streams for collapsed (truncated) files
#                                                        CPU, minutes
#
# The adversarial cells (112_adv_*) are compressed by experiments/adversarial/run.sh
# through compress.py (--suffix); any other crop set can be compressed the same way.
#
# Environment:
#   PYTHON    interpreter (default: python)
#   GPUS      space-separated GPU ids for the GPU stages (default: "0"); the index
#             rows are sharded over them, one process per GPU
#   WORKERS   CPU processes of stage 1 (default: number of cores - 2)
#   DATASETS  datasets (default: "colorferet kk")
#   SKIP      space-separated stage numbers to skip, e.g. SKIP="3 9"
#   FROM_RESULTS  set to 1 to render stage 7 from the shipped results/
#
# All stages resume: existing bitstreams, cached PNGs and run folders are kept.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
SKIP="${SKIP:-}"
DATASETS="${DATASETS:-colorferet kk}"
read -r -a GPU_LIST <<<"${GPUS:-0}"
WORKERS="${WORKERS:-$(($(nproc) > 2 ? $(nproc) - 2 : 1))}"
C=experiments/compress

skip() { [[ " $SKIP " == *" $1 "* ]]; }
use() { [[ " $DATASETS " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}
# gpu_grid ARGS...: compress.py ARGS sharded over $GPUS (one process per GPU)
gpu_grid() {
	local n=${#GPU_LIST[@]} i pids=()
	for i in "${!GPU_LIST[@]}"; do
		echo "==> [gpu ${GPU_LIST[$i]}] $C/compress.py $* --shard $i:$n"
		"$PY" $C/compress.py "$@" --gpu "${GPU_LIST[$i]}" --shard "$i:$n" &
		pids+=($!)
	done
	for i in "${pids[@]}"; do wait "$i"; done
}

if ! skip 1; then
	for ds in $DATASETS; do
		run $C/compress.py --dataset "$ds" --codecs cpu --workers "$WORKERS"
	done
	if use colorferet; then
		run $C/compress.py --dataset colorferet --codecs cpu --resolutions 112 \
			--budgets 768,960 --workers "$WORKERS"
	fi
fi

if ! skip 2; then
	if use colorferet; then
		gpu_grid --dataset colorferet --codecs jpeg_ai
		gpu_grid --dataset colorferet --codecs jpeg_ai --resolutions 112 --budgets 768,960
	fi
	if use kk; then
		gpu_grid --dataset kk --codecs jpeg_ai --resolutions 112,224
	fi
fi

if ! skip 3 && use colorferet; then
	# the paper ran CompressAI on the first 300 index rows (a single process, so
	# that the 300-row slice is the same whatever the number of GPUs)
	run $C/compress.py --dataset colorferet --codecs compressai --limit 300 \
		--gpu "${GPU_LIST[0]}"
	gpu_grid --dataset colorferet --codecs compressai --resolutions 112 --budgets 768,960
fi

if ! skip 4; then
	for ds in $DATASETS; do
		gpu_grid --dataset "$ds" --codecs ours
	done
	if use colorferet; then
		gpu_grid --dataset colorferet --codecs ours --resolutions 112 --budgets 768,960
	fi
fi

if ! skip 5; then
	gpus=$(
		IFS=,
		echo "${GPU_LIST[*]}"
	)
	# shellcheck disable=SC2086  # DATASETS is a word list
	run $C/decode_cache.py --datasets "$(echo $DATASETS | tr ' ' ',')" \
		--budgets 1024,512,768,960 --gpus "$gpus"
fi

if ! skip 6; then
	run $C/generate_file_size_boxplots.py
fi

if ! skip 7; then
	src=()
	[ "${FROM_RESULTS:-0}" = 1 ] && src=(--from-results)
	run $C/generate_codec_properties_table.py "${src[@]}"
	run $C/generate_jpegai_speed_table.py
fi

if ! skip 8 && use kk; then
	run $C/measure_jpegai_kk.py --phase all --gpu "${GPU_LIST[0]}"
fi

if ! skip 9; then
	# shellcheck disable=SC2086
	run $C/scan_jpegai_collapses.py --datasets "$(echo $DATASETS | tr ' ' ',')"
fi
