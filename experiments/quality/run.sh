#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Image-quality study of the paper (Section 6 and Figure 1), from the compressed cells
# under $FACE1KB_WORK_ROOT/<dataset>/compressed/ (experiments/compress) and the
# aligned crops under $FACE1KB_DATA_ROOT.
#
#   1. PSNR / SSIM / MS-SSIM / LPIPS / DISTS of every       1 GPU, ~10 h for both
#      compressed crop; per-image shards stay in              datasets (decoded caches
#      $FACE1KB_WORK_ROOT/<ds>/quality/                       for JPEG-AI; decoding
#      -> quality/quality_<ds>.csv, quality_summary.csv       JPEG-AI live adds ~40 h)
#   2. face image quality (FIQ), 8 cells x 250 crops         1 GPU, ~1 h
#      -> quality/fiq_<ds>.csv
#   3. quality matrix, budget sweep, FIQ grid and figure     CPU, seconds
#      -> tables/quality_matrix.tex, budget_sweep.tex, fiq_grid.tex,
#         figures/fiq_by_codec.png
#   4. decoded-crop montages (Figures 1, 8, 24-28)           1 GPU, ~1 min
#      -> figures/visual_*.png, budget_224_*.png, comparison_1kb.png,
#         abstract_codec_strip.png. These show dataset faces: keep them local,
#         never commit or redistribute them.
#
# Outputs go to $FACE1KB_OUTPUT_ROOT.
#
# Environment:
#   PYTHON        interpreter (default: python)
#   DEVICE        GPU device (default: cuda)
#   SKIP          space-separated step numbers to skip, e.g. SKIP="1 2 4"
#   PAPER_HEADER  set to 1 to write the first comment line of the published tables
#                 (it names the script that originally produced them)
#   FROM_RESULTS  set to 1 to render step 3 (and the captions of step 4) from the
#                 shipped paper results (<repo>/results/quality)
#   FIQ_SHADING   shading rule of the FIQ grid: printed (default) or raw (the rule
#                 of the published table)

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
SKIP="${SKIP:-}"
SRC=()
if [ "${FROM_RESULTS:-0}" = 1 ]; then
	SRC=(--from-results)
fi
if [ "${PAPER_HEADER:-0}" = 1 ]; then
	SRC+=(--paper-header)
fi

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}

if ! skip 1; then
	# resumable: cells whose shard exists are skipped
	run experiments/quality/measure_quality.py --device "$DEVICE"
	# the 768 / 960 B face1kb cells of Color FERET (intermediate budget points)
	run experiments/quality/measure_quality.py --datasets colorferet \
		--codecs ours_fast,ours_accurate --resolutions 112 --budgets 768,960 \
		--device "$DEVICE"
fi
if ! skip 2; then
	run experiments/quality/fiq.py --device "$DEVICE"
fi
if ! skip 3; then
	run experiments/quality/quality_tables.py ${SRC[@]+"${SRC[@]}"}
	run experiments/quality/fiq_report.py ${SRC[@]+"${SRC[@]}"} --shading "${FIQ_SHADING:-printed}"
fi
if ! skip 4; then
	if [ "${FROM_RESULTS:-0}" = 1 ]; then
		run experiments/quality/visual_quality.py --from-results --device "$DEVICE"
	else
		run experiments/quality/visual_quality.py --device "$DEVICE"
	fi
fi
echo "done"
