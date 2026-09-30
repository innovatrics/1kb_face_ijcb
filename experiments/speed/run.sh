#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Speed benchmark of the paper (tab:speed, fig:speed-trend): per-crop encode and
# decode time of every codec at its budget-fitting setting, on one CPU core and on
# one GPU, then the table and the figure. Needs the aligned Color FERET crops. The
# CPU stages pin the process to one core (taskset); run them on a quiet host --
# timings taken under load are not a property of the codec.
#
#   1. classical codecs + JPEG-FzT, CPU (1 core), all cells, n=200   ~30 min
#   2. face1kb + CompressAI, GPU, all cells, n=60                    ~15 min GPU
#   3. face1kb + CompressAI, CPU (1 core), 64/112/224 px, 1024 B, n=30   ~20 min
#   4. JPEG-AI, CPU (1 core), all resolutions at 1024 B, n=10       ~10 min
#   5. JPEG-AI, GPU, 112 px, n=38                                    ~3 min GPU
#   6. table + figure from the CSVs of stages 1-5                    seconds
#      (FROM_RESULTS=1: from the shipped results/quality/ instead)
#
# Environment:
#   PYTHON   interpreter (default: python)
#   GPU      GPU id of stages 2 and 5 (default: 0)
#   CORE     CPU core of stages 1, 3 and 4 (default: 0)
#   SKIP     space-separated stage numbers to skip
#   FROM_RESULTS  set to 1 to render stage 6 from results/
#
# Timings depend on the hardware; the realised sizes and settings are
# deterministic. The released face1kb weights select slightly different settings
# and sizes than the paper's face1kb rows (see docs/reproduce_compress.md).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
GPU="${GPU:-0}"
CORE="${CORE:-0}"
SKIP="${SKIP:-}"
S=experiments/speed
Q="$("$PY" -c 'from face1kb import config; print(config.output_dir("quality"))')"

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}
run1() {
	echo "==> [core $CORE] $*"
	taskset -c "$CORE" "$PY" "$@"
}

skip 1 || run1 $S/benchmark_speed.py --device cpu \
	--codecs jpeg,webp,jpeg_xl,avif,heif,jpeg2000,jpeg_fzt --n 200 \
	--out "$Q/speed_cpu_classical.csv"
skip 2 || run $S/benchmark_speed.py --device gpu --gpu "$GPU" \
	--codecs ours_fast,ours_accurate,neural_bmshj2018,neural_mbt2018_mean --n 60 \
	--out "$Q/speed_gpu_learned.csv"
skip 3 || run1 $S/benchmark_speed.py --device cpu \
	--codecs neural_bmshj2018,neural_mbt2018_mean,ours_fast,ours_accurate \
	--resolutions 64,112,224 --budgets 1024 --n 30 --out "$Q/speed_cpu_learned.csv"
skip 4 || run1 $S/benchmark_jpegai_speed.py --device cpu --n 10 \
	--out "$Q/speed_cpu_jpegai.csv"
skip 5 || run $S/benchmark_jpegai_speed.py --device gpu --gpu "$GPU" --resolutions 112 \
	--n 38 --out "$Q/speed_gpu_jpegai.csv"
if ! skip 6; then
	src=()
	[ "${FROM_RESULTS:-0}" = 1 ] && src=(--from-results)
	run $S/generate_speed_table.py "${src[@]}"
fi
