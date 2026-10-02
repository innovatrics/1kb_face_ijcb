#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Recompression study (paper Section "Compressed-on-Compressed Recompression"):
# the 8 x 8 chain matrices of Color FERET and AI-Solutions-KK at 112 px, 1024 and
# 512 B, their EdgeFace-XS EER, the ArcFace rescore of Color FERET, and the four
# LaTeX tables.
#
# Needs: aligned 112 px crops, index.csv and pairs.parquet (experiments/prepare), the
# single-pass compressed cells compressed/112px_<budget>B/<codec> of the benchmark
# (experiments/compress), the face1kb weights (git lfs pull) and the matchers
# (scripts/fetch_models.py).
#
# Cost (one 8-core CPU, one RTX 2080 Ti class GPU):
#   produce  CPU only. 49 cells x 2 budgets: Color FERET (11,335 crops) about 1.1 M
#            double encodes, ~20 h with 8 workers; AI-Solutions-KK (first 2,500
#            crops) ~4-5 h. About 1.5 GB of files under WORK_ROOT/<ds>/recompressed.
#   score    1 GPU, low use (the time is file decoding, CPU): ~1-2 h per dataset and
#            budget (49 cells + 7 single-pass references of up to 17.5 k crops).
#   ours     1 GPU: ~25 min per dataset and budget (120 crops, ~1,000
#            face1kb-ACCURATE encodes).
#   tables   CPU, seconds.
# Every stage resumes: existing files and scored cells are skipped.
#
# Environment: PYTHON (default python), WORKERS (default 8), DEVICE (default cuda),
# SKIP (space-separated stage names to skip: produce score ours arcface tables).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-cuda}"
SKIP="${SKIP:-}"
E=experiments/recompression

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}

stages=""
skip produce || stages="produce"
skip score || stages="${stages:+$stages,}score"
skip ours || stages="${stages:+$stages,}ours"

if [ -n "$stages" ]; then
	for budget in 1024 512; do
		run "$E/recompress.py" --dataset colorferet --budget "$budget" \
			--stages "$stages" --workers "$WORKERS" --device "$DEVICE"
		run "$E/recompress.py" --dataset kk --budget "$budget" --limit 2500 \
			--stages "$stages" --workers "$WORKERS" --device "$DEVICE"
	done
fi

# independent-matcher rescore of Color FERET (classical block and Ours rows)
if ! skip arcface; then
	for budget in 1024 512; do
		run "$E/recompress.py" --dataset colorferet --budget "$budget" \
			--stages score,ours --model arcface_antelopev2 --device "$DEVICE"
	done
fi

if ! skip tables; then
	run "$E/make_tables.py"
fi
