#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Sample difficulty (paper Section "Trivial and Difficult Samples"): per-image
# compression difficulty of 300 random crops per dataset at 112 px, 1024 and 512 B,
# for the six classical codecs, JPEG-AI and both face1kb variants; then the shipped
# aggregates, the two tables, the predictor figure and the (local-only) montage.
#
#   1. compute.py   1 GPU: ~20 s per crop and budget pair without JPEG-AI (about
#                   3-4 h for 2 x 300 crops, JPEGAI=0); JPEG-AI adds ~1-2 h (2,400
#                   streams of up to three reference encodes and one decode)
#   2. analyze.py   CPU, seconds
#   3. make_tables.py  CPU, seconds
#   4. montage      1 GPU, ~1 min (20 re-encodes); shows dataset faces, keep it local
#
# Needs: aligned 112 px crops, Color FERET labels.csv and AI-Solutions-KK
# attributes.csv (experiments/prepare), the face1kb weights, the id-cos matcher
# (default cvlface_ir101; scripts/fetch_models.py) and, for JPEG-AI, the reference
# software (scripts/setup_jpegai.sh).
#
# The paper scored id_cos with a proprietary matcher that is not distributed; a run
# with a public matcher gives different difficulty values. The per-image CSV
# (OUTPUT_ROOT/difficulty/sample_difficulty.csv) holds per-image dataset attributes:
# keep it local.
#
# Environment: PYTHON (default python), DEVICE (default cuda), ID_MODEL (default:
# the public held-out matcher), JPEGAI=0 to skip JPEG-AI, SKIP (step numbers).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
SKIP="${SKIP:-}"
E=experiments/difficulty

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}

if ! skip 1; then
	args=(--device "$DEVICE")
	[ -n "${ID_MODEL:-}" ] && args+=(--id-model "$ID_MODEL")
	[ "${JPEGAI:-1}" = 0 ] && args+=(--no-jpegai)
	run "$E/compute.py" "${args[@]}"
fi
if ! skip 2; then
	run "$E/analyze.py"
fi
if ! skip 3; then
	run "$E/make_tables.py"
fi
if ! skip 4; then
	run "$E/compute.py" --montage-only --device "$DEVICE"
fi
