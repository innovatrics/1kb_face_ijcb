#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Embedding arrays of every crop set with the face-recognition roster (the input of
# experiments/accuracy/run.sh and of the fairness, recompression, preprocessing,
# resolution and adversarial studies). Sources are discovered on disk, so run this
# after the compression stage; finished arrays are skipped, so it can be re-run.
#
#   1. core grid: aligned 64-224 px and the 12 codecs at      GPU; about 250 arrays
#      1024 / 512 B, all 14 matchers, both datasets           per matcher, ~35-45 s
#                                                             each on an RTX 2080 Ti;
#                                                             ~30 GPU-hours
#   2. Color FERET budget sweep (112 px, 768 / 960 B),        GPU, ~1 h
#      four anchors
#   3. crop variants at 112 px, four anchors: preprocessing   GPU, ~1-3 h (only the
#      (_A1.._C2), crop tightness (_tight/_mid/_fill) and     variant folders that
#      adversarial (_adv_<attack>_<eps>) crops and their      exist are embedded)
#      compressed cells
#
# Model weights (5.7 GiB) are fetched once from the official sources
# (scripts/fetch_models.py) before the first active step; with all three steps in SKIP
# nothing is downloaded. JPEG-AI cells are embedded only from their decoded PNG
# cache, and the face1kb cells are much faster with it: run
# experiments/compress/decode_cache.py --budgets 1024,512,768,960 first (step 5 of
# experiments/compress/run.sh), plus decode_cache.py --align-suffix <suffix> for each
# crop variant of step 3 whose JPEG-AI cells are needed (see
# docs/reproduce_accuracy.md). JPEG-AI cells without a cache are skipped.
#
# Environment:
#   PYTHON   interpreter (default: python)
#   GPUS     comma list of GPU ids (default: 0); one worker process per GPU
#   SKIP     space-separated step numbers to skip, e.g. SKIP="3"
#   DATASETS comma list (default: colorferet,kk)
#
# Usage: [VAR=value ...] bash experiments/embed/run.sh   (-h / --help prints this text)

set -euo pipefail

case "${1:-}" in
-h | --help)
	sed -n '4,/^$/{s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"
	exit 0
	;;
"") ;;
*)
	echo "error: unknown argument '$1' (configure with environment variables, see --help)" >&2
	exit 2
	;;
esac

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
SKIP="${SKIP:-}"
GPUS="${GPUS:-0}"
DATASETS="${DATASETS:-colorferet,kk}"
E=experiments/embed/compute_embeddings.py

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}

if ! skip 1 || ! skip 2 || ! skip 3; then
	run scripts/fetch_models.py --only all
fi

if ! skip 1; then
	run $E --datasets "$DATASETS" --models all --gpus "$GPUS"
fi
if ! skip 2; then
	run $E --datasets colorferet --models anchor --resolutions 112 \
		--budgets 768,960 --gpus "$GPUS"
fi
if ! skip 3; then
	variants="_A1,_A2,_A3,_A4,_B1,_B2,_C1,_C2,_tight,_mid,_fill"
	for attack in hfc clip liae; do
		for eps in 003 006 010; do
			variants="$variants,_adv_${attack}_${eps}"
		done
	done
	run $E --datasets "$DATASETS" --models anchor --resolutions 112 \
		--suffixes "$variants" --gpus "$GPUS"
fi
