#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Fairness study of the paper (Section 11), from the embedding arrays of the four
# anchor matchers under $FACE1KB_WORK_ROOT/<dataset>/embeddings/ (experiments/accuracy)
# and the attributes (Color FERET labels.csv, AI-Solutions-KK attributes.csv;
# experiments/prepare).
#
#   1. subgroup EER and disparity, every core-grid source   1 GPU, ~3-4 h, ~16 GB RAM
#      -> fairness/fairness_<ds>.csv, fairness_disparity_<ds>.csv
#   2. differential FMR, 112 px, 512 / 1024 B               1 GPU, ~20 min
#      -> fairness/fmr_fairness_<ds>.csv
#   3. KK skin-tone disparity CIs, 112 px / 1024 B          1 GPU, ~30 min
#      -> fairness/disparity_ci_kk.csv
#   4. LaTeX tables of Section 11                           CPU, seconds
#      -> tables/fairness_*.tex, disparity_ci_kk.tex, fmr_fairness_cf.tex
#
# Scoring uses the CUDA scorer, as the published tables; set DEVICE=cpu for the NumPy
# scorer (can move an EER by one impostor trial). Outputs go to $FACE1KB_OUTPUT_ROOT.
#
# Environment:
#   PYTHON        interpreter (default: python)
#   DEVICE        scoring device (default: cuda)
#   SKIP          space-separated step numbers to skip, e.g. SKIP="1 2 3" renders
#                 only the tables
#   PAPER_HEADER  set to 1 to write the first comment line of the published tables
#                 (it names the script that originally produced them)
#   FROM_RESULTS  set to 1 to render the tables of step 4 from the shipped paper
#                 results (<repo>/results/fairness) instead of step 1-3 outputs

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
SKIP="${SKIP:-}"

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}

if ! skip 1; then
	run experiments/fairness/subgroup.py --device "$DEVICE"
fi
if ! skip 2; then
	run experiments/fairness/fmr_fairness.py --device "$DEVICE"
fi
if ! skip 3; then
	run experiments/fairness/disparity_ci.py --device "$DEVICE"
fi
if ! skip 4; then
	ARGS=()
	if [ "${FROM_RESULTS:-0}" = 1 ]; then
		ARGS+=(--from-results)
	fi
	if [ "${PAPER_HEADER:-0}" = 1 ]; then
		ARGS+=(--paper-header)
	fi
	run experiments/fairness/tables.py ${ARGS[@]+"${ARGS[@]}"}
fi
echo "done"
