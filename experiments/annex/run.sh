#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# ISO/IEC 29794-5 Annex E/F parameter study (paper section "The ISO/IEC 29794-5 Annex
# E/F parameter tables"), end to end:
#
#   0. prep      evaluation subsets, crop cache 56-224 px, 106-point landmarks
#   1. baseline  uncompressed references, both datasets, 5 matchers
#   2. exp1      our configurations vs the Annex E/F configurations, both datasets
#   3. grid      Stage A: 336 cells, Color FERET 2000-crop prefix, 2 matchers
#   4. jpegai    JPEG-AI arm: 16 cells, first 500 Color FERET crops, 5 matchers
#   5. stage B   score the prefix, flag sweep around the Stage-A winners (62 cells),
#                score the prefix again -> scores_pop2000.csv
#   6. confirm   the 6 winners at the full population, both datasets, 5 matchers
#   7. score     every cell at its own population -> scores.csv
#   8. ci        paired subject-level bootstrap -> ci.csv
#   9. tables    annex_{exp1,ci,joint,axes,manip,selfsim,bytes,jpegai}.tex
#
# scores_pop2000.csv is final after step 5, before the confirmation run: step 6
# records its cells under the arm "winner" and recomputes their arrays, and scoring
# the prefix again afterwards would change tab:annex-joint and tab:annex-bytes. The
# shipped results were scored in this order.
#
# Cost (one RTX 2080 Ti; compression is CPU-bound, about 7 s per cell of 256 crops
# with 16 workers):
#   0 prep      ~3 min per dataset (landmarks on the GPU)
#   1-2         ~30 min per dataset
#   3 grid      ~4 h (336 cells x 2000 crops)
#   4 jpegai    ~8 GPU-hours (2-3 s per JPEG-AI encode, 8,000 crops)
#   5 stage B   ~1 h (62 cells x 2000 crops, plus ~5 min per scoring pass)
#   6 confirm   ~15 min per dataset
#   7-9         ~15 min on the GPU (scoring, 200 bootstrap resamples); tables seconds
#
# Prerequisites:
#   * pip install -e ".[codecs,eval,jpegai]", scripts/setup_jpegai.sh (JPEG-AI arm),
#     python scripts/fetch_models.py --only anchors,cvlface_vit_b
#   * the command-line tools cwebp (libwebp) and jpegtran (libjpeg-turbo with
#     arithmetic coding)
#   * the aligned crops aligned_<res> of both datasets and index.csv
#     (experiments/prepare/run.sh; aligned_56 is derived there, 80 px is resampled
#     from 224 px by prep.py when no aligned_80 folder exists)
#
# Environment:
#   PYTHON   interpreter (default: python)
#   DEVICE   torch device of the matchers and the scoring (default: cuda)
#   WORKERS  compression processes (default: 24)
#   SKIP     space-separated step numbers to skip, e.g. SKIP="4"

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
DEVICE="${DEVICE:-cuda}"
WORKERS="${WORKERS:-24}"
SKIP="${SKIP:-}"

HERE=experiments/annex
skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" -u "$@"
}
sweep() { run $HERE/sweep.py --workers "$WORKERS" --device "$DEVICE" "$@"; }
score_prefix() {
	run $HERE/score.py --datasets colorferet --pop-limit 2000 \
		--out scores_pop2000.csv --device "$DEVICE"
}

if ! skip 0; then
	ctx=0
	[[ "$DEVICE" == cpu ]] && ctx=-1
	run $HERE/prep.py --ctx-id "$ctx"
fi
if ! skip 1; then
	for d in colorferet kk; do sweep --stage baseline --dataset "$d"; done
fi
if ! skip 2; then
	for d in colorferet kk; do sweep --stage exp1 --dataset "$d"; done
fi
if ! skip 3; then
	sweep --stage A --dataset colorferet --limit 2000
fi
if ! skip 4; then
	sweep --stage jpegai --dataset colorferet --limit 500
fi
if ! skip 5; then
	score_prefix
	run $HERE/stageb.py
	sweep --stage B --dataset colorferet --limit 2000
	score_prefix
fi
if ! skip 6; then
	run $HERE/stageb.py --confirm
	for d in colorferet kk; do sweep --stage confirm --dataset "$d"; done
fi
if ! skip 7; then
	run $HERE/score.py --datasets colorferet,kk --device "$DEVICE"
fi
if ! skip 8; then
	run $HERE/ci.py --device "$DEVICE" --reps 200
fi
if ! skip 9; then
	run $HERE/generate_annex_tables.py
fi
echo "==> done"
