#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Verification accuracy, significance and the accuracy tables and figures of the
# paper (Sections 5.2-5.6, 7.6, 7.8 and 14), from the embedding arrays of
# experiments/embed/run.sh.
#
#   1. metrics.csv: every (dataset, model, source)       1 GPU (CUDA scorer), RTX 2080 Ti:
#      array, about 4,600 arrays                          ~4 s (CF) / ~9 s (KK) per array
#                                                         with CIs, ~8 h in total; with
#                                                         DEVICES="cuda:0 cuda:1" the two
#                                                         datasets run in parallel
#   2. significance CSVs: 4 anchors x CF 112/1024 and     CPU (NumPy scorer), ~5-8 min per
#      KK {112,224}/1024 (12 cells, 55 pairs each)        CF cell, ~12 min per KK cell
#   3. identity-cosine tail (ArcFace, 112 px, 512 B)     CPU, < 1 min
#   4. tables and figures from the CSVs: rate_eer (8), frr_far / frr_far_summary /
#      model_effect (13 + 10 figures), best_config, h4_budget (+ figure),
#      heldout_cvlface, idcos_tail, posthoc_stats, sig_cf_edgeface, sig_kk
#                                                        CPU, ~1 min
#
# Environment:
#   PYTHON        interpreter (default: python)
#   SKIP          space-separated step numbers to skip, e.g. SKIP="1 2"
#   DEVICES       scoring devices of step 1, one per dataset process (default:
#                 "cuda"; with two entries Color FERET and AI-Solutions-KK run in
#                 parallel)
#   FROM_RESULTS  set to 1 to run step 4 on the shipped results/ instead of
#                 outputs/ (renders the paper tables without any data)
#   PAPER_HEADER  set to 1 to write the comment lines of the published tables
#                 (byte-identical output)
#
# Outputs: outputs/accuracy/{metrics,significance_<dataset>}.csv, posthoc_scalars.txt,
# outputs/quality/idcos_tail_512.csv, outputs/tables/*.tex, outputs/figures/*.png
# (FACE1KB_OUTPUT_ROOT).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
SKIP="${SKIP:-}"
read -r -a DEVS <<<"${DEVICES:-cuda}"
A=experiments/accuracy

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}

# ------------------------------------------------------------- 1. metrics.csv
if ! skip 1; then
	if [ "${#DEVS[@]}" -ge 2 ]; then
		run $A/compute_accuracy.py --datasets colorferet --models all \
			--device "${DEVS[0]}" --no-cpu-fallback --no-combined &
		run $A/compute_accuracy.py --datasets kk --models all \
			--device "${DEVS[1]}" --no-cpu-fallback --no-combined &
		wait
		run $A/compute_accuracy.py --merge
	else
		run $A/compute_accuracy.py --models all --device "${DEVS[0]}" --no-cpu-fallback
	fi
fi

# ------------------------------------------------------- 2. significance tests
if ! skip 2; then
	run $A/compute_significance.py --datasets colorferet --models anchor \
		--res 112 --budgets 1024
	run $A/compute_significance.py --datasets kk --models anchor \
		--res 112,224 --budgets 1024
fi

# ------------------------------------------------------ 3. identity-cosine tail
if ! skip 3; then
	run $A/idcos_tail.py --budget 512
fi

# ---------------------------------------------------------- 4. tables, figures
if ! skip 4; then
	src=()
	[ "${FROM_RESULTS:-0}" = 1 ] && src=(--from-results)
	hdr=()
	[ "${PAPER_HEADER:-0}" = 1 ] && hdr=(--paper-header)
	run $A/rate_eer.py "${src[@]}" "${hdr[@]}"
	run $A/frr_far.py "${src[@]}" "${hdr[@]}"
	run $A/best_config.py "${src[@]}"
	run $A/h4_curve.py "${src[@]}" "${hdr[@]}"
	run $A/heldout_table.py "${src[@]}" "${hdr[@]}"
	if [ "${FROM_RESULTS:-0}" = 1 ]; then
		run $A/idcos_tail.py --render-only --from-results
	elif skip 3; then
		run $A/idcos_tail.py --render-only
	fi
	run $A/posthoc_stats.py "${src[@]}" "${hdr[@]}"
	run $A/posthoc_stats.py "${src[@]}" --exclude-prefix edgeface
	run $A/significance_tables.py "${src[@]}" "${hdr[@]}"
fi
