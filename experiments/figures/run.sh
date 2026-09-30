#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Summary figures of the paper (Figures 22, 40-45) and the shipped results/ tree.
#
#   1. render the seven summary figures and derived_stats.json   CPU, ~15 s, <1 GB RAM
#      from the aggregate CSVs (no data, no GPU)
#   2. optional (COLLECT=1): copy the aggregates of a full run    CPU, ~5 s
#      from $FACE1KB_OUTPUT_ROOT into results/
#
# With COLLECT=1, step 1 always reads $FACE1KB_OUTPUT_ROOT (FROM is ignored) and runs
# before step 2, because report_summary/derived_stats.json, one of the collected
# files, is written by step 1. Step 2 keeps the shipped copy of a file that
# $FACE1KB_OUTPUT_ROOT does not have (--keep-shipped): the public pipeline does not
# produce every shipped file (e.g. codec_properties.csv, the findings register).
#
# Environment:
#   FROM     where step 1 reads the aggregates: "results" (default; the shipped
#            paper results) or "outputs" ($FACE1KB_OUTPUT_ROOT, after running the
#            other experiment areas)
#   COLLECT  set to 1 to render from outputs and then run step 2 (overwrites
#            results/; review the diff before committing)
#   PYTHON   interpreter (default: python)
#
# Outputs: $FACE1KB_OUTPUT_ROOT/figures/*.png and
# $FACE1KB_OUTPUT_ROOT/report_summary/derived_stats.json. Needs matplotlib (the
# `eval` extra).

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
FROM="${FROM:-results}"

run() {
	echo "==> $*"
	"$PY" "$@"
}

if [ "${COLLECT:-0}" = 1 ]; then
	run experiments/figures/render_figures.py
	run experiments/figures/collect_results.py --keep-shipped
	echo "done"
	exit 0
fi

case "$FROM" in
results) run experiments/figures/render_figures.py --from-results ;;
outputs) run experiments/figures/render_figures.py ;;
*)
	echo "FROM must be 'results' or 'outputs', got '$FROM'" >&2
	exit 2
	;;
esac
echo "done"
