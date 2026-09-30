#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Rate--identity codec comparison of the paper (Tables 7, 8, 40 and 41) and the
# JPEG-AI operation-point figure (Figure 30), starting from the aligned crops under
# $FACE1KB_DATA_ROOT.
#
#   1. 64-crop comparison at 112 and 224 px, 1024 / 512 B,  1 GPU + CPU, ~3 h at
#      both datasets, every codec encoded on the fly          112 px and ~4 h at
#      -> codec_comparison/res<res>/comparison.json            224 px (JPEG-AI ~20%)
#   2. the four comparison tables                            CPU, seconds
#      -> tables/codec_comparison_{112,224}.tex, codec_results{,_112}.tex
#      (fidelity columns from quality/quality_summary.csv, experiments/quality)
#   3. the JPEG-AI trade-off figure from jpegai_decoders.csv  CPU, seconds
#      -> figures/jpegai_decoder_tradeoff.{png,pdf}
#
# The id-cos columns of the paper were measured with a proprietary face matcher
# that is not distributed. Step 1 uses the public matcher ID_MODEL instead, so its
# id-cos differ from the paper; FROM_RESULTS=1 renders steps 2-3 from the shipped
# results, the only source of the published id-cos values.
#
# Environment:
#   PYTHON        interpreter (default: python)
#   DEVICE        GPU device (default: cuda)
#   ID_MODEL      identity model of step 1 (default: cvlface_ir101; any face1kb.fr
#                 model, including plugins named in FACE1KB_FR_PLUGINS)
#   JPEGAI        set to 0 to leave JPEG-AI out of step 1
#   SKIP          space-separated step numbers to skip, e.g. SKIP="1"
#   PAPER_HEADER  set to 1 to write the first comment line of the published tables
#                 (it names the script that originally produced them)
#   FROM_RESULTS  set to 1 to render steps 2-3 from <repo>/results

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
EXTRA=()
if [ "${JPEGAI:-1}" = 0 ]; then
	EXTRA=(--no-jpegai)
fi

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}

if ! skip 1; then
	for res in 112 224; do
		run experiments/codec_comparison/evaluate.py --res "$res" --device "$DEVICE" \
			--id-model "${ID_MODEL:-cvlface_ir101}" ${EXTRA[@]+"${EXTRA[@]}"}
	done
fi
if ! skip 2; then
	run experiments/codec_comparison/tables.py ${SRC[@]+"${SRC[@]}"}
fi
if ! skip 3; then
	if [ "${FROM_RESULTS:-0}" = 1 ]; then
		run experiments/codec_comparison/jpegai_tradeoff.py --from-results
	else
		run experiments/codec_comparison/jpegai_tradeoff.py
	fi
fi
echo "done"
