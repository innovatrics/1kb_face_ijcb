#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Resolution-information analysis (paper Section "Ablations and Preprocessing",
# resolution-information trade-off): embedding decomposition vs. the clean 112 px
# crop, spectral energy retention and clean EER at 64-224 px, for both datasets and
# every public matcher with embeddings; then the decomposition table (AI-Solutions-KK,
# four anchors) and the summary figure.
#
# Needs: the clean embeddings embeddings/<model>/aligned_<res>.npy of both datasets
#   python experiments/embed/compute_embeddings.py --models all --kind aligned
# and the aligned 224 px crops (spectral sample of 400 crops per dataset).
#
# Cost: CPU only. analyze.py ~30-60 min (140 EERs over 3 M non-mated trials each;
# mostly embedding loads), ~6 GB RAM; render.py seconds. No GPU.
#
# Environment: PYTHON (default python), MODELS (default roster), SKIP ("analyze"
# and/or "render").

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY="${PYTHON:-python}"
SKIP="${SKIP:-}"
E=experiments/resolution

skip() { [[ " $SKIP " == *" $1 "* ]]; }
run() {
	echo "==> $*"
	"$PY" "$@"
}

if ! skip analyze; then
	run "$E/analyze.py" --models "${MODELS:-roster}"
fi
if ! skip render; then
	run "$E/render.py"
fi
