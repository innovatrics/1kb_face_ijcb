#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Create a Python virtual environment for face1kb and fetch the released weights.
#
# Usage:
#   scripts/setup_env.sh            # latest compatible packages (pyproject extras)
#   scripts/setup_env.sh --paper    # exact package versions of the paper environment
#   scripts/setup_env.sh --core     # only the codecs (encode / decode)
#
# Environment variables:
#   PYTHON  interpreter used to create the venv (default: python3, or python3.11
#           with --paper: the paper environment used Python 3.11, several of its
#           pins need Python >= 3.11, and --paper checks for 3.11)
#   VENV    venv location (default: .venv in the repository root)
#
# Requirements: Linux, Python >= 3.10 (3.11 for --paper), an NVIDIA GPU with a driver
# that supports CUDA 12.8 for the GPU code paths, git-lfs for the weights. A fresh
# venv gets torch 2.10.0, the CUDA 12.8 build, in every mode (pyproject bounds torch
# below 2.11, whose PyPI wheels are CUDA 13 builds that need a newer driver); the
# paper bitstreams were verified with this torch version.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

VENV="${VENV:-$REPO/.venv}"
MODE="full"
case "${1:-}" in
"") ;;
--paper) MODE="paper" ;;
--core) MODE="core" ;;
-h | --help)
	sed -n '4,21p' "$0"
	exit 0
	;;
*)
	echo "unknown option: $1 (use --paper, --core or no option)" >&2
	exit 2
	;;
esac

if [ "$MODE" = paper ]; then
	PYTHON="${PYTHON:-python3.11}"
else
	PYTHON="${PYTHON:-python3}"
fi

# requirements/paper.txt pins packages that need Python 3.11 (numpy 2.4.6, scipy
# 1.17.0, pandas 3.0.0, ...); fail before installing anything.
check_python() {
	local py="$1"
	if [ "$MODE" = paper ]; then
		if ! "$py" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))'; then
			echo "error: --paper needs Python 3.11, $py is $("$py" -V 2>&1);" \
				"set PYTHON=/path/to/python3.11 (and a fresh VENV)" >&2
			exit 1
		fi
	elif ! "$py" -c 'import sys; sys.exit(sys.version_info[:2] < (3, 10))'; then
		echo "error: face1kb needs Python >= 3.10, $py is $("$py" -V 2>&1)" >&2
		exit 1
	fi
}

if [ ! -d "$VENV" ]; then
	if ! command -v "$PYTHON" >/dev/null 2>&1; then
		echo "error: interpreter '$PYTHON' not found; set PYTHON=/path/to/python" >&2
		exit 1
	fi
	check_python "$PYTHON"
	echo "==> Creating virtual environment in $VENV"
	"$PYTHON" -m venv "$VENV"
else
	echo "==> Using the existing virtual environment in $VENV"
fi
check_python "$VENV/bin/python"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip

case "$MODE" in
paper)
	echo "==> Installing the exact paper environment (requirements/paper.txt)"
	# --no-deps: the pins are the complete closure of the paper environment, which
	# deliberately deviates from some declared requirements (e.g. numpy 2.4.6
	# although compressai 1.2.8 declares numpy<2; see the notes in the file).
	pip install --no-deps -r requirements/paper.txt
	pip install --no-deps -e .
	;;
core)
	echo "==> Installing face1kb (codecs only)"
	pip install -e .
	;;
full)
	echo "==> Installing face1kb with all extras"
	pip install -e ".[codecs,jpegai,eval,preprocess,mst,train,adversarial,dev]"
	# insightface requires the CPU-only `onnxruntime`, which shadows onnxruntime-gpu
	# (both install the same `onnxruntime` module). Keep only the GPU build.
	if pip show onnxruntime >/dev/null 2>&1; then
		pip uninstall -y onnxruntime
		pip install --force-reinstall --no-deps "onnxruntime-gpu==1.23.2"
	fi
	# MediaPipe (selfie segmenter of the preprocessing study) without its
	# dependencies: they would add opencv-contrib-python next to opencv-python.
	pip install --no-deps "mediapipe==0.10.35" "sounddevice==0.5.5"
	;;
esac

echo "==> Fetching the released weights (git lfs)"
if command -v git >/dev/null && git lfs version >/dev/null 2>&1; then
	git lfs install --local >/dev/null
	git lfs pull --include "weights/*.safetensors"
else
	echo "    WARNING: git-lfs not found; install it and run: git lfs pull" >&2
fi

echo "==> Checking the installation"
python - <<'EOF'
import face1kb
from face1kb import config

print(f"face1kb {face1kb.__version__}")
for v in config.CODEC_VARIANTS:
    p = config.weights_path(v)
    ok = p.is_file() and p.stat().st_size > 1024
    print(f"  weights {v:9s}: {'ok' if ok else 'MISSING (git lfs pull)'}  {p}")
try:
    import torch

    print(f"  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("  torch not importable")
EOF

cat <<EOF

Done. Next steps:
  source $VENV/bin/activate
  python -m face1kb.codec encode face.png face.f1k --variant accurate --budget 1024
  python -m face1kb.codec decode face.f1k face_decoded.png
  pytest -m "not data"                  # tests that need no datasets
See docs/codec.md for the codec API and docs/training.md for training.
Third-party FR models and the JPEG-AI reference software are set up separately
(scripts/fetch_models.py, scripts/setup_jpegai.sh).
EOF
