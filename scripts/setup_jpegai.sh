#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Set up the JPEG-AI reference software (Rec. ITU-T T.840.1 | ISO/IEC 6048-1) for the
# JPEG-AI baseline, exactly as the benchmark ran it:
#
#   1. clone the official repository and check out commit e9648f9 (detached),
#   2. apply third_party/jpeg-ai.patch,
#   3. restore the upstream bitrate-matcher caches (cfg/betas) and make the folders
#      the reference software writes at run time writable,
#   4. install the Python dependencies (pip install -e ".[jpegai]" in this repo),
#   5. fetch the models with git-lfs (models/ only, about 1 GB),
#   6. build the C++ entropy-coder libraries for the active Python,
#   7. if a CUDA GPU is visible, encode and decode one synthetic image
#      (python -m face1kb.baselines.verify --codecs jpeg_ai).
#
# The script is idempotent: re-running it skips the steps that are already done.
#
# Usage:
#   scripts/setup_jpegai.sh [--dir DIR] [--no-install] [--no-models] [--no-build]
#                           [--no-check]
#
# Environment variables:
#   FACE1KB_JPEGAI_DIR  checkout location (default: third_party/jpeg-ai-reference-software
#                       in this repository); --dir overrides it. face1kb reads the same
#                       variable, so export it if you use a non-default location.
#   PYTHON              interpreter of the face1kb environment (default: python)
#
# Requirements: git, git-lfs, make, g++ with OpenMP, the Python development headers.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git"
COMMIT="e9648f9a4bb98d8fb7333e97d4df4f0979520a7c"
PATCH="$REPO/third_party/jpeg-ai.patch"
PYBIND11_VERSION="3.0.4"

DEST="${FACE1KB_JPEGAI_DIR:-$REPO/third_party/jpeg-ai-reference-software}"
PYTHON="${PYTHON:-python}"
DO_INSTALL=1
DO_MODELS=1
DO_BUILD=1
DO_CHECK=1

while [ $# -gt 0 ]; do
	case "$1" in
	--dir)
		DEST="$2"
		shift 2
		;;
	--no-install)
		DO_INSTALL=0
		shift
		;;
	--no-models)
		DO_MODELS=0
		shift
		;;
	--no-build)
		DO_BUILD=0
		shift
		;;
	--no-check)
		DO_CHECK=0
		shift
		;;
	-h | --help)
		sed -n '4,28p' "$0"
		exit 0
		;;
	*)
		echo "unknown option: $1 (see --help)" >&2
		exit 2
		;;
	esac
done

need() {
	command -v "$1" >/dev/null 2>&1 || {
		echo "ERROR: '$1' is required but not installed" >&2
		exit 1
	}
}
need git
need make
need g++
git lfs version >/dev/null 2>&1 || {
	echo "ERROR: git-lfs is required (https://git-lfs.com)" >&2
	exit 1
}

# Smudging is skipped everywhere: the models are fetched explicitly in step 5, and the
# calibration images of the repository are not needed.
export GIT_LFS_SKIP_SMUDGE=1

echo "==> [1/7] JPEG-AI reference software at $DEST"
FRESH=0
if [ ! -d "$DEST/.git" ]; then
	mkdir -p "$(dirname "$DEST")"
	git clone --no-checkout "$URL" "$DEST"
	FRESH=1
fi
if ! git -C "$DEST" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
	git -C "$DEST" fetch origin
fi
HEAD="$(git -C "$DEST" rev-parse HEAD 2>/dev/null || true)"
if [ "$FRESH" = 1 ] || [ "$HEAD" != "$COMMIT" ]; then
	if [ "$FRESH" = 0 ] && ! git -C "$DEST" diff --quiet HEAD -- 2>/dev/null; then
		echo "ERROR: $DEST has local changes and is not at $COMMIT; move it away or" >&2
		echo "       reset it (git -C \"$DEST\" checkout -f $COMMIT) and re-run" >&2
		exit 1
	fi
	git -C "$DEST" -c advice.detachedHead=false checkout --detach "$COMMIT"
fi

echo "==> [2/7] Applying third_party/jpeg-ai.patch"
if git -C "$DEST" apply --reverse --check "$PATCH" 2>/dev/null; then
	echo "    already applied"
else
	git -C "$DEST" apply "$PATCH"
fi

echo "==> [3/7] Restoring the upstream bitrate-matcher caches (cfg/betas)"
# With --set_target_bpp (the benchmark's rate knob) the reference encoder deletes and
# re-writes a per-image log cfg/betas/<op>/betas_*.txt inside the checkout; it never
# reads that log back in this mode, so the bitstreams do not depend on it. face1kb
# redirects the log to a private temporary folder, which keeps the checkout unchanged
# and lets concurrent processes share it; the upstream files are restored here in case
# another tool rewrote them. The entropy coder writes lib_wrappers/mans/cache.pt on its
# first use, so the checkout must be writable (the check in step 7 creates the file).
git -C "$DEST" checkout -- cfg/betas
chmod -R u+w "$DEST/cfg/betas" "$DEST/src/codec/entropy_coding/lib_wrappers"

if [ "$DO_INSTALL" = 1 ]; then
	echo "==> [4/7] Installing the Python dependencies ($PYTHON)"
	"$PYTHON" -m pip install -e "${REPO}[jpegai]"
	if ! "$PYTHON" -c "import pybind11" 2>/dev/null; then
		"$PYTHON" -m pip install "pybind11==$PYBIND11_VERSION"
	fi
else
	echo "==> [4/7] Skipping the Python dependencies (--no-install)"
fi

if [ "$DO_MODELS" = 1 ]; then
	echo "==> [5/7] Fetching the models (git lfs, models/ only)"
	git -C "$DEST" lfs install --local --skip-smudge >/dev/null
	git -C "$DEST" lfs pull --include "models/**"
else
	echo "==> [5/7] Skipping the models (--no-models)"
fi

if [ "$DO_BUILD" = 1 ]; then
	echo "==> [6/7] Building the entropy-coder libraries"
	SUFFIX="$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
	INCLUDES="$("$PYTHON" -m pybind11 --includes)"
	CPP="$DEST/src/codec/entropy_coding/cpp_exts"
	LIBS="$DEST/src/codec/entropy_coding/lib_wrappers"
	# The upstream Makefiles take the suffix from python3-config, which may belong to
	# another interpreter; pass the values of the active Python instead.
	if [ ! -f "$LIBS/mans/ans$SUFFIX" ]; then
		make -C "$CPP/mans" PYBIND_INCLUDES="$INCLUDES" PYTHON_SUFFIX="$SUFFIX"
	fi
	if [ ! -f "$LIBS/direct/ec_direct$SUFFIX" ]; then
		make -C "$CPP/direct" PYBIND_INCLUDES="$INCLUDES" PYTHON_SUFFIX="$SUFFIX"
	fi
else
	echo "==> [6/7] Skipping the build (--no-build)"
fi

if [ "$DO_CHECK" = 1 ] && command -v nvidia-smi >/dev/null 2>&1 &&
	"$PYTHON" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
	echo "==> [7/7] Checking the installation (one encode + decode on the GPU)"
	(cd "$REPO" && PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" FACE1KB_JPEGAI_DIR="$DEST" \
		"$PYTHON" -m face1kb.baselines.verify --codecs jpeg_ai --strict)
else
	echo "==> [7/7] Skipping the check (no CUDA GPU or --no-check); run later:"
	echo "    FACE1KB_JPEGAI_DIR=$DEST $PYTHON -m face1kb.baselines.verify --codecs jpeg_ai"
fi

echo "==> Done. JPEG-AI reference software: $DEST"
if [ "$DEST" != "$REPO/third_party/jpeg-ai-reference-software" ]; then
	echo "    export FACE1KB_JPEGAI_DIR=$DEST"
fi
