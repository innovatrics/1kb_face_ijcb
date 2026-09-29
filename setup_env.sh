#!/usr/bin/env bash
# Prepare the Python environment for the replication pipeline.
#
# Creates a virtual environment in .venv, installs the Python dependencies
# and checks the external tools. Optionally clones the JPEG-AI reference
# software at the commit used in the paper (--with-jpeg-ai); its models and
# dependencies still need the manual steps described in docs/JPEG_AI.md.
#
# Usage:
#   ./setup_env.sh [--with-jpeg-ai]

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
JPEGAI_DIR="third_party/jpeg-ai-reference-software"
JPEGAI_URL="https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git"
# Commit used in the paper: 17 commits after the tag DIS (see docs/JPEG_AI.md).
JPEGAI_COMMIT="e9648f9a4bb98d8fb7333e97d4df4f0979520a7c"

echo "==> Creating virtual environment (.venv)"
if [ ! -d .venv ]; then
    "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Installing Python dependencies"
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Checking external tools"
if command -v cjxl > /dev/null && command -v djxl > /dev/null; then
    echo "    JPEG XL tools found: $(cjxl --version 2>&1 | head -1)"
else
    echo "    WARNING: cjxl/djxl not found - JPEG XL will be skipped."
    echo "             Install with: sudo apt install libjxl-tools"
fi

if [ "${1:-}" = "--with-jpeg-ai" ]; then
    echo "==> Cloning the JPEG-AI reference software (commit ${JPEGAI_COMMIT:0:7})"
    if [ ! -d "$JPEGAI_DIR" ]; then
        git clone "$JPEGAI_URL" "$JPEGAI_DIR"
        git -C "$JPEGAI_DIR" checkout --quiet "$JPEGAI_COMMIT"
    elif [ "$(git -C "$JPEGAI_DIR" rev-parse HEAD)" != "$JPEGAI_COMMIT" ]; then
        echo "    WARNING: $JPEGAI_DIR exists but is not at commit"
        echo "             $JPEGAI_COMMIT used in the paper; results may differ."
    fi
    echo "    Clone finished. Complete the setup manually:"
    echo "    see docs/JPEG_AI.md (models, local changes, Python packages and"
    echo "    C++ extensions), then export JPEGAI_REPO_DIR=\$PWD/$JPEGAI_DIR"
else
    echo "==> JPEG-AI reference software not requested (--with-jpeg-ai);"
    echo "    the jpeg_ai codec will be skipped unless JPEGAI_REPO_DIR is"
    echo "    set. See docs/JPEG_AI.md."
fi

echo "==> Done. Activate the environment with: source .venv/bin/activate"
