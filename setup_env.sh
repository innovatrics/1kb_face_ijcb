#!/usr/bin/env bash
# Prepare the Python environment for the replication pipeline.
#
# Creates a virtual environment in .venv, installs the Python dependencies
# and checks the external tools. Optionally clones the JPEG-AI reference
# software (--with-jpeg-ai); its models and dependencies still need the
# manual steps described in docs/JPEG_AI.md.
#
# Usage:
#   ./setup_env.sh [--with-jpeg-ai]

set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
JPEGAI_DIR="third_party/jpeg-ai-reference-software"
JPEGAI_URL="https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git"

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
    echo "==> Cloning the JPEG-AI reference software"
    if [ ! -d "$JPEGAI_DIR" ]; then
        git clone "$JPEGAI_URL" "$JPEGAI_DIR"
    fi
    echo "    Clone finished. Complete the setup manually:"
    echo "    see docs/JPEG_AI.md (install its requirements and download"
    echo "    the models), then export JPEGAI_REPO_DIR=\$PWD/$JPEGAI_DIR"
else
    echo "==> JPEG-AI reference software not requested (--with-jpeg-ai);"
    echo "    the jpeg_ai codec will be skipped unless JPEGAI_REPO_DIR is"
    echo "    set. See docs/JPEG_AI.md."
fi

echo "==> Done. Activate the environment with: source .venv/bin/activate"
