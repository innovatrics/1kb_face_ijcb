#!/usr/bin/env bash
# Run the complete replication pipeline end to end:
#
#   1. download the aligned dataset from HuggingFace,
#   2. report the verification-pair statistics,
#   3. compress every image with all available codecs (1 kB budget),
#   4. decode all bitstreams back to PNG,
#   5. compute face embeddings (open-source models + proprietary/mockup),
#   6. compute verification accuracy (EER, FRR at FAR operating points),
#   7. compute image quality (SSIM, LPIPS),
#   8. collect file sizes and measure codec speed,
#   9. run the statistical significance tests.
#
# Configuration via environment variables:
#   RESOLUTIONS  - resolutions to process (default: "112 224")
#   FACE1KB_*    - path overrides, see face1kb/config.py
#   JPEGAI_REPO_DIR - JPEG-AI reference software (see docs/JPEG_AI.md)
#
# Usage:
#   ./run_all.sh

set -euo pipefail
cd "$(dirname "$0")"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

PY=${PYTHON:-python}
RESOLUTIONS=${RESOLUTIONS:-"112 224"}
CODECS=(jpeg jpeg2000 jpeg_xl webp jpeg_fzt jpeg_ai)
DATA_ROOT=${FACE1KB_DATA_ROOT:-data}

banner() {
    echo
    echo "======================================================================"
    echo "==> $1"
    echo "======================================================================"
}

banner "1/9 Download aligned dataset"
$PY -m face1kb.data.download_dataset

banner "2/9 Verification-pair statistics"
$PY -m face1kb.data.define_pairs

for RES in $RESOLUTIONS; do
    banner "3/9 Compress all images (${RES}px, 1 kB budget)"
    $PY -m face1kb.compression.compress_dataset \
        --resolution "$RES" --skip-existing

    banner "4/9 Decode bitstreams to PNG (${RES}px)"
    $PY -m face1kb.compression.decompress_dataset \
        --resolution "$RES" --skip-existing

    banner "5/9 Embeddings (${RES}px)"
    $PY -m face1kb.embeddings.compute_embeddings \
        --resolution "$RES" --source original --skip-existing
    $PY -m face1kb.embeddings.compute_embeddings_proprietary \
        --resolution "$RES" --source original --skip-existing
    for CODEC in "${CODECS[@]}"; do
        if [ -d "$DATA_ROOT/decompressed_$RES/$CODEC" ]; then
            $PY -m face1kb.embeddings.compute_embeddings \
                --resolution "$RES" --source "$CODEC" --skip-existing
            $PY -m face1kb.embeddings.compute_embeddings_proprietary \
                --resolution "$RES" --source "$CODEC" --skip-existing
        fi
    done

    banner "6/9 Verification accuracy (${RES}px)"
    $PY -m face1kb.metrics.compute_accuracy --resolution "$RES"

    banner "7/9 Image quality: SSIM + LPIPS (${RES}px)"
    $PY -m face1kb.metrics.compute_image_quality --resolution "$RES"

    banner "8/9 File sizes and codec speed (${RES}px)"
    $PY -m face1kb.metrics.collect_file_sizes --resolution "$RES"
    $PY -m face1kb.metrics.measure_speed --resolution "$RES"

    banner "9/9 Statistical tests (${RES}px)"
    $PY -m face1kb.metrics.statistical_tests --resolution "$RES"
done

banner "Pipeline finished - see outputs/metrics/"
