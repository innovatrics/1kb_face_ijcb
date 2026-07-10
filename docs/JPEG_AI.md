# Using the JPEG-AI reference software

JPEG-AI is the learning-based image coding standard **Rec. ITU-T T.840.1 |
ISO/IEC 6048-1**. The paper evaluates it through the official [reference
software](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software), which
is licensed separately (BSD) and therefore not bundled with this repository.
This document describes how to obtain it and configure it exactly as used
for the paper's experiments.

## Version used in the paper

- Repository: `https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software`
- Version: **DIS-17** (commit `e9648f9`)

Newer versions will most likely work, but bitstream sizes and
reconstructions may differ slightly from the published numbers.

## Installation

```bash
# 1. Clone (setup_env.sh --with-jpeg-ai does this for you)
git clone https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git \
    third_party/jpeg-ai-reference-software
cd third_party/jpeg-ai-reference-software
git checkout DIS-17

# 2. Install its Python dependencies into the SAME environment that runs
#    this pipeline (the wrapper imports the encoder/decoder in-process).
#    The pins in its requirements.txt are old; installing on top of our
#    environment with newer versions is what the paper's setup used.
pip install -r requirements.txt

# 3. Download the pretrained models (needs git-lfs and dvc; the script
#    pulls models/*/*.dvc from the project's remote)
./scripts/download_models.sh

# 4. Point the pipeline at the checkout
export JPEGAI_REPO_DIR=$PWD
```

A CUDA-capable GPU is strongly recommended: encoding runs per image and is
orders of magnitude slower on CPU (see Table 1 of the paper).

## How the pipeline invokes it

The wrapper in [`face1kb/compression/jpeg_ai.py`](../face1kb/compression/jpeg_ai.py)
keeps one `RecoEncoderProcess` and one `RecoDecoderProcess` alive so the
model weights are loaded only once per run, then calls them with:

```text
<input.png> <output.bin> --set_target_bpp <bpp*100> \
    --cfg cfg/tools_off.json cfg/profiles/high.json
```

- `cfg/profiles/high.json` selects the **High Operating Point (HOP)**
  profile - the best-quality configuration, as stated in Sec. 2.5 of the
  paper.
- `cfg/tools_off.json` disables the optional tool extensions so the coding
  chain corresponds to the plain HOP configuration.
- `--set_target_bpp` takes the target bits-per-pixel multiplied by 100.
  Because the encoder does not accept a target file size, the pipeline
  bisects this parameter (range 0.10-0.30 bpp) for the highest bitrate
  whose bitstream fits the 1024 B budget
  (`find_max_bpp_within` in the wrapper). When even the lowest candidate
  does not fit, the image is encoded at the minimum bitrate and retained
  (paper Sec. 2.5).

Decoding uses the same reference software without extra parameters:
bitstream in, PNG out.

## Troubleshooting

- `RuntimeError: JPEG-AI reference software not found` - set
  `JPEGAI_REPO_DIR` to the checkout directory.
- Import errors from `src.reco.coders` - the reference software's
  dependencies are not installed in the active environment (step 2).
- Very slow encoding (~seconds per image) is expected; the reference
  implementation is not optimised. Budget several hours for the full
  dataset per resolution, or restrict the run with
  `--codecs` on the other stages first.
