# Using the JPEG-AI reference software

JPEG-AI is the learning-based image coding standard **Rec. ITU-T T.840.1 |
ISO/IEC 6048-1**. The paper evaluates it through the official [reference
software](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software), which
is licensed separately (BSD) and therefore not bundled with this repository.
This document describes how to obtain it and configure it exactly as used
for the paper's experiments.

## Version used in the paper

- Repository: `https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software`
- Version: commit **`e9648f9a4bb98d8fb7333e97d4df4f0979520a7c`**
  (`e9648f9`, "Added calibration dataset", 11 March 2026), 17 commits
  after the tag `DIS` on `main`. `git describe` prints it as
  `DIS-17-ge9648f9`; there is no tag or branch named `DIS-17`, so check
  out the commit hash.

Newer versions will most likely work, but bitstream sizes and
reconstructions may differ slightly from the published numbers.

## Installation

Run these steps in the environment of this pipeline (Python >= 3.10, after
`pip install -r requirements.txt` of this repository, or in the `.venv` of
`setup_env.sh`): the wrapper imports the encoder and decoder in-process.
Besides Python you need `git-lfs`, `make`, `g++` with OpenMP and the
development headers of the active Python (e.g. `python3.11-dev` on
Debian/Ubuntu).

```bash
# 1. Clone and pin the paper's commit (setup_env.sh --with-jpeg-ai does
#    both for you). When git-lfs is set up (git lfs install), the clone
#    also downloads the pretrained models (models/, about 1 GB) and the
#    test images.
git clone https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git \
    third_party/jpeg-ai-reference-software
cd third_party/jpeg-ai-reference-software
git checkout e9648f9a4bb98d8fb7333e97d4df4f0979520a7c   # DIS + 17 commits

# 2. Fetch the models if the clone did not (cloned without git-lfs or
#    with GIT_LFS_SKIP_SMUDGE=1); only models/ is needed.
git lfs install
git lfs pull --include "models/**"

# 3. Apply the compatibility changes described under "Local changes"
#    below (path relative to the default checkout location).
git apply ../../docs/jpeg-ai-compat.patch

# 4. Install the packages the encoder and decoder import on top of this
#    repository's requirements (pynvml >= 12 no longer has pynvml.smi).
pip install addict commentjson einops GPUtil prettytable psutil ptflops \
    pytorch-msssim "pynvml<12" pybind11

# 5. Build the C++ entropy-coding extensions (ans, ec_direct) for the
#    active Python.
suffix=$(python -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
make -C src/codec/entropy_coding/cpp_exts/mans PYTHON_SUFFIX="$suffix"
make -C src/codec/entropy_coding/cpp_exts/direct PYTHON_SUFFIX="$suffix"

# 6. Point the pipeline at the checkout
export JPEGAI_REPO_DIR=$PWD
```

Do not install the reference software's own `requirements.txt`: it pins
the PyTorch 1.10 / NumPy 1.19 era (`torch==1.10.2`, `numpy==1.19.1`,
`scipy==1.5.2`, `opencv-python==4.5.5.62`, ...) and cannot be resolved on
Python >= 3.10. Its `dvc` is not needed either, since the models come
through git-lfs. Step 5 builds the same libraries as the upstream
`make build_test_libs` (`scripts/build_ec_lib.sh`), which activates a conda
environment and takes the file-name suffix from `python3-config`; passing
`PYTHON_SUFFIX` makes the build independent of both. With these steps
(Python 3.11, PyTorch 2.10, NumPy 2.4) the pipeline reproduces the paper's
224×224 JPEG-AI bitstreams byte for byte.

A CUDA-capable GPU is strongly recommended: encoding runs per image and is
considerably slower on CPU (Table 2 of the paper: about 1.6 s on the GPU vs
3.0 s on the CPU per 112×112 image, and 1.6 s vs 8.4 s per 224×224 image).
Install a PyTorch build that supports your CUDA driver.

### Local changes

The checkout used for the paper differs from commit `e9648f9` in the
following files (`git status`):

- Compatibility with current PyTorch, NumPy and ptflops, required with the
  setup above and applied in step 3 by
  [`jpeg-ai-compat.patch`](jpeg-ai-compat.patch) - without each of them,
  loading the models or decoding fails. None of them changes the coding
  algorithm.
  - `src/__init__.py`: PyTorch >= 2.6 defaults to `torch.load(...,
    weights_only=True)`, which cannot load the model checkpoints; wrap
    `torch.load` so that it defaults to `weights_only=False`.
  - `src/codec/entropy_coding/binarizers.py`: `bypass_coder.decode(...)`
    returns an array; index its first element (`[0]`) in
    `decode_unsigned_unary` and `decode_unsigned_expgolomb_k0`.
  - `src/reco/coders/decoder.py`: call `coder.setup_ptflops_custom_hooks()`
    only when `calc_ptflops` is requested (ptflops >= 0.6.8 no longer
    provides the `conv_flops_counter_hook` those hooks import).
- Build and setup scripts, not needed with the steps above:
  - `scripts/build_ec_lib.sh`, `scripts/build_test_libs.sh`: the conda
    environment is activated only when `conda` is on the `PATH`.
  - `Makefile`, `Dockerfile`, `scripts/setup_system.sh`,
    `scripts/sFTP_mirror/README.md`: `sudo` dropped from the commands (and
    from the packages the `Dockerfile` installs), plus whitespace fixes.
- Written by the reference software at run time:
  `cfg/betas/hop/betas_tools_off.txt`. Every encode with the configuration
  below replaces the file with the bitrate matcher's entry for the image
  just encoded, so a used checkout always shows it as modified
  (`cfg/betas/bop/betas_tools_off.txt` likewise after encodes with the base
  or simple profile).

Git ignores the other generated files: the compiled extensions of step 5
and `src/codec/entropy_coding/lib_wrappers/mans/cache.pt`, which the
entropy coder writes on first use.

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
  does not fit, the image is encoded at `--set_target_bpp 1` (0.01 bpp)
  and retained (paper Sec. 2.5).

Decoding uses the same reference software without extra parameters:
bitstream in, PNG out.

## Troubleshooting

- `RuntimeError: JPEG-AI reference software not found` - set
  `JPEGAI_REPO_DIR` to the checkout directory.
- `ModuleNotFoundError` from `src.reco.coders` - a package of step 4 is
  missing in the active environment (`No module named 'pynvml.smi'`:
  pynvml >= 12 is installed; install `"pynvml<12"`).
- `ImportError: cannot import name 'ans' from partially initialized module
  ...lib_wrappers.mans` (or `ec_direct` in `lib_wrappers.direct`) - the C++
  extensions are not built for the active Python; repeat step 5 and check
  that `src/codec/entropy_coding/lib_wrappers/mans/ans<suffix>` exists
  with the suffix printed by the `sysconfig` command of step 5.
- `RuntimeError: JPEG-AI compression produced no output` - the wrapper
  hides the reference software's output and exceptions. Run one encode by
  hand inside the checkout to see the error:

  ```bash
  python -m src.reco.coders.encoder <input.png> out.bin \
      --set_target_bpp 14 --cfg cfg/tools_off.json cfg/profiles/high.json
  ```

  The reference software runs from inside its checkout, so path overrides
  such as `FACE1KB_DATA_ROOT` must be absolute.
- Very slow encoding (~seconds per image) is expected; the reference
  implementation is not optimised. Budget several hours for the full
  dataset per resolution, or restrict the run with
  `--codecs` on the other stages first.
