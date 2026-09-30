# SPDX-License-Identifier: MIT
"""Learned baselines from CompressAI: bmshj2018-factorized and mbt2018-mean.

Both use the pretrained MSE models of CompressAI 1.2.8 (``compressai.zoo``;
the weights are downloaded by CompressAI into the torch hub cache).

* Input: RGB in ``[0, 1]`` (float32), edge-padded right/bottom to a multiple of
  64 px.
* Rate knob: the model quality index 1..8 (:data:`QUALITIES`), binary-searched
  on the **entropy-coded size**, i.e. the summed lengths of the latent strings.
* Container (``.ptci``): a pickle (protocol 4) of ``{"strings", "shape", "size",
  "codec", "quality"}``. It adds 125 B (bmshj2018) or 135 B (mbt2018) to the
  entropy-coded size in almost all stored files (pickle writes a byte string
  shorter than 256 B with a 1-byte length, so the overhead is 3 B less when the
  latent string ``y`` is shorter than 256 B, and would be 3 B more if an mbt2018
  hyperprior string ``z`` reached 256 B), so the file on disk can exceed the budget
  although the search fitted; the paper's fit tables count the file size. Reading
  uses a restricted unpickler that accepts only plain containers, bytes, numbers
  and ``torch.Size``.
* Decode: ``decompress``, clamp to ``[0, 1]``, crop, ``(x * 255).round()``.

Bitstreams depend on the device (the entropy-model tables are rebuilt with
``update(force=True)``); the paper encoded and decoded on CUDA. A stream encoded on
CUDA does not decode on the CPU (the entropy decoder desynchronises and the
reconstruction is noise); :func:`decode` logs a warning for a non-CUDA device.

Numerics: every call runs with fixed cuDNN / TF32 settings (:func:`paper_numerics`):
cuDNN's default algorithm selection, no autotuning, no TF32. The paper encoded the
main grid this way; the 768 B / 960 B budget-sweep cells were encoded in a process
that had run the JPEG-AI reference software first, which switches cuDNN to its
deterministic algorithms globally, so :func:`encode_to_budget` takes
``deterministic=True`` to reproduce those cells.

Some of cuDNN's default kernels are non-deterministic, so two CUDA decodes of a file
can differ by 1 grey level in a few pixels. For mbt2018-mean this also matters for
the entropy decoder: the latent is decoded with scale indexes that the decoder
recomputes from the hyperprior, and when a scale lies on a boundary of the scale
table a different kernel (a non-deterministic one, or the deterministic kernels of
the budget-sweep encoder) can select another index; the rANS decoder then
desynchronises and the reconstruction turns into noise. :func:`decode` therefore
verifies the decoded latent: re-encoding it with the decoder's indexes must
reproduce the stored string exactly, which holds only when the indexes are the
encoder's. Otherwise the hyperprior is re-run with the default kernels and then with
the deterministic kernels. ``verify=False`` decodes once, as the paper's code did.
The encoder is affected in the same way: for a file with a boundary scale a re-encode
can write another valid stream of the same size.
"""

from __future__ import annotations

import io
import logging
import pickle
from collections.abc import Sequence
from contextlib import contextmanager

import numpy as np

from .search import binary_search_fit

logger = logging.getLogger(__name__)

__all__ = [
    "CODECS",
    "EXTENSION",
    "MODELS",
    "QUALITIES",
    "compress_quality",
    "decode",
    "encode_to_budget",
    "entropy_size",
    "get_net",
    "pack",
    "paper_numerics",
    "to_tensor",
    "unpack",
]

#: Codec names -> CompressAI zoo model names.
MODELS: dict[str, str] = {
    "neural_bmshj2018": "bmshj2018-factorized",
    "neural_mbt2018_mean": "mbt2018-mean",
}
#: CompressAI codecs, in the order of the paper tables.
CODECS: tuple[str, ...] = tuple(MODELS)
#: File extension of the container.
EXTENSION = ".ptci"
#: Quality indices searched (lowest rate first).
QUALITIES: tuple[int, ...] = tuple(range(1, 9))
#: Spatial padding multiple of the models.
PAD_MULTIPLE = 64
#: Pickle protocol of the stored containers.
PICKLE_PROTOCOL = 4
#: Latent symbols beyond this magnitude only come from a desynchronised decode.
_MAX_SYMBOL = 1 << 15

_NETS: dict[tuple, object] = {}


def _check_codec(codec: str) -> str:
    if codec not in MODELS:
        raise ValueError(f"unknown CompressAI codec {codec!r}; choose from {CODECS}")
    return codec


def get_net(codec: str, quality: int, device="cuda"):
    """Pretrained model of ``codec`` at ``quality`` on ``device`` (cached).

    The entropy-model tables are rebuilt with ``update(force=True)``.
    """
    import torch  # noqa: PLC0415
    from compressai.zoo import bmshj2018_factorized, mbt2018_mean  # noqa: PLC0415

    _check_codec(codec)
    dev = torch.device(device)
    key = (codec, int(quality), str(dev))
    net = _NETS.get(key)
    if net is None:
        builder = {
            "neural_bmshj2018": bmshj2018_factorized,
            "neural_mbt2018_mean": mbt2018_mean,
        }[codec]
        net = builder(quality=int(quality), pretrained=True).eval().to(dev)
        net.update(force=True)
        _NETS[key] = net
    return net


def to_tensor(img, device="cuda"):
    """``(1, 3, H', W')`` float tensor of ``img`` edge-padded to a multiple of 64.

    Returns ``(x, (h, w))`` with the original size.
    """
    import torch  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    if isinstance(img, Image.Image):
        img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 RGB image, got shape {arr.shape}")
    h, w, _ = arr.shape
    ph, pw = (-h) % PAD_MULTIPLE, (-w) % PAD_MULTIPLE
    arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode="edge")
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    return x, (h, w)


def entropy_size(strings) -> int:
    """Return the summed length of the latent strings (the size searched on)."""
    return sum(len(s) for ss in strings for s in ss)


def pack(strings, shape, size: tuple[int, int], codec: str, quality: int) -> bytes:
    """Serialise a compressed image to the ``.ptci`` container."""
    return pickle.dumps(
        {
            "strings": strings,
            "shape": shape,
            "size": size,
            "codec": codec,
            "quality": int(quality),
        },
        protocol=PICKLE_PROTOCOL,
    )


class _RestrictedUnpickler(pickle.Unpickler):
    """Unpickler that resolves no global except ``torch.Size``."""

    def find_class(self, module, name):
        if (module, name) == ("torch", "Size"):
            import torch  # noqa: PLC0415

            return torch.Size
        raise pickle.UnpicklingError(
            f"global {module}.{name} is not allowed in a .ptci container"
        )


def unpack(data: bytes) -> dict:
    """Parse a ``.ptci`` container safely.

    Only dicts, lists, tuples, bytes, strings, numbers and ``torch.Size`` are
    accepted; any other global raises :class:`pickle.UnpicklingError`.
    """
    obj = _RestrictedUnpickler(io.BytesIO(data)).load()
    if not isinstance(obj, dict):
        raise ValueError("a .ptci container must hold a dict")
    missing = {"strings", "shape", "size", "codec", "quality"} - set(obj)
    if missing:
        raise ValueError(f".ptci container misses {sorted(missing)}")
    strings = obj["strings"]
    if not isinstance(strings, list) or not all(
        isinstance(ss, list) and all(isinstance(s, bytes) for s in ss) for ss in strings
    ):
        raise ValueError(".ptci 'strings' must be a list of lists of bytes")
    _check_codec(obj["codec"])
    return obj


def compress_quality(
    img, codec: str, quality: int, device="cuda", *, deterministic: bool = False
):
    """Compress at one quality; return ``(entropy_size, container_bytes)``."""
    import torch  # noqa: PLC0415

    x, size = to_tensor(img, device)
    return _compress(x, size, codec, quality, device, torch, deterministic)


def _compress(x, size, codec, quality, device, torch, deterministic=False):
    net = get_net(codec, quality, device)
    with torch.no_grad(), paper_numerics(deterministic):
        out = net.compress(x)
    return entropy_size(out["strings"]), pack(
        out["strings"], out["shape"], size, codec, quality
    )


def encode_to_budget(
    img,
    codec: str,
    budget: int,
    *,
    device="cuda",
    qualities: Sequence[int] = QUALITIES,
    deterministic: bool = False,
) -> tuple[bytes, dict]:
    """Encode with the highest quality whose entropy-coded size fits ``budget``.

    Returns the ``.ptci`` container and ``{"fitted", "setting", "size"}``, where
    ``size`` is the entropy-coded size compared against the budget (the container
    is larger by the pickle overhead). When nothing fits, quality
    ``qualities[0]`` is returned with ``fitted=False``.

    ``deterministic`` selects cuDNN's deterministic algorithms (see
    :func:`paper_numerics`): False for the main grid of the paper, True for its
    768 B / 960 B budget-sweep cells.
    """
    import torch  # noqa: PLC0415

    _check_codec(codec)
    x, size = to_tensor(img, device)
    r = binary_search_fit(
        lambda q: _compress(x, size, codec, q, device, torch, deterministic),
        list(qualities),
        budget,
    )
    return r.payload, {"fitted": r.fitted, "setting": r.setting, "size": r.size}


@contextmanager
def paper_numerics(deterministic: bool = False):
    """Fix the cuDNN / TF32 settings for a CompressAI call (restored on exit).

    No autotuning (``benchmark=False``), full-fp32 convolutions and matmuls (no
    TF32), and cuDNN's default algorithm selection, or its deterministic algorithms
    with ``deterministic=True``.
    """
    import torch  # noqa: PLC0415

    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    prev = (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32, matmul.allow_tf32)
    cudnn.deterministic, cudnn.benchmark = bool(deterministic), False
    cudnn.allow_tf32, matmul.allow_tf32 = False, False
    try:
        yield
    finally:
        (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32, matmul.allow_tf32) = (
            prev
        )


def _decompress_verified(net, strings, shape, attempts: int):
    """``MeanScaleHyperprior.decompress`` with a round-trip check of the latent.

    The hyperprior runs up to ``attempts`` times with cuDNN's default kernels, then
    once with its deterministic kernels, until the decoded latent re-encodes to the
    stored string. Returns ``(x_hat, attempts_used, verified)``.
    """
    gc = net.gaussian_conditional
    z_hat = net.entropy_bottleneck.decompress(strings[1], shape)
    plan = [False] * max(1, int(attempts)) + [True]
    n, ok, y_hat = 0, False, None
    while not ok and n < len(plan):
        with paper_numerics(plan[n]):
            scales_hat, means_hat = net.h_s(z_hat).chunk(2, 1)
        n += 1
        indexes = gc.build_indexes(scales_hat)
        y = gc.decompress(strings[0], indexes, means=means_hat)
        # A desynchronised decode yields huge symbols, which the re-encoder cannot
        # handle (its escape coding loops without end near the int32 limits).
        if (y - means_hat).abs().max().item() > _MAX_SYMBOL:
            y_hat = y if y_hat is None else y_hat
            continue
        y_hat = y
        ok = gc.compress(y, indexes, means=means_hat) == list(strings[0])
    return net.g_s(y_hat).clamp_(0, 1), n, ok


def decode(
    data: bytes, *, device="cuda", verify: bool = True, attempts: int = 4
) -> np.ndarray:
    """Decode a ``.ptci`` container to an ``H x W x 3`` ``uint8`` RGB array.

    Parameters
    ----------
    data
        The ``.ptci`` container.
    device
        Torch device. Decode on the device type the stream was encoded on: the
        paper bitstreams (encoded on CUDA) decode to noise on the CPU, because the
        entropy-model tables differ; a warning is logged for a non-CUDA device.
    verify
        mbt2018-mean only: check that the decoded latent re-encodes to the stored
        string and re-run the hyperprior until it does (see the module notes).
        A warning is logged if no attempt verifies; the last decode is returned.
    attempts
        Hyperprior runs with the default kernels before the deterministic kernels
        are tried (``verify`` only).
    """
    import torch  # noqa: PLC0415

    d = unpack(data)
    if torch.device(device).type != "cuda":
        logger.warning(
            "decoding a CompressAI stream on %s: streams encoded on CUDA (all paper "
            "streams) do not decode correctly on another device type",
            device,
        )
    net = get_net(d["codec"], int(d["quality"]), device)
    with torch.no_grad(), paper_numerics():
        if verify and d["codec"] == "neural_mbt2018_mean":
            x_hat, n, ok = _decompress_verified(net, d["strings"], d["shape"], attempts)
            if not ok:
                logger.warning(
                    "mbt2018-mean latent did not verify in %d decode attempts; the "
                    "reconstruction may be corrupt",
                    n,
                )
            elif n > 1:
                logger.info("mbt2018-mean latent verified at decode attempt %d", n)
        else:
            x_hat = net.decompress(d["strings"], d["shape"])["x_hat"]
    x = x_hat.clamp(0, 1)[0].cpu()
    h, w = d["size"]
    return (x[:, :h, :w].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
