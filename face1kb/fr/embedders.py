# SPDX-License-Identifier: MIT
"""Embedder base class, input handling and the generic ONNX / torch embedders.

All embedders share one contract: :meth:`Embedder.embed` takes a list of aligned
**112 x 112 RGB uint8** crops (or one ``(N, 112, 112, 3)`` uint8 array) and returns
an ``(N, D)`` float32 array of **raw** embeddings (not L2-normalised; normalise at
scoring time). Each family applies its own preprocessing internally.

The generic :class:`OnnxEmbedder` and :class:`TorchEmbedder` back the user hooks
:func:`face1kb.fr.register_onnx` and :func:`face1kb.fr.register_torch`; the built-in
LVFace models also run through :class:`OnnxEmbedder`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Side length of the aligned crops every embedder consumes.
INPUT_SIZE = 112
#: Embedding dimension of the built-in evaluators.
EMB_DIM = 512

#: A normalisation spec: ``"[-1,1]"`` -> ``(x / 255 - 0.5) / 0.5``, ``"[0,1]"`` ->
#: ``x / 255``, ``"none"`` -> ``x`` (0..255), ``(mean, std)`` in pixel units (scalars
#: or per-channel triples, in the model's channel order) -> ``(x - mean) / std``, or a
#: callable mapping the float32 batch (0..255, final layout) to the model input.
Normalize = str | tuple | Callable[[np.ndarray], np.ndarray] | None


def as_batch(images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Stack aligned crops into one ``(N, 112, 112, 3)`` uint8 array.

    Raises
    ------
    ValueError
        If a crop is not ``112 x 112 x 3``.
    TypeError
        If the crops are not ``uint8``.
    """
    if isinstance(images, np.ndarray) and images.ndim == 4:
        x = images
    else:
        seq = list(images)
        if not seq:
            return np.zeros((0, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
        x = np.stack([np.asarray(a) for a in seq])
    if x.ndim != 4 or x.shape[1:] != (INPUT_SIZE, INPUT_SIZE, 3):
        raise ValueError(
            f"expected aligned {INPUT_SIZE}x{INPUT_SIZE} RGB crops, got batch shape "
            f"{x.shape}; resize the crops to {INPUT_SIZE} px first"
        )
    if x.dtype != np.uint8:
        raise TypeError(f"expected uint8 crops, got {x.dtype}")
    return x


def normalize(x: np.ndarray, spec: Normalize, channel_axis: int = 1) -> np.ndarray:
    """Apply a normalisation spec (see :data:`Normalize`) to a float32 batch."""
    if spec is None or (isinstance(spec, str) and spec.lower() == "none"):
        return x
    if isinstance(spec, str):
        key = spec.replace(" ", "").lower()
        if key == "[-1,1]":
            return ((x / 255.0) - 0.5) / 0.5
        if key == "[0,1]":
            return x / 255.0
        raise ValueError(
            f"unknown normalisation {spec!r}; use '[-1,1]', '[0,1]', 'none'"
        )
    if callable(spec):
        return np.asarray(spec(x), dtype=np.float32)
    mean, std = spec
    shape = [1] * x.ndim
    mean_a = np.asarray(mean, dtype=np.float32)
    std_a = np.asarray(std, dtype=np.float32)
    if mean_a.ndim:
        shape[channel_axis] = mean_a.size
        mean_a = mean_a.reshape(shape)
    if std_a.ndim:
        shape = [1] * x.ndim
        shape[channel_axis] = std_a.size
        std_a = std_a.reshape(shape)
    return (x - mean_a) / std_a


def resize_batch(x: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly resize a ``(N, H, W, 3)`` uint8 batch to ``size`` (OpenCV)."""
    if x.shape[1] == size and x.shape[2] == size:
        return x
    import cv2  # noqa: PLC0415

    return np.stack(
        [cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR) for a in x]
    )


def resolve_device(device=None) -> str:
    """``None`` -> ``"cuda"`` when available, else ``"cpu"``; normalise to a string."""
    if device is None:
        try:
            import torch  # noqa: PLC0415

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    device = str(device)
    if device != "cpu" and not device.startswith("cuda"):
        raise ValueError(
            f"device must be 'cpu', 'cuda' or 'cuda:<index>', got {device}"
        )
    return device


def ort_providers(device: str) -> list:
    """ONNX Runtime execution providers for a device string."""
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if ":" in device:
        index = int(device.split(":", 1)[1])
        return [("CUDAExecutionProvider", {"device_id": index}), "CPUExecutionProvider"]
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]


def preload_cuda() -> None:
    """Import torch first so that ONNX Runtime finds the CUDA / cuDNN libraries."""
    try:
        import torch  # noqa: F401, PLC0415
    except ImportError:  # pragma: no cover - torch is a core dependency
        pass


def _check_providers(session, device: str, name: str) -> None:
    if device != "cpu" and "CUDAExecutionProvider" not in session.get_providers():
        logger.warning(
            "%s: CUDAExecutionProvider is not available, running on CPU (install "
            "onnxruntime-gpu and uninstall the CPU-only onnxruntime package)",
            name,
        )


class Embedder:
    """Base class of all embedders.

    Attributes
    ----------
    name
        Registry name.
    family
        Model family (``lvface``, ``arcface``, ``cvlface``, ``edgeface``, ``topofr``,
        or ``onnx`` / ``torch`` / ``custom`` for user-registered models).
    device
        Device string (``cpu``, ``cuda`` or ``cuda:<index>``).
    dim
        Embedding dimension.
    """

    name: str = ""
    family: str = ""
    device: str = "cpu"
    dim: int = EMB_DIM

    def embed(self, images: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
        """Embed aligned 112 x 112 RGB uint8 crops -> ``(N, dim)`` float32 (raw)."""
        x = as_batch(images)
        if len(x) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._embed(x)

    __call__ = embed

    def _embed(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, device={self.device!r})"


class OnnxEmbedder(Embedder):
    """ONNX Runtime embedder with a declarative preprocessing spec.

    The crops are resized to ``size`` (bilinear, only if ``size != 112``), converted
    to ``color`` order, cast to float32, laid out as ``layout`` and normalised with
    ``normalize``. Models with a fixed batch dimension are run in chunks of that
    size (the last chunk is padded and the padding outputs are dropped).

    Parameters
    ----------
    path
        ONNX model file.
    name
        Registry name.
    device
        ``cpu``, ``cuda`` or ``cuda:<index>``.
    input_name, output_name
        Model input / output to use (default: the first of each).
    color
        ``"RGB"`` or ``"BGR"``: channel order the model expects.
    normalize
        Normalisation spec (see :data:`Normalize`).
    size
        Input side length the model expects.
    layout
        ``"NCHW"`` or ``"NHWC"``.
    family
        Family label stored on the embedder.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        name: str,
        device: str = "cpu",
        input_name: str | None = None,
        output_name: str | None = None,
        color: str = "RGB",
        normalize: Normalize = "[-1,1]",
        size: int = INPUT_SIZE,
        layout: str = "NCHW",
        family: str = "onnx",
    ):
        import onnxruntime as ort  # noqa: PLC0415

        color = color.upper()
        layout = layout.upper()
        if color not in ("RGB", "BGR"):
            raise ValueError(f"color must be 'RGB' or 'BGR', got {color!r}")
        if layout not in ("NCHW", "NHWC"):
            raise ValueError(f"layout must be 'NCHW' or 'NHWC', got {layout!r}")
        self.name = name
        self.family = family
        self.device = device
        self.path = Path(path)
        self.color = color
        self.normalize = normalize
        self.size = int(size)
        self.layout = layout
        if device != "cpu":
            preload_cuda()
        self.session = ort.InferenceSession(
            str(self.path), providers=ort_providers(device)
        )
        _check_providers(self.session, device, name)
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        names = [i.name for i in inputs]
        if input_name is None:
            if len(inputs) != 1:
                raise ValueError(f"{path} has inputs {names}; pass input_name=")
            input_name = names[0]
        elif input_name not in names:
            raise ValueError(f"input {input_name!r} not in {names}")
        out_names = [o.name for o in outputs]
        if output_name is None:
            output_name = out_names[0]
        elif output_name not in out_names:
            raise ValueError(f"output {output_name!r} not in {out_names}")
        self.input_name = input_name
        self.output_name = output_name
        shape = inputs[names.index(input_name)].shape
        dim0 = shape[0] if shape else None
        self.fixed_batch = dim0 if isinstance(dim0, int) and dim0 > 0 else None
        out_shape = outputs[out_names.index(output_name)].shape
        if out_shape and isinstance(out_shape[-1], int):
            self.dim = int(out_shape[-1])

    def preprocess(self, x: np.ndarray) -> np.ndarray:
        """Map a ``(N, 112, 112, 3)`` uint8 RGB batch to the model input."""
        x = resize_batch(x, self.size)
        if self.color == "BGR":
            x = x[..., ::-1]
        x = x.astype(np.float32)
        if self.layout == "NCHW":
            x = np.transpose(x, (0, 3, 1, 2))
            axis = 1
        else:
            axis = 3
        x = normalize(x, self.normalize, channel_axis=axis)
        return np.ascontiguousarray(x, dtype=np.float32)

    def _run(self, chunk: np.ndarray) -> np.ndarray:
        n = len(chunk)
        if self.fixed_batch is not None and n < self.fixed_batch:
            # A model exported with a fixed batch size only accepts full batches:
            # pad the last chunk with copies of its last crop, drop their outputs.
            pad = np.repeat(chunk[-1:], self.fixed_batch - n, axis=0)
            chunk = np.concatenate([chunk, pad], 0)
        out = self.session.run([self.output_name], {self.input_name: chunk})[0]
        return np.asarray(out)[:n].reshape(n, -1)

    def _embed(self, x: np.ndarray) -> np.ndarray:
        inp = self.preprocess(x)
        step = self.fixed_batch or len(inp)
        outs = [self._run(inp[i : i + step]) for i in range(0, len(inp), step)]
        out = outs[0] if len(outs) == 1 else np.concatenate(outs, 0)
        return out.astype(np.float32)


class TorchEmbedder(Embedder):
    """Wrap a torch ``nn.Module`` as an embedder with a preprocessing spec.

    The module receives a float32 ``(N, 3, size, size)`` tensor in ``color`` order,
    normalised with ``normalize``; if it returns a tuple or list, the first element
    is the embedding.

    Parameters
    ----------
    module
        The network (moved to ``device`` and set to eval mode).
    name
        Registry name.
    device
        ``cpu``, ``cuda`` or ``cuda:<index>``.
    color, normalize, size
        As for :class:`OnnxEmbedder`.
    forward_kwargs
        Extra keyword arguments of every forward call.
    """

    family = "torch"

    def __init__(
        self,
        module,
        *,
        name: str,
        device: str = "cpu",
        color: str = "RGB",
        normalize: Normalize = "[-1,1]",
        size: int = INPUT_SIZE,
        forward_kwargs: dict | None = None,
    ):
        import torch  # noqa: PLC0415

        color = color.upper()
        if color not in ("RGB", "BGR"):
            raise ValueError(f"color must be 'RGB' or 'BGR', got {color!r}")
        self.torch = torch
        self.name = name
        self.device = device
        self.color = color
        self.normalize = normalize
        self.size = int(size)
        self.forward_kwargs = dict(forward_kwargs or {})
        self.model = module.to(device).eval()

    def _embed(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        x = resize_batch(x, self.size)
        if self.color == "BGR":
            x = x[..., ::-1]
        arr = normalize(
            np.transpose(x.astype(np.float32), (0, 3, 1, 2)), self.normalize
        )
        t = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32)).to(
            self.device
        )
        with torch.no_grad():
            out = self.model(t, **self.forward_kwargs)
        out = out[0] if isinstance(out, (tuple, list)) else out
        return out.float().cpu().numpy().reshape(len(x), -1)
