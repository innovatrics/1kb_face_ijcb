# SPDX-License-Identifier: MIT
"""Decode any stored bitstream of the benchmark by its file extension.

========  =====================================================================
ext       decoder
========  =====================================================================
.jpg      Pillow (JPEG)
.jp2      Pillow (JPEG 2000)
.webp     Pillow (WebP)
.jxl      Pillow + pillow-jxl-plugin (JPEG XL)
.avif     Pillow (AVIF)
.heic     Pillow + pillow-heif (HEIF)
.fzt      JPEG-FzT inverse F-transform (:mod:`face1kb.baselines.jpeg_fzt`)
.jpegai   JPEG-AI reference decoder, HOP profile (:mod:`face1kb.baselines.jpeg_ai`)
.ptci     CompressAI container (:mod:`face1kb.baselines.compressai_codecs`)
.bin      face1kb container, FAST or ACCURATE by its header (:func:`face1kb.decode`)
========  =====================================================================

All decoders return an ``H x W x 3`` ``uint8`` RGB array.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["DECODERS", "decode_bytes", "decode_file", "decoded_cache_path"]

#: Extensions handled by :func:`decode_bytes`.
DECODERS: tuple[str, ...] = (
    ".jpg",
    ".jp2",
    ".webp",
    ".jxl",
    ".avif",
    ".heic",
    ".fzt",
    ".jpegai",
    ".ptci",
    ".bin",
)
_PILLOW = (".jpg", ".jpeg", ".jp2", ".webp", ".jxl", ".avif", ".heic", ".heif", ".png")


def decode_bytes(
    data: bytes,
    ext: str,
    *,
    res: int | None = None,
    device: str = "cuda",
    weights_dir: str | Path | None = None,
    **jpegai_kwargs,
) -> np.ndarray:
    """Decode ``data`` stored with file extension ``ext``.

    Parameters
    ----------
    data
        File contents.
    ext
        File extension, e.g. ``".webp"`` (see the module table).
    res
        Output size of JPEG-FzT (default: twice the stored low-resolution size,
        i.e. the original size for the even benchmark resolutions).
    device
        Torch device of the CompressAI and face1kb decoders. The paper
        bitstreams must be decoded on CUDA.
    weights_dir
        Folder of the face1kb weights for ``.bin`` (default
        ``face1kb.config.WEIGHTS_DIR``).
    **jpegai_kwargs
        Options of :func:`face1kb.baselines.jpeg_ai.decode` for ``.jpegai``
        (``profile``, ``tools_off``, ``extra_args``, ``workdir``, ``repo_dir``).
        Options a decoder does not take are ignored.

    Raises
    ------
    ValueError
        For an unknown extension.
    face1kb.baselines.jpeg_ai.JpegAIError
        If the JPEG-AI reference decoder fails.
    """
    ext = ext.lower() if ext.startswith(".") else "." + ext.lower()
    if ext in _PILLOW:
        from . import classical  # noqa: PLC0415

        return classical.decode(data)
    if ext == ".fzt":
        from . import jpeg_fzt  # noqa: PLC0415

        return jpeg_fzt.decode(data, res)
    if ext == ".jpegai":
        from . import jpeg_ai  # noqa: PLC0415

        opts = {k: v for k, v in jpegai_kwargs.items() if k in jpeg_ai.DECODE_OPTIONS}
        out = jpeg_ai.decode(data, **opts)
        if out is None:
            raise jpeg_ai.JpegAIError("JPEG-AI decoding failed")
        return out
    if ext == ".ptci":
        from . import compressai_codecs  # noqa: PLC0415

        return compressai_codecs.decode(data, device=device)
    if ext == ".bin":
        import face1kb  # noqa: PLC0415

        return face1kb.decode(data, device=device, weights_dir=weights_dir)
    raise ValueError(f"no decoder for extension {ext!r}; known: {DECODERS}")


def decode_file(
    path: str | Path,
    *,
    res: int | None = None,
    device: str = "cuda",
    cache: bool = False,
    **kwargs,
) -> np.ndarray:
    """Decode the bitstream file ``path``, choosing the decoder by its extension.

    See :func:`decode_bytes` for the arguments. The file is only read; decoders
    that need files (JPEG-AI) work on a temporary copy. With ``cache=True`` the
    decoded PNG at :func:`decoded_cache_path` is returned when it exists (the
    benchmark cached the slow JPEG-AI and face1kb decodes this way); otherwise the
    bitstream is decoded.
    """
    path = Path(path)
    if cache:
        cached = decoded_cache_path(path)
        if cached is not None and cached.is_file():
            from PIL import Image  # noqa: PLC0415

            with Image.open(cached) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)
    return decode_bytes(
        path.read_bytes(), path.suffix, res=res, device=device, **kwargs
    )


def decoded_cache_path(bitstream: str | Path) -> Path | None:
    """PNG cache location of a bitstream inside a compressed-cell tree.

    Maps ``.../compressed/<cell>/<codec>/<rel>.<ext>`` to
    ``.../decoded/<cell>/<codec>/<rel>.png`` (the layout of
    :func:`face1kb.config.decoded_dir`), or with ``FACE1KB_LAYOUT=legacy`` to the
    sibling ``.../compressed/<cell>/<codec>_png/<rel>.png``. Returns ``None`` if the
    path is not inside a ``compressed`` tree.
    """
    from face1kb import config  # noqa: PLC0415

    parts = Path(bitstream).parts
    if "compressed" not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index("compressed")
    if len(parts) < i + 4:
        return None
    cell, codec, rel = parts[i + 1], parts[i + 2], parts[i + 3 :]
    if config.LAYOUT == "legacy":
        root = Path(*parts[: i + 1], cell, codec + "_png")
    else:
        root = Path(*parts[:i], "decoded", cell, codec)
    return root.joinpath(*rel).with_suffix(".png")
