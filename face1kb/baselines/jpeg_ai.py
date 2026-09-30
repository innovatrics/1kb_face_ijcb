# SPDX-License-Identifier: MIT
"""JPEG-AI through its reference software, run in-process.

JPEG-AI (Rec. ITU-T T.840.1 | ISO/IEC 6048-1) is evaluated with the official
reference software, which is not bundled: ``scripts/setup_jpegai.sh`` clones it
from :data:`REFERENCE_URL`, checks out :data:`REFERENCE_COMMIT`, applies
``third_party/jpeg-ai.patch``, fetches the models and builds the entropy-coder
libraries. Its location is ``face1kb.config.JPEGAI_DIR`` (``FACE1KB_JPEGAI_DIR``).

Configuration of the benchmark
------------------------------
* Coding configuration ``cfg/tools_off.json`` plus the **high operating point**
  profile ``cfg/profiles/high.json`` (HOP) for encode and decode.
* Rate knob: the reference encoder's target bits per pixel times 100
  (``--set_target_bpp``), clamped to :data:`BPP_RANGE` = 2..50.
* Budget fit (:func:`analytic_fit`): one encode at the analytic target
  ``budget * 8 / pixels * 100``; if it overshoots, at most two corrections that
  scale the target by ``0.92 * budget / size``. No bisection.
* A bitstream smaller than :data:`MIN_PLAUSIBLE_FRAC` of the budget is treated as
  a failed encode (a truncated write), never as a fit.
* Decoding first checks the substream structure (:func:`check_bitstream`): the
  reference decoder never returns on an empty, truncated or foreign file.

Runtime notes
-------------
* The reference software resolves its configuration files relative to its
  checkout, so every call runs with the working directory set to the checkout
  (serialised by a process-wide lock) and its top-level ``src`` package on
  ``sys.path``.
* The models of an encoder or decoder instance are loaded by its first call, so
  one instance is kept per coding configuration.
* It runs on the current CUDA device; select a GPU with ``CUDA_VISIBLE_DEVICES``.
* Bitstreams and reconstructions are written to a temporary directory
  (``TMPDIR``); stored bitstreams are copied there before decoding.
* With ``--set_target_bpp`` the reference encoder rewrites a per-image bitrate
  log inside the checkout (``cfg/betas/*/betas_*.txt``) that it never reads back;
  :class:`ReferenceSoftware` redirects it to a private temporary folder by default,
  which leaves the bitstreams unchanged and the checkout untouched.
* The entropy coder writes ``lib_wrappers/mans/cache.pt`` into the checkout on its
  first use; run one encode before starting concurrent processes.
* Instances hold GPU state and are not thread-safe; use one process per GPU.
* The reference software switches PyTorch to cuDNN's deterministic algorithms and
  disables TF32 globally, and never switches back. The wrapper applies those
  settings for the duration of each call and restores the caller's settings
  afterwards, so torch code that runs later in the same process (CompressAI, the
  face1kb codecs) keeps its own numerics. The paper's scripts did not restore them;
  see :mod:`face1kb.baselines.compressai_codecs` for the one place where this
  mattered.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import sysconfig
import tempfile
import threading
import weakref
from collections.abc import Callable, Sequence
from contextlib import contextmanager, redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from .search import FitResult

logger = logging.getLogger(__name__)

__all__ = [
    "BPP_RANGE",
    "DECODE_OPTIONS",
    "DEFAULT_PROFILE",
    "EXTENSION",
    "MIN_PLAUSIBLE_FRAC",
    "PROFILES",
    "REFERENCE_COMMIT",
    "REFERENCE_URL",
    "JpegAIError",
    "ReferenceSoftware",
    "analytic_fit",
    "analytic_guess",
    "available",
    "check_bitstream",
    "check_setup",
    "decode",
    "decode_file",
    "encode_bpp",
    "encode_file",
    "encode_to_budget",
    "get_reference",
    "hinted_fit",
    "step_down_fit",
]

#: Official repository of the JPEG-AI reference software.
REFERENCE_URL = "https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software.git"
#: Commit used for the paper.
REFERENCE_COMMIT = "e9648f9a4bb98d8fb7333e97d4df4f0979520a7c"
#: File extension of a JPEG-AI bitstream.
EXTENSION = ".jpegai"
#: Operating points -> profile files under ``cfg/profiles`` (simple / base / high).
PROFILES: dict[str, str] = {"sop": "simple", "bop": "base", "hop": "high"}
#: Operating point of the benchmark grid.
DEFAULT_PROFILE = "hop"
#: Coding configuration of the benchmark (all optional tools off).
TOOLS_OFF_CFG = "cfg/tools_off.json"
#: Target bpp x 100 limits of the budget fit (the models top out at 0.5 bpp).
BPP_RANGE: tuple[int, int] = (2, 50)
#: A bitstream below this fraction of the budget is a failed encode, not a fit.
MIN_PLAUSIBLE_FRAC = 0.25
#: Size recorded for a failed encode inside the budget searches.
FAILED_SIZE = 1 << 30
#: Keyword options of :func:`decode`.
DECODE_OPTIONS: tuple[str, ...] = (
    "profile",
    "tools_off",
    "extra_args",
    "workdir",
    "repo_dir",
)


class JpegAIError(RuntimeError):
    """The reference software is missing or produced no usable bitstream."""


# ------------------------------------------------------------ bitstream check
#: Start-of-codestream and end-of-codestream markers.
MARKER_SOC, MARKER_EOC = 0xFF80, 0xFF81
#: Picture-header marker (the first substream).
MARKER_PIH = 0xFF82
#: Markers of the substreams defined by the reference software
#: (``src/codec/bitstream_structure/layouts_def.py``).
SUBSTREAM_MARKERS = frozenset(
    (0xFF82, 0xFF83, 0xFF84, 0xFF88, 0xFF89, 0xFF8A, 0xFF8B, 0xFF8C)
)


def check_bitstream(data: bytes) -> None:
    """Check the substream structure of a JPEG-AI bitstream.

    Walks the markers as the reference decoder's ``read_substreams`` does: an
    SOC marker, then substreams (2-byte marker, exp-Golomb (k=0) coded size,
    payload), the first one being the picture header, up to an EOC marker. Bytes
    after EOC are ignored, as by the reference decoder.

    The reference decoder does not check for the end of the file while it looks
    for EOC and loops forever on an empty, truncated or foreign file, so
    :meth:`ReferenceSoftware.decode_file` runs this check first.

    Raises
    ------
    JpegAIError
        If ``data`` is not a complete JPEG-AI bitstream.
    """
    n = len(data)
    if n < 2 or int.from_bytes(data[:2], "big") != MARKER_SOC:
        raise JpegAIError("not a JPEG-AI bitstream (no SOC marker 0xFF80)")
    pos, first = 2, True
    while True:
        if pos + 2 > n:
            raise JpegAIError(f"truncated JPEG-AI bitstream (no EOC marker, {n} B)")
        marker = int.from_bytes(data[pos : pos + 2], "big")
        pos += 2
        if marker == MARKER_EOC:
            return
        if marker not in SUBSTREAM_MARKERS:
            raise JpegAIError(
                f"unknown JPEG-AI marker 0x{marker:04X} at byte {pos - 2}"
            )
        if first and marker != MARKER_PIH:
            raise JpegAIError("JPEG-AI bitstream does not start with a picture header")
        first = False
        # substream size: unsigned exp-Golomb k=0, MSB first, byte-aligned after
        bit, zeros = pos * 8, 0
        while True:
            if bit >= n * 8 or zeros > 32:
                raise JpegAIError("truncated JPEG-AI substream header")
            b = (data[bit >> 3] >> (7 - (bit & 7))) & 1
            bit += 1
            if b:
                break
            zeros += 1
        if bit + zeros > n * 8:
            raise JpegAIError("truncated JPEG-AI substream header")
        value = 0
        for _ in range(zeros):
            value = (value << 1) | ((data[bit >> 3] >> (7 - (bit & 7))) & 1)
            bit += 1
        size = (1 << zeros) + value - 1
        pos = (bit + 7) >> 3
        if pos + size > n:
            raise JpegAIError(
                f"truncated JPEG-AI substream 0x{marker:04X} "
                f"({size} B declared, {n - pos} B left)"
            )
        pos += size


# --------------------------------------------------------------------- setup
def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def check_setup(repo_dir: str | Path | None = None) -> list[str]:
    """List what is missing in a reference-software checkout (empty when usable)."""
    repo = Path(repo_dir) if repo_dir is not None else _default_repo_dir()
    if not (repo / "src" / "reco" / "coders").is_dir():
        return [f"no JPEG-AI reference software checkout at {repo}"]
    problems = []
    models = [p for p in (repo / "models").rglob("*.pth") if p.is_file()]
    real = [p for p in models if p.stat().st_size > 1024]
    if not real:
        problems.append(
            f"no model weights under {repo / 'models'} (git lfs pull --include "
            "'models/**' in the checkout)"
        )
    wrappers = repo / "src" / "codec" / "entropy_coding" / "lib_wrappers"
    suffix = _ext_suffix()
    for sub, stem in (("mans", "ans"), ("direct", "ec_direct")):
        if not (wrappers / sub / f"{stem}{suffix}").is_file():
            problems.append(
                f"entropy-coder library {sub}/{stem}{suffix} is not built for this "
                "Python (run scripts/setup_jpegai.sh)"
            )
    return problems


def _default_repo_dir() -> Path:
    from face1kb import config  # noqa: PLC0415

    return config.JPEGAI_DIR


def available(repo_dir: str | Path | None = None) -> bool:
    """Return True if the reference software, its models and libraries are present."""
    return not check_setup(repo_dir)


def cfg_files(profile: str | None = DEFAULT_PROFILE, tools_off: bool = True) -> list:
    """Return the configuration files passed with ``--cfg`` for an operating point.

    ``profile`` is ``"sop"``, ``"bop"``, ``"hop"`` (or the profile file stem
    ``"simple"``, ``"base"``, ``"high"``), or ``None`` for no profile.
    """
    files = [TOOLS_OFF_CFG] if tools_off else []
    if profile is not None:
        stem = PROFILES.get(profile.lower(), profile.lower())
        if stem not in PROFILES.values():
            raise ValueError(f"unknown JPEG-AI profile {profile!r}")
        files.append(f"cfg/profiles/{stem}.json")
    return files


_CWD_LOCK = threading.RLock()


@contextmanager
def _reference_numerics():
    """Apply the reference software's torch settings; restore the caller's on exit.

    The reference software sets ``cudnn.deterministic = True``, ``cudnn.benchmark =
    False`` and disables TF32 in its coding functions and leaves them set; they are
    applied here before each call (as for every call after the first in a process)
    and the previous values are restored afterwards.
    """
    import torch  # noqa: PLC0415

    cudnn, matmul = torch.backends.cudnn, torch.backends.cuda.matmul
    prev = (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32, matmul.allow_tf32)
    cudnn.deterministic, cudnn.benchmark = True, False
    cudnn.allow_tf32, matmul.allow_tf32 = False, False
    try:
        yield
    finally:
        (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32, matmul.allow_tf32) = (
            prev
        )


class ReferenceSoftware:
    """In-process driver of one JPEG-AI reference-software checkout.

    Parameters
    ----------
    repo_dir
        The checkout (default ``face1kb.config.JPEGAI_DIR``).
    quiet
        Suppress the reference software's console output.
    private_bitrate_log
        Redirect the bitrate matcher's per-image log (``cfg/betas/*/betas_*.txt``
        inside the checkout) to a private temporary folder of this instance.
        With ``--set_target_bpp`` the reference software enables bitrate matching
        in a mode that only writes that file (it deletes and re-creates it and
        never reads it back), so the bitstreams are unaffected; redirecting it keeps
        the checkout unchanged and lets concurrent processes share one checkout.
    """

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        quiet: bool = True,
        private_bitrate_log: bool = True,
    ):
        self.repo_dir = Path(repo_dir or _default_repo_dir()).resolve()
        self.quiet = quiet
        self.private_bitrate_log = private_bitrate_log
        self._encoders: dict[tuple, object] = {}
        self._decoders: dict[tuple, object] = {}
        self._brm_cfg: str | None = None

    def __repr__(self) -> str:
        return f"ReferenceSoftware({str(self.repo_dir)!r})"

    # ----------------------------------------------------------------- plumbing
    def _bitrate_log_cfg(self) -> str:
        """Return a configuration pointing the bitrate matcher at a private folder."""
        if self._brm_cfg is None:
            folder = tempfile.mkdtemp(prefix="jpegai_brm_")
            weakref.finalize(self, shutil.rmtree, folder, ignore_errors=True)
            cfg = os.path.join(folder, "bitrate_log.json")
            with open(cfg, "w", encoding="utf-8") as f:
                json.dump(
                    {"model": {"bitrate_matcher": {"bitrate_config_path": folder}}}, f
                )
            self._brm_cfg = cfg
        return self._brm_cfg

    def require(self) -> None:
        """Raise :class:`JpegAIError` if the checkout is not usable."""
        problems = check_setup(self.repo_dir)
        if problems:
            raise JpegAIError(
                "JPEG-AI reference software not usable: "
                + "; ".join(problems)
                + ". See docs/baselines.md (scripts/setup_jpegai.sh)."
            )

    @contextmanager
    def _inside(self):
        repo = str(self.repo_dir)
        loaded = sys.modules.get("src")
        if loaded is not None:
            origin = str(Path(getattr(loaded, "__file__", "") or "").resolve())
            if not origin.startswith(repo + os.sep):
                raise JpegAIError(
                    "a different top-level module 'src' is already imported "
                    f"({origin}); the JPEG-AI reference software needs that name"
                )
        with _CWD_LOCK, _reference_numerics():
            prev = os.getcwd()
            if repo not in sys.path:
                sys.path.insert(0, repo)
            os.chdir(repo)
            try:
                if self.quiet:
                    with redirect_stdout(io.StringIO()):
                        yield
                else:
                    yield
            finally:
                os.chdir(prev)

    def _coder(self, kind: str, key: tuple):
        cache = self._encoders if kind == "enc" else self._decoders
        coder = cache.get(key)
        if coder is None:
            self.require()
            with self._inside():
                from src.reco.coders import (  # noqa: PLC0415
                    RecoDecoderProcess,
                    RecoEncoderProcess,
                )

                cls = RecoEncoderProcess if kind == "enc" else RecoDecoderProcess
                coder = cache[key] = cls(None)
        return coder

    # ----------------------------------------------------------------- file API
    def encode_file(
        self,
        png_path: str | Path,
        bit_path: str | Path,
        bpp_m100: int,
        *,
        rec_path: str | Path | None = None,
        profile: str | None = DEFAULT_PROFILE,
        tools_off: bool = True,
        extra_args: Sequence[str] = (),
    ) -> bool:
        """Encode a PNG at a target bpp x 100; return True if a bitstream was written.

        Runs ``src.reco.coders.encoder`` with ``<png> <bits> --set_target_bpp
        <bpp_m100> --cfg <cfg files> [-r <rec_path>] [extra_args]`` (plus the
        private bitrate-log configuration, see the class docstring). Errors of the
        reference software are logged and reported as ``False``.
        """
        cfg = cfg_files(profile, tools_off)
        if self.private_bitrate_log and cfg:
            # (without --cfg the reference encoder loads its default test
            # configuration, which an added file would replace)
            cfg.append(self._bitrate_log_cfg())
        args = [str(png_path), str(bit_path), "--set_target_bpp", str(int(bpp_m100))]
        if cfg:
            args += ["--cfg", *cfg]
        if rec_path is not None:
            args += ["-r", str(rec_path)]
        args += [str(a) for a in extra_args]
        enc = self._coder("enc", (tuple(cfg), tuple(extra_args)))
        try:
            with self._inside():
                enc.process(args)
        except Exception as exc:  # noqa: BLE001 - the reference SW raises anything
            logger.warning("JPEG-AI encode of %s failed: %s", png_path, exc)
            return False
        return os.path.exists(bit_path)

    def decode_file(
        self,
        bit_path: str | Path,
        png_path: str | Path,
        *,
        profile: str | None = DEFAULT_PROFILE,
        tools_off: bool = True,
        extra_args: Sequence[str] = (),
    ) -> bool:
        """Decode a bitstream to a PNG; return True if the PNG was written.

        Runs ``src.reco.coders.decoder`` with ``<bits> <png> [--cfg <cfg files>]
        [extra_args]``. The decoder may create log folders two levels above
        ``bit_path``; :meth:`decode` therefore works on a temporary copy.

        The file is first checked with :func:`check_bitstream` (the reference
        decoder never returns on an empty, truncated or foreign file); a file that
        fails the check is logged and reported as ``False``.
        """
        try:
            check_bitstream(Path(bit_path).read_bytes())
        except (OSError, JpegAIError) as exc:
            logger.warning("JPEG-AI decode of %s refused: %s", bit_path, exc)
            return False
        cfg = cfg_files(profile, tools_off)
        args = [str(bit_path), str(png_path)]
        if cfg:
            args += ["--cfg", *cfg]
        args += [str(a) for a in extra_args]
        dec = self._coder("dec", (tuple(cfg), tuple(extra_args)))
        try:
            with self._inside():
                dec.process(args)
        except Exception as exc:  # noqa: BLE001
            logger.warning("JPEG-AI decode of %s failed: %s", bit_path, exc)
            return False
        return os.path.exists(png_path)

    # ---------------------------------------------------------------- array API
    def encode_bpp(
        self,
        img,
        bpp_m100: int,
        *,
        profile: str | None = DEFAULT_PROFILE,
        tools_off: bool = True,
        return_recon: bool = False,
        workdir: str | Path | None = None,
    ):
        """Encode an image at one target bpp x 100.

        Returns the bitstream (``None`` if the encoder failed), or
        ``(bitstream, reconstruction)`` with ``return_recon`` (the encoder-side
        reconstruction as ``uint8`` array).
        """
        with tempfile.TemporaryDirectory(prefix="jpegai_", dir=workdir) as td:
            png = os.path.join(td, "input.png")
            _to_pil(img).save(png)
            out = self._encode_png(png, td, bpp_m100, profile, tools_off, return_recon)
        return out

    def _encode_png(self, png, td, bpp, profile, tools_off, return_recon):
        bits = os.path.join(td, f"stream_{int(bpp)}.bin")
        rec = os.path.join(td, f"stream_{int(bpp)}.png")
        ok = self.encode_file(
            png, bits, bpp, rec_path=rec, profile=profile, tools_off=tools_off
        )
        data = Path(bits).read_bytes() if ok else None
        recon = None
        if return_recon and ok and os.path.exists(rec):
            with Image.open(rec) as im:
                recon = np.asarray(im.convert("RGB"), dtype=np.uint8)
        for p in (bits, rec):
            if os.path.exists(p):
                os.remove(p)
        return (data, recon) if return_recon else data

    def decode(
        self,
        data: bytes,
        *,
        profile: str | None = DEFAULT_PROFILE,
        tools_off: bool = True,
        extra_args: Sequence[str] = (),
        workdir: str | Path | None = None,
    ) -> np.ndarray | None:
        """Decode a bitstream to an ``uint8`` RGB array, ``None`` on error.

        Malformed input (empty, truncated or not JPEG-AI, see
        :func:`check_bitstream`) is an error.
        """
        with tempfile.TemporaryDirectory(prefix="jpegai_", dir=workdir) as td:
            sub = os.path.join(td, "bits")
            os.makedirs(sub)
            bits = os.path.join(sub, "stream" + EXTENSION)
            Path(bits).write_bytes(data)
            rec = os.path.join(td, "rec.png")
            ok = self.decode_file(
                bits, rec, profile=profile, tools_off=tools_off, extra_args=extra_args
            )
            if not ok:
                return None
            with Image.open(rec) as im:
                return np.asarray(im.convert("RGB"), dtype=np.uint8)

    def encode_to_budget(
        self,
        img,
        budget: int,
        *,
        profile: str | None = DEFAULT_PROFILE,
        tools_off: bool = True,
        bpp_range: tuple[int, int] = BPP_RANGE,
        min_plausible_frac: float = MIN_PLAUSIBLE_FRAC,
        workdir: str | Path | None = None,
    ) -> tuple[bytes, dict]:
        """Encode with the analytic budget fit; see :func:`encode_to_budget`."""
        self.require()
        pil = _to_pil(img)
        px = pil.size[0] * pil.size[1]
        lo, hi = bpp_range
        with tempfile.TemporaryDirectory(prefix="jpegai_", dir=workdir) as td:
            png = os.path.join(td, "input.png")
            pil.save(png)

            def encode(bpp):
                bpp = int(max(lo, min(hi, round(bpp))))
                data = self._encode_png(png, td, bpp, profile, tools_off, False)
                if data is None:
                    return bpp, FAILED_SIZE, b""
                if len(data) < min_plausible_frac * budget:
                    logger.warning(
                        "JPEG-AI produced an implausible %d B stream for a %d B budget "
                        "at bpp x100 = %d; treated as a failed encode",
                        len(data),
                        budget,
                        bpp,
                    )
                    return bpp, FAILED_SIZE, b""
                return bpp, len(data), data

            r = analytic_fit(encode, analytic_guess(budget, px), budget, lo, hi)
        if not r.payload:
            raise JpegAIError(
                f"JPEG-AI produced no usable bitstream (last target bpp x100 = "
                f"{r.setting}, budget {budget} B)"
            )
        return r.payload, {"fitted": r.fitted, "setting": r.setting, "size": r.size}


def _to_pil(img) -> Image.Image:
    from .classical import to_pil  # noqa: PLC0415

    return to_pil(img, strip_metadata=True)


# --------------------------------------------------------------- budget fits
def analytic_guess(budget: int, pixels: int) -> float:
    """Analytic first target: ``budget * 8 / pixels * 100`` (bpp x 100, unrounded)."""
    return budget * 8 / pixels * 100


def analytic_fit(
    encode: Callable[[float], tuple[int, int, bytes]],
    guess: float,
    budget: int,
    lo: int,
    hi: int,
) -> FitResult:
    """Predict-then-correct fit of the target bpp (at most three encodes).

    Parameters
    ----------
    encode
        ``encode(bpp) -> (bpp_used, size, data)``; it clamps and rounds ``bpp``
        to an integer in ``[lo, hi]`` and reports a failed encode as a huge size.
    guess
        First target (:func:`analytic_guess`).
    budget
        Byte budget.
    lo, hi
        Target limits.

    Returns
    -------
    FitResult
        ``setting`` is the target bpp x 100 of the returned stream. When nothing
        fits, the last encode is returned with ``fitted=False``.

    Notes
    -----
    After an overshoot the next target is ``round(bpp * budget / size * 0.92)``
    clamped to ``[lo, hi]``; a target already tried is lowered by 2, and the search
    stops if it is still a repeat or below ``lo``. A failed encode counts as
    :data:`FAILED_SIZE`, so the correction after it targets ``lo`` (bpp x100 = 2),
    and that stream is kept if it is plausible.
    """
    bpp, size, data = encode(guess)
    best = (bpp, size, data) if size <= budget else None
    tried = {bpp}
    for _ in range(2):
        if best is not None:
            break
        if size >= FAILED_SIZE:
            logger.warning(
                "JPEG-AI encode at bpp x100 = %d failed; the correction targets the "
                "lowest rate %d",
                bpp,
                lo,
            )
        nxt = int(max(lo, min(hi, round(bpp * budget / max(size, 1) * 0.92))))
        if nxt in tried:
            nxt -= 2
        if nxt < lo or nxt in tried:
            break
        tried.add(nxt)
        bpp, size, data = encode(nxt)
        if size <= budget:
            best = (bpp, size, data)
    if best is not None:
        return FitResult(best[0], best[1], best[2], True)
    return FitResult(bpp, size, data, False)


def step_down_fit(
    encode: Callable[[int], tuple[int, bytes] | None],
    budget: int,
    pixels: int,
    *,
    lo: int = BPP_RANGE[0],
    hi: int = BPP_RANGE[1],
    tries: int = 3,
) -> FitResult | None:
    """Budget fit of the codec-comparison and sample-difficulty studies.

    Starts at the analytic target rounded and clamped to ``[lo, hi]``; after an
    overshoot the target becomes ``round(bpp * budget / size * 0.92)`` and the loop
    stops when that is not lower than the current target, is below ``2`` or the
    encoder fails. ``encode(bpp)`` returns ``(size, data)`` or ``None`` on failure.
    No plausibility floor is applied. Returns the first fitting encode, else the
    last attempted encode with ``fitted=False`` if it succeeded, else ``None``.
    """
    bpp = int(max(lo, min(hi, round(analytic_guess(budget, pixels)))))
    last = None
    for _ in range(tries):
        out = encode(bpp)
        if out is None:
            last = None
            break
        size, data = out
        last = FitResult(bpp, size, data, size <= budget)
        if size <= budget:
            return last
        nb = int(round(bpp * budget / max(size, 1) * 0.92))
        if nb >= bpp or nb < 2:
            break
        bpp = nb
    return last


def hinted_fit(
    encode: Callable[[int], tuple[int, bytes] | None],
    budget: int,
    pixels: int,
    *,
    hint: int | None = None,
    lo: int = BPP_RANGE[0],
    hi: int = BPP_RANGE[1],
    tries: int = 3,
    min_plausible_frac: float = MIN_PLAUSIBLE_FRAC,
) -> FitResult | None:
    """Budget fit of the ISO/IEC 29794-5 annex study (warm-started target).

    Starts at ``hint`` (the target that fitted the previous crop) or at the
    clamped analytic target, stops on a repeated target, and accepts a stream only
    if ``min_plausible_frac * budget <= size <= budget``. After a miss the target
    becomes ``round(bpp * budget / size * 0.92)`` clamped to ``[lo, hi]`` (a failed
    encode counts as a huge size). Returns the accepted encode or ``None``.
    """
    guess = hint or int(max(lo, min(hi, round(analytic_guess(budget, pixels)))))
    tried: set[int] = set()
    for _ in range(tries):
        if guess in tried:
            break
        tried.add(guess)
        out = encode(guess)
        size, data = out if out is not None else (FAILED_SIZE, b"")
        if min_plausible_frac * budget <= size <= budget:
            return FitResult(guess, size, data, True)
        guess = int(max(lo, min(hi, round(guess * budget / max(size, 1) * 0.92))))
    return None


# ------------------------------------------------------- module-level helpers
_DEFAULT: dict[str, ReferenceSoftware] = {}


def get_reference(repo_dir: str | Path | None = None) -> ReferenceSoftware:
    """Shared :class:`ReferenceSoftware` for ``repo_dir`` (default checkout)."""
    key = str(Path(repo_dir or _default_repo_dir()).resolve())
    ref = _DEFAULT.get(key)
    if ref is None:
        ref = _DEFAULT[key] = ReferenceSoftware(key)
    return ref


def encode_file(png_path, bit_path, bpp_m100: int, **kwargs) -> bool:
    """:meth:`ReferenceSoftware.encode_file` on the default checkout."""
    return get_reference(kwargs.pop("repo_dir", None)).encode_file(
        png_path, bit_path, bpp_m100, **kwargs
    )


def decode_file(bit_path, png_path, **kwargs) -> bool:
    """:meth:`ReferenceSoftware.decode_file` on the default checkout."""
    return get_reference(kwargs.pop("repo_dir", None)).decode_file(
        bit_path, png_path, **kwargs
    )


def encode_bpp(img, bpp_m100: int, **kwargs):
    """:meth:`ReferenceSoftware.encode_bpp` on the default checkout."""
    return get_reference(kwargs.pop("repo_dir", None)).encode_bpp(
        img, bpp_m100, **kwargs
    )


def decode(data: bytes, **kwargs) -> np.ndarray | None:
    """Decode a JPEG-AI bitstream with the benchmark configuration (HOP, tools off).

    Keyword arguments are passed to :meth:`ReferenceSoftware.decode`
    (``repo_dir`` selects the checkout). Returns ``None`` if decoding failed,
    including for a malformed bitstream (:func:`check_bitstream`).
    """
    return get_reference(kwargs.pop("repo_dir", None)).decode(data, **kwargs)


def encode_to_budget(img, budget: int, **kwargs) -> tuple[bytes, dict]:
    """Encode an image to at most ``budget`` bytes as in the benchmark.

    Parameters
    ----------
    img
        ``H x W x 3`` ``uint8`` RGB array (or PIL image).
    budget
        Byte budget.
    **kwargs
        ``profile`` (default ``"hop"``), ``tools_off`` (default True),
        ``bpp_range``, ``min_plausible_frac``, ``workdir`` and ``repo_dir``.

    Returns
    -------
    tuple[bytes, dict]
        The bitstream and ``{"fitted", "setting", "size"}``; ``setting`` is the
        target bpp x 100. If no target fits, the last encode is returned with
        ``fitted=False``.

    Raises
    ------
    JpegAIError
        If the reference software is not set up, or if the final attempt produced
        no plausible bitstream.
    """
    return get_reference(kwargs.pop("repo_dir", None)).encode_to_budget(
        img, budget, **kwargs
    )
