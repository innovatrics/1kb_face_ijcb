# SPDX-License-Identifier: MIT
"""The five built-in evaluator families of the paper's 14-model roster.

Preprocessing per family (inputs are aligned 112 x 112 RGB uint8 crops):

==========  ==========================================================  ==========
family      preprocessing                                               runtime
==========  ==========================================================  ==========
lvface      RGB, ``(x / 255 - 0.5) / 0.5``, NCHW                        ONNX
arcface     RGB -> BGR, then insightface ``get_feat`` (which swaps back ONNX
            to RGB and applies its own mean/std)                        (insightface)
cvlface     RGB, ``(x / 255 - 0.5) / 0.5``, NCHW; ViT-B also receives   torch
            the fixed ArcFace 5-point template as keypoints
edgeface    RGB, ``(x / 255 - 0.5) / 0.5``, NCHW                        torch
topofr      RGB, ``(x / 255 - 0.5) / 0.5``, NCHW, ``phase="infer"``     torch
==========  ==========================================================  ==========

The outputs are the raw embeddings of each network (not L2-normalised).
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import sources
from .download import ensure_file, models_root
from .embedders import Embedder, OnnxEmbedder, _check_providers, ort_providers
from .embedders import preload_cuda as _preload_cuda

logger = logging.getLogger(__name__)

#: Canonical ArcFace 5-point template for 112 x 112 crops (eyes, nose tip, mouth
#: corners; x, y in pixels). CVLface ViT-B receives it, divided by 112, as the
#: keypoints of every crop instead of running its own landmark detector.
ARCFACE_TEMPLATE_112: tuple[tuple[float, float], ...] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


def _ensure_all(name: str, files, download: bool, verify: bool) -> list[Path]:
    return [ensure_file(f, download=download, verify=verify, model=name) for f in files]


# ----------------------------------------------------------------------- LVFace
def load_lvface(
    name: str, variant: str, device: str, download: bool, verify: bool
) -> OnnxEmbedder:
    """LVFace-T/S/B/L (ViT, Glint360K) ONNX embedder."""
    (path,) = _ensure_all(name, (sources.lvface_file(variant),), download, verify)
    return OnnxEmbedder(
        path,
        name=name,
        device=device,
        color="RGB",
        normalize="[-1,1]",
        family="lvface",
    )


# ---------------------------------------------------------------------- ArcFace
class ArcFaceEmbedder(Embedder):
    """insightface ArcFace ResNet-100 (``glintr100.onnx`` of ``antelopev2``).

    The model file is loaded by explicit path through
    ``insightface.model_zoo.get_model`` and embedded with its ``get_feat``; there is
    no fallback to any other insightface model.

    Parameters
    ----------
    path
        ``glintr100.onnx``.
    name
        Registry name.
    device
        ``cpu``, ``cuda`` or ``cuda:<index>``.
    """

    family = "arcface"

    def __init__(self, path: str | Path, *, name: str, device: str):
        if device != "cpu":
            _preload_cuda()
        from insightface.model_zoo import get_model  # noqa: PLC0415

        self.name = name
        self.device = device
        self.path = Path(path)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rec = get_model(str(self.path), providers=ort_providers(device))
        logger.debug("insightface: %s", buf.getvalue().strip())
        if rec is None or getattr(rec, "taskname", None) != "recognition":
            raise RuntimeError(f"{path} is not an insightface recognition model")
        rec.prepare(ctx_id=-1 if device == "cpu" else 0)
        _check_providers(rec.session, device, name)
        self.rec = rec
        self.session = rec.session

    def _embed(self, x: np.ndarray) -> np.ndarray:
        # get_feat expects BGR crops (cv2 convention) and swaps them back to RGB.
        bgr = [a[:, :, ::-1] for a in x]
        return self.rec.get_feat(bgr).astype(np.float32)


def load_arcface(
    name: str, variant: str, device: str, download: bool, verify: bool
) -> ArcFaceEmbedder:
    """ArcFace R100 (Glint360K) from the insightface ``antelopev2`` pack."""
    (path,) = _ensure_all(name, (sources.GLINTR100,), download, verify)
    return ArcFaceEmbedder(path, name=name, device=device)


# ---------------------------------------------------------------------- CVLface
#: variant -> snapshot folder of the CVLface model loaded in this process.
_CVLFACE_LOADED: dict[str, str] = {}


def _claim_cvlface(variant: str, snap: str) -> None:
    """Enforce one CVLface snapshot per process.

    Both snapshots ship a top-level Python package called ``models`` with different
    contents; once one is imported, the other snapshot would silently build its
    network from the wrong package.
    """
    for other, other_snap in _CVLFACE_LOADED.items():
        if other != variant or other_snap != snap:
            raise RuntimeError(
                f"cvlface_{variant} cannot be loaded in a process that already "
                f"loaded cvlface_{other}: both snapshots ship a top-level `models` "
                "package. Load each CVLface model in its own process."
            )
    mod = sys.modules.get("models")
    if mod is not None and variant not in _CVLFACE_LOADED:
        where = os.path.dirname(os.path.dirname(os.path.abspath(mod.__file__ or "")))
        if where != snap:
            raise RuntimeError(
                f"a different top-level `models` package ({mod.__file__}) is already "
                f"imported; cvlface_{variant} needs that name for its snapshot code. "
                "Load it in a fresh process."
            )


def _refuse_build(cmd, *args, **kwargs):
    raise subprocess.CalledProcessError(1, cmd)


class CVLfaceEmbedder(Embedder):
    """CVLface AdaFace models (WebFace12M): IR-101 and ViT-B with KP-RPE.

    The Hugging Face snapshot's own ``wrapper.py`` builds the network (it reads
    ``pretrained_model/*`` relative to the snapshot, hence the temporary change of
    working directory). ViT-B is fed the fixed ArcFace template
    (:data:`ARCFACE_TEMPLATE_112` / 112) as keypoints for every crop; the snapshot's
    face aligner is not used, because all inputs are already aligned to that
    template.

    ViT-B uses the optional compiled ``rpe_index_cpp`` op when it is importable and
    otherwise the snapshot's pure-PyTorch gather, which gives the same numbers. The
    snapshot tries to compile and install the op on first import and then calls
    ``sys.exit()``; the loader disables the ``sys.exit`` and, unless
    ``build_rpe_ops=True``, also the compile attempt.

    Parameters
    ----------
    snapshot
        Folder with the snapshot files.
    variant
        ``ir101`` or ``vit_b``.
    name
        Registry name.
    device
        ``cpu``, ``cuda`` or ``cuda:<index>``.
    build_rpe_ops
        Let the ViT-B snapshot try to compile ``rpe_index_cpp`` if it is missing.
    """

    family = "cvlface"

    def __init__(
        self,
        snapshot: str | Path,
        *,
        variant: str,
        name: str,
        device: str,
        build_rpe_ops: bool = False,
    ):
        import torch  # noqa: PLC0415

        snap = os.path.abspath(str(snapshot))
        _claim_cvlface(variant, snap)
        self.name = name
        self.device = device
        self.variant = variant
        self.snapshot = Path(snap)
        self.uses_keypoints = variant == "vit_b"
        mod_name = f"_face1kb_cvlface_{variant}_wrapper"
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(snap, "wrapper.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        cwd = os.getcwd()
        added = snap not in sys.path
        if added:
            sys.path.insert(0, snap)  # wrapper.py does `from models import get_model`
        real_exit = sys.exit
        real_check_call = subprocess.check_call
        buf = io.StringIO()
        sys.exit = lambda *a, **k: None  # type: ignore[assignment]
        if not build_rpe_ops:
            subprocess.check_call = _refuse_build  # type: ignore[assignment]
        try:
            with contextlib.redirect_stdout(buf):
                os.chdir(snap)
                spec.loader.exec_module(mod)
                if self.uses_keypoints:
                    # Import the KP-RPE code first: when rpe_index_cpp is missing its
                    # build attempt changes the working directory without restoring
                    # it, which would break the relative weight paths below.
                    importlib.import_module("models.vit_kprpe")
                    os.chdir(snap)
                model = mod.CVLFaceRecognitionModel(mod.ModelConfig())
        except Exception:
            logger.error("CVLface snapshot output:\n%s", buf.getvalue())
            raise
        finally:
            sys.exit = real_exit
            subprocess.check_call = real_check_call
            os.chdir(cwd)
            if added and snap in sys.path:
                sys.path.remove(snap)
        logger.debug("CVLface snapshot output:\n%s", buf.getvalue())
        _CVLFACE_LOADED[variant] = snap
        self.rpe_op = None
        if self.uses_keypoints:
            kp = sys.modules.get("models.vit_kprpe.RPE.KPRPE.kprpe_shared")
            self.rpe_op = getattr(kp, "RPEIndexFunction", None) is not None
            if not self.rpe_op:
                logger.info(
                    "%s: rpe_index_cpp not available, using the pure-PyTorch "
                    "KP-RPE gather (same numbers, slower)",
                    name,
                )
        self.torch = torch
        self.model = model.to(device).eval()
        self.kps = (
            torch.tensor([list(p) for p in ARCFACE_TEMPLATE_112], device=device) / 112.0
        ).unsqueeze(0)

    def _embed(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        t = torch.from_numpy(x.astype(np.float32)).permute(0, 3, 1, 2)
        t = (((t / 255.0) - 0.5) / 0.5).contiguous().to(self.device)
        with torch.no_grad():
            if self.uses_keypoints:
                e = self.model(t, self.kps.expand(t.shape[0], -1, -1))
            else:
                e = self.model(t)
        e = e[0] if isinstance(e, (tuple, list)) else e
        return e.float().cpu().numpy()


def load_cvlface(
    name: str,
    variant: str,
    device: str,
    download: bool,
    verify: bool,
    build_rpe_ops: bool = False,
) -> CVLfaceEmbedder:
    """CVLface AdaFace IR-101 or ViT-B + KP-RPE (WebFace12M)."""
    _ensure_all(name, sources.cvlface_files(variant), download, verify)
    snap = models_root() / sources.cvlface_dir(variant)
    return CVLfaceEmbedder(
        snap, variant=variant, name=name, device=device, build_rpe_ops=build_rpe_ops
    )


# --------------------------------------------------------------------- torch nets
class _TorchNetEmbedder(Embedder):
    """EdgeFace / TopoFR: RGB, ``(x / 255 - 0.5) / 0.5``, NCHW, one forward pass."""

    def __init__(self, model, *, name: str, family: str, device: str, **fwd):
        import torch  # noqa: PLC0415

        self.torch = torch
        self.name = name
        self.family = family
        self.device = device
        self.forward_kwargs = fwd
        self.model = model.to(device).eval()

    def _embed(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        # The input stays a permuted (channels-last) view of the NHWC batch: the
        # memory layout selects the cuDNN kernels, and this is the layout that
        # reproduces the paper's embedding arrays bit for bit.
        t = torch.from_numpy(x.astype(np.float32)).permute(0, 3, 1, 2)
        t = (((t / 255.0) - 0.5) / 0.5).to(self.device)
        with torch.no_grad():
            return self.model(t, **self.forward_kwargs).float().cpu().numpy()


def build_edgeface_net(name: str, download: bool = True, verify: bool = True):
    """EdgeFace network with its official weights (``nn.Module``, CPU, eval mode).

    The architecture comes from :func:`face1kb.codec.identity_loss.build_edgeface`
    (the vendored EdgeFace code); the weight file is the one shared with the codec
    training, located, downloaded and verified like every other evaluator file and
    loaded with ``strict=True``.
    """
    import torch  # noqa: PLC0415

    from face1kb.codec.identity_loss import (  # noqa: PLC0415
        EDGEFACE_ARCHS,
        build_edgeface,
    )

    file = sources.edgeface_file(name)
    path = ensure_file(file, download=download, verify=verify, model=name)
    model = build_edgeface(EDGEFACE_ARCHS[name], pretrained=False)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.eval()


def load_edgeface(
    name: str, variant: str, device: str, download: bool, verify: bool
) -> _TorchNetEmbedder:
    """EdgeFace XXS/XS/S/Base (EdgeNeXt hybrid, WebFace260M subsets)."""
    net = build_edgeface_net(name, download=download, verify=verify)
    return _TorchNetEmbedder(net, name=name, family="edgeface", device=device)


#: Module name under which the fetched TopoFR ``backbones`` package is imported.
#: EdgeFace's upstream code uses the same top-level name ``backbones``, so TopoFR's
#: copy must not claim it.
TOPOFR_MODULE = "_face1kb_topofr_backbones"


def _topofr_module(download: bool, verify: bool):
    code_dir = (models_root() / sources.TOPOFR_CODE_DIR).absolute()
    mod = sys.modules.get(TOPOFR_MODULE)
    if mod is not None:
        if Path(mod.__file__).parent != code_dir:
            raise RuntimeError(
                f"{TOPOFR_MODULE} is already imported from {mod.__file__}, not from "
                f"{code_dir}"
            )
        return mod
    _ensure_all("topofr", sources.TOPOFR_CODE_FILES, download, verify)
    spec = importlib.util.spec_from_file_location(
        TOPOFR_MODULE,
        code_dir / "__init__.py",
        submodule_search_locations=[str(code_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[TOPOFR_MODULE] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        del sys.modules[TOPOFR_MODULE]
        raise
    return mod


#: Number of Glint360K identities (rows of the classifier stored in the TopoFR
#: checkpoints); needed to build the network without its checkpoint.
TOPOFR_NUM_CLASSES = 360232


def build_topofr_net(
    variant: str, pretrained: bool = True, download: bool = True, verify: bool = True
):
    """TopoFR IResNet (``r50``, ``r100``, ``r200``) as an ``nn.Module`` on the CPU.

    With ``pretrained=True`` the Glint360K checkpoint is loaded with
    ``strict=True``; the classifier size is read from the checkpoint.
    """
    import torch  # noqa: PLC0415

    mod = _topofr_module(download, verify)
    if not pretrained:
        return mod.get_model(variant, fp16=False, num_classes=TOPOFR_NUM_CLASSES)
    name = f"topofr_{variant}"
    (path,) = _ensure_all(name, (sources.topofr_weights(variant),), download, verify)
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = mod.get_model(variant, fp16=False, num_classes=ckpt["weight"].shape[0])
    model.load_state_dict(ckpt, strict=True)
    return model.eval()


def load_topofr(
    name: str, variant: str, device: str, download: bool, verify: bool
) -> _TorchNetEmbedder:
    """TopoFR R50/R100/R200 (IResNet, Glint360K)."""
    net = build_topofr_net(variant, pretrained=True, download=download, verify=verify)
    return _TorchNetEmbedder(
        net, name=name, family="topofr", device=device, phase="infer"
    )
