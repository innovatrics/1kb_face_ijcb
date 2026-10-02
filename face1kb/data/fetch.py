# SPDX-License-Identifier: MIT
"""Download and checksum helpers for the third-party models of the data layer.

Models are fetched from their official sources into
:func:`face1kb.config.models_dir` on first use and verified by SHA-256:

* MediaPipe selfie segmenter (Apache-2.0), used by the ``B2`` preprocessing
  operator;
* insightface ``buffalo_l`` model pack (insightface model zoo; the pretrained
  models are for non-commercial research use only), whose ``genderage`` model
  estimates age and gender of the AI-Solutions-KK crops;
* BiRefNet (MIT) is loaded through ``transformers`` from the Hugging Face Hub at a
  pinned revision (:data:`BIREFNET_REVISION`), see
  :func:`face1kb.data.preprocess.load_birefnet`.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from face1kb import config

log = logging.getLogger(__name__)

#: Official URL of the MediaPipe selfie segmenter (float16, version 1).
SELFIE_SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter/float16/1/selfie_segmenter.tflite"
)
#: SHA-256 of :data:`SELFIE_SEGMENTER_URL`.
SELFIE_SEGMENTER_SHA256 = (
    "191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b"
)
#: Official insightface ``buffalo_l`` model pack.
BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
#: SHA-256 of :data:`BUFFALO_L_URL`.
BUFFALO_L_SHA256 = "80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f"
#: SHA-256 of the models of the ``buffalo_l`` pack, as extracted from
#: :data:`BUFFALO_L_URL`.
BUFFALO_L_MODELS: dict[str, str] = {
    "1k3d68.onnx": "df5c06b8a0c12e422b2ed8947b8869faa4105387f199c477af038aa01f9a45cc",  # noqa: E501
    "2d106det.onnx": "f001b856447c413801ef5c42091ed0cd516fcd21f2d6b79635b1e733a7109dbf",  # noqa: E501
    "det_10g.onnx": "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91",  # noqa: E501
    "genderage.onnx": "4fde69b1c810857b88c64a335084f1c3fe8f01246c9a191b48c7bb756d6652fb",  # noqa: E501
    "w600k_r50.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",  # noqa: E501
}
#: SHA-256 of ``buffalo_l/genderage.onnx``.
GENDERAGE_SHA256 = BUFFALO_L_MODELS["genderage.onnx"]
#: Hugging Face repository and pinned revision of BiRefNet.
BIREFNET_REPO = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _publish(tmp: str, dst: Path) -> None:
    """Move a finished temporary file into place with the default file mode."""
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp, 0o666 & ~umask)  # mkstemp creates files readable by the owner only
    os.replace(tmp, dst)


def download(url: str, dst: str | os.PathLike, sha256: str | None = None) -> Path:
    """Download ``url`` to ``dst`` atomically, verifying ``sha256`` if given."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, suffix=".part")
    os.close(fd)
    try:
        log.info("downloading %s", url)
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310
            shutil.copyfileobj(r, f)
        if sha256 is not None:
            got = sha256_file(tmp)
            if got != sha256:
                raise OSError(f"checksum mismatch for {url}: {got} != {sha256}")
        _publish(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dst


def _verified(path: Path, sha256: str) -> bool:
    if not path.is_file():
        return False
    got = sha256_file(path)
    if got != sha256:
        raise OSError(
            f"{path} has sha256 {got}, expected {sha256}; delete it to re-download"
        )
    return True


def selfie_segmenter_path(download_missing: bool = True) -> Path:
    """Path of the MediaPipe selfie segmenter, downloading it if needed."""
    path = config.models_dir("mediapipe", "selfie_segmenter.tflite")
    if _verified(path, SELFIE_SEGMENTER_SHA256):
        return path
    if not download_missing:
        raise FileNotFoundError(path)
    return download(SELFIE_SEGMENTER_URL, path, SELFIE_SEGMENTER_SHA256)


def insightface_pack_dir(pack: str = "buffalo_l") -> Path:
    """Folder of an insightface model pack: ``<models root>/insightface/<pack>``."""
    return config.models_dir("insightface", pack)


def _extract_buffalo_l(names: list[str]) -> None:
    """Download the ``buffalo_l`` pack and (re)write the models ``names`` from it.

    Each model is extracted to a temporary file, checked against
    :data:`BUFFALO_L_MODELS` and moved into place, so an interrupted extraction
    never leaves a truncated model behind. The archive is deleted afterwards.
    """
    pack = insightface_pack_dir("buffalo_l")
    pack.mkdir(parents=True, exist_ok=True)
    zpath = download(BUFFALO_L_URL, pack.parent / "buffalo_l.zip", BUFFALO_L_SHA256)
    try:
        with zipfile.ZipFile(zpath) as z:
            members = {Path(n).name: n for n in z.namelist() if n.endswith(".onnx")}
            for name in names:
                if name not in members:
                    raise FileNotFoundError(f"{name} missing from {BUFFALO_L_URL}")
                target = pack / name
                fd, tmp = tempfile.mkstemp(dir=pack, suffix=".part")
                os.close(fd)
                try:
                    with z.open(members[name]) as src, open(tmp, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    got = sha256_file(tmp)
                    if got != BUFFALO_L_MODELS[name]:
                        raise OSError(
                            f"{name} from {BUFFALO_L_URL} has sha256 {got}, "
                            f"expected {BUFFALO_L_MODELS[name]}"
                        )
                    _publish(tmp, target)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
    finally:
        zpath.unlink(missing_ok=True)


def insightface_model_path(
    name: str, pack: str = "buffalo_l", download_missing: bool = True
) -> Path:
    """Path of one verified model of an insightface pack, fetching it if needed.

    Parameters
    ----------
    name : str
        Model file of the pack, e.g. ``genderage.onnx`` or ``2d106det.onnx``
        (see :data:`BUFFALO_L_MODELS`).
    pack : str
        Model pack; only ``buffalo_l`` is pinned.
    download_missing : bool
        Fetch the pack when the model is missing or fails its checksum (the model
        is then re-extracted from the pinned archive). Otherwise a missing model
        raises :class:`FileNotFoundError` and a corrupt one :class:`OSError`.

    Returns
    -------
    Path
        ``<models root>/insightface/<pack>/<name>``, with a verified SHA-256.
    """
    if pack != "buffalo_l" or name not in BUFFALO_L_MODELS:
        raise ValueError(f"no pinned insightface model {pack}/{name}")
    path = insightface_pack_dir(pack) / name
    try:
        if _verified(path, BUFFALO_L_MODELS[name]):
            return path
    except OSError:
        if not download_missing:
            raise
        log.warning("%s fails its checksum; re-extracting it", path)
    if not download_missing:
        raise FileNotFoundError(path)
    # extract every missing or corrupt model of the pack from one download
    todo = []
    for other, sha in BUFFALO_L_MODELS.items():
        p = insightface_pack_dir(pack) / other
        if other == name or not p.is_file() or sha256_file(p) != sha:
            todo.append(other)
    _extract_buffalo_l(todo)
    if not _verified(path, BUFFALO_L_MODELS[name]):
        raise FileNotFoundError(path)
    return path


def genderage_path(download_missing: bool = True) -> Path:
    """Path of insightface ``buffalo_l/genderage.onnx``, fetching the pack if needed.

    See :func:`insightface_model_path`; when the pack is downloaded, all its models
    are extracted, but other models (e.g. the ``2d106det`` landmark model) should
    be requested with :func:`insightface_model_path` rather than assumed present.
    """
    return insightface_model_path("genderage.onnx", "buffalo_l", download_missing)
