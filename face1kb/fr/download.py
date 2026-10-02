# SPDX-License-Identifier: MIT
"""Download and verify the evaluator files under ``FACE1KB_MODELS_ROOT``.

Every file of :mod:`face1kb.fr.sources` is written to a temporary file next to its
destination, checked against its pinned SHA-256 and only then moved into place, so
an interrupted or corrupted download never leaves a file that looks valid. Files
that cannot be downloaded automatically (Google Drive quota, no ``gdown``) raise
:class:`ManualDownloadRequired`, whose message says where to get the file and
where to put it.
"""

from __future__ import annotations

import glob
import hashlib
import logging
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from .sources import Archive, ModelSource, RemoteFile, model_source

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20
_USER_AGENT = "face1kb-fetch/2.0 (+https://github.com/innovatrics/1kb_face_ijcb)"
# (absolute path, sha256) -> (size, mtime_ns) of files verified in this process.
_VERIFIED: dict[tuple[str, str], tuple[int, int]] = {}


class ManualDownloadRequired(RuntimeError):
    """A file could not be downloaded automatically and must be fetched by hand.

    Parameters
    ----------
    file
        The file that is missing.
    dst
        Where the file has to be saved.
    reason
        Why the automatic download failed.
    """

    def __init__(self, file: RemoteFile, dst: Path, reason: str):
        self.file = file
        self.dst = dst
        self.reason = reason
        where = file.manual_url or (file.urls[0] if file.urls else "its source")
        super().__init__(
            f"could not download {file.path} automatically ({reason}).\n"
            f"  Download it by hand from {where}\n"
            f"  and save it (keeping its name) as {dst}\n"
            f"  expected: {file.size:,} bytes, sha256 {file.sha256}\n"
            "  Then run the fetch again to verify it."
        )


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(_CHUNK):
            h.update(block)
    return h.hexdigest()


def models_root(root: str | Path | None = None) -> Path:
    """``root`` if given, else ``FACE1KB_MODELS_ROOT`` (:data:`config.MODELS_ROOT`)."""
    if root is not None:
        return Path(root)
    from face1kb import config  # noqa: PLC0415

    return config.MODELS_ROOT


def local_path(file: RemoteFile, root: str | Path | None = None) -> Path:
    """Destination of ``file`` under the models root."""
    return models_root(root) / file.path


def check_file(file: RemoteFile, root: str | Path | None = None) -> str | None:
    """Verify one file; return a description of the problem, or ``None`` if valid."""
    dst = local_path(file, root)
    if not dst.is_file():
        return f"{dst}: missing"
    st = dst.stat()
    key = (str(dst.absolute()), file.sha256)
    if _VERIFIED.get(key) == (st.st_size, st.st_mtime_ns):
        return None
    if file.size and st.st_size != file.size:
        return f"{dst}: {st.st_size} bytes, expected {file.size}"
    got = sha256_file(dst)
    if got != file.sha256:
        return f"{dst}: sha256 {got}, expected {file.sha256}"
    _VERIFIED[key] = (st.st_size, st.st_mtime_ns)
    return None


def _publish(tmp: Path, dst: Path) -> None:
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp, 0o666 & ~umask)  # mkstemp creates files with mode 0600
    tmp.replace(dst)


def _tempfile(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dst.parent, prefix=f".{dst.name}.", suffix=".part")
    os.close(fd)
    return Path(name)


def _progress(total: int, desc: str):
    try:
        from tqdm import tqdm  # noqa: PLC0415
    except ImportError:  # pragma: no cover - tqdm is a core dependency
        return None
    return tqdm(
        total=total or None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
        leave=False,
    )


def _download_to(url: str, tmp: Path, size: int, desc: str) -> str:
    """Stream ``url`` into ``tmp``; return the SHA-256 of what was written."""
    h = hashlib.sha256()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    bar = _progress(size, desc)
    try:
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while block := r.read(_CHUNK):
                f.write(block)
                h.update(block)
                if bar is not None:
                    bar.update(len(block))
    finally:
        if bar is not None:
            bar.close()
    return h.hexdigest()


def _fetch_urls(
    urls: tuple[str, ...], dst: Path, sha256: str, size: int, desc: str
) -> Path:
    errors = []
    for url in urls:
        logger.info("downloading %s", url)
        tmp = _tempfile(dst)
        try:
            got = _download_to(url, tmp, size, desc)
            if got != sha256:
                errors.append(f"{url}: sha256 {got} != {sha256}")
                continue
            _publish(tmp, dst)
            return dst
        except OSError as e:  # URLError and HTTPError are OSErrors
            errors.append(f"{url}: {e}")
        finally:
            tmp.unlink(missing_ok=True)
    raise RuntimeError(f"could not fetch {dst.name}: " + "; ".join(errors))


def _fetch_archive_member(file: RemoteFile, dst: Path) -> Path:
    archive: Archive = file.archive  # type: ignore[assignment]
    tmp_zip = _tempfile(dst.with_name(dst.name + ".zip"))
    try:
        archive_name = Path(archive.urls[0].rsplit("/", 1)[-1]).name
        _fetch_urls(archive.urls, tmp_zip, archive.sha256, archive.size, archive_name)
        tmp = _tempfile(dst)
        try:
            h = hashlib.sha256()
            with zipfile.ZipFile(tmp_zip) as z, z.open(file.member) as src:
                with open(tmp, "wb") as out:
                    while block := src.read(_CHUNK):
                        out.write(block)
                        h.update(block)
            if h.hexdigest() != file.sha256:
                raise RuntimeError(
                    f"{file.member} in the archive has sha256 {h.hexdigest()}, "
                    f"expected {file.sha256}"
                )
            _publish(tmp, dst)
        finally:
            tmp.unlink(missing_ok=True)
    finally:
        tmp_zip.unlink(missing_ok=True)
    return dst


def _fetch_gdrive(file: RemoteFile, dst: Path) -> Path:
    try:
        import gdown  # noqa: PLC0415
    except ImportError as e:
        raise ManualDownloadRequired(file, dst, "gdown is not installed") from e
    tmp = _tempfile(dst)
    try:
        logger.info("downloading Google Drive file %s", file.gdrive_id)
        try:
            out = gdown.download(id=file.gdrive_id, output=str(tmp), quiet=False)
        except Exception as e:  # noqa: BLE001 - gdown raises many exception types
            raise ManualDownloadRequired(file, dst, f"gdown failed: {e}") from e
        if out is None or not tmp.is_file() or tmp.stat().st_size == 0:
            raise ManualDownloadRequired(file, dst, "gdown returned no file")
        got = sha256_file(tmp)
        if got != file.sha256:
            raise ManualDownloadRequired(
                file, dst, f"the downloaded file has sha256 {got}"
            )
        _publish(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)
        # gdown streams into its own "<output name>*.part" file next to ``tmp`` and
        # leaves it behind when the transfer is cut off.
        for part in tmp.parent.glob(glob.escape(tmp.name) + "*.part"):
            part.unlink(missing_ok=True)
    return dst


def fetch_file(file: RemoteFile, root: str | Path | None = None) -> Path:
    """Download one file (whatever its source) and verify it; return its path.

    An existing valid file is kept. An existing file with the wrong content raises
    ``RuntimeError`` (delete it, or use ``scripts/fetch_models.py --force``).
    """
    dst = local_path(file, root)
    if dst.exists():
        problem = check_file(file, root)
        if problem is not None:
            raise RuntimeError(f"{problem}; delete it to download it again")
        return dst
    if file.urls:
        _fetch_urls(file.urls, dst, file.sha256, file.size, dst.name)
    elif file.archive is not None:
        _fetch_archive_member(file, dst)
    elif file.gdrive_id is not None:
        _fetch_gdrive(file, dst)
    else:  # pragma: no cover - every pinned file has a source
        raise ManualDownloadRequired(file, dst, "no automatic source")
    problem = check_file(file, root)
    if problem is not None:  # pragma: no cover - verified before publishing
        raise RuntimeError(problem)
    return dst


def ensure_file(
    file: RemoteFile,
    root: str | Path | None = None,
    *,
    download: bool = True,
    verify: bool = True,
    model: str = "",
) -> Path:
    """Return the path of a file, downloading it first if needed and allowed.

    Parameters
    ----------
    file
        The pinned file.
    root
        Models root (default ``FACE1KB_MODELS_ROOT``).
    download
        Download a missing file. With ``False`` a missing file raises
        ``FileNotFoundError``.
    verify
        Check the SHA-256 of an existing file (once per process and file).
    model
        Evaluator name, used in error messages.
    """
    dst = local_path(file, root)
    if dst.is_file():
        if verify:
            problem = check_file(file, root)
            if problem is not None:
                raise RuntimeError(
                    f"{problem}; delete it or run "
                    f"`python scripts/fetch_models.py --only {model or '...'} --force`"
                )
        return dst
    if not download:
        raise FileNotFoundError(
            f"{dst} is missing; run `python scripts/fetch_models.py --only "
            f"{model or '<model>'}` (FACE1KB_MODELS_ROOT={models_root(root)})"
        )
    return fetch_file(file, root)


def _edgeface_arch(src: ModelSource) -> str | None:
    name = src.extra.get("edgeface")
    if name is None:
        return None
    from face1kb.codec.identity_loss import EDGEFACE_ARCHS  # noqa: PLC0415

    return EDGEFACE_ARCHS[name]


def fetch(
    name: str, *, models_root: str | Path | None = None, force: bool = False
) -> list[Path]:
    """Download (if needed) and verify every file of a built-in evaluator.

    Parameters
    ----------
    name
        Built-in evaluator name (see :data:`face1kb.fr.ROSTER`).
    models_root
        Destination root (default ``FACE1KB_MODELS_ROOT``).
    force
        Delete files that fail verification and download them again.

    Returns
    -------
    list of Path
        Local paths of the evaluator's files.

    Raises
    ------
    ManualDownloadRequired
        If a file has to be downloaded by hand (e.g. a Google Drive quota).
    """
    src = model_source(name)
    root = models_root
    arch = _edgeface_arch(src)
    paths = []
    for file in src.files:
        dst = local_path(file, root)
        if force and dst.exists() and check_file(file, root) is not None:
            logger.warning("removing %s (failed verification)", dst)
            dst.unlink()
        if arch is not None:
            # One copy of the EdgeFace weights, fetched exactly as the codec does.
            from face1kb.codec.identity_loss import (  # noqa: PLC0415
                EDGEFACE_WEIGHTS,
                fetch_pretrained,
            )

            spec = EDGEFACE_WEIGHTS[arch]
            fetch_pretrained(spec, models_root=Path(root) if root is not None else None)
            problem = check_file(file, root)
            if problem is not None:  # pragma: no cover - fetch_pretrained verified
                raise RuntimeError(problem)
            paths.append(dst)
        else:
            paths.append(fetch_file(file, root))
    return paths


def verify(name: str, *, models_root: str | Path | None = None) -> list[str]:
    """Check every file of a built-in evaluator; return the list of problems."""
    src = model_source(name)
    problems = []
    for file in src.files:
        problem = check_file(file, models_root)
        if problem is not None:
            problems.append(problem)
    return problems


def present(name: str, *, models_root: str | Path | None = None) -> bool:
    """Whether every file of a built-in evaluator exists (no hashing)."""
    return all(local_path(f, models_root).is_file() for f in model_source(name).files)
