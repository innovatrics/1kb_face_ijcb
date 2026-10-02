# SPDX-License-Identifier: MIT
"""Official sources of the evaluator weights, pinned by revision and SHA-256.

No third-party weight or code file is redistributed with face1kb. Every file an
evaluator needs is downloaded from its official location into
``FACE1KB_MODELS_ROOT`` (see :mod:`face1kb.config`) and verified against the
SHA-256 recorded here:

========================  ====================================================
``lvface/``               LVFace ONNX models (Hugging Face
                          ``bytedance-research/LVFace``)
``insightface/``          ``glintr100.onnx`` extracted from the insightface
                          ``antelopev2.zip`` release asset
``cvlface/<repo>/``       the files of the CVLface Hugging Face snapshots that the
                          evaluator executes and loads (``wrapper.py``, ``models/``,
                          ``pretrained_model/``)
``edgeface/``             EdgeFace checkpoints (the specification is owned by
                          :mod:`face1kb.codec.identity_loss`, which the codecs use)
``topofr/``               TopoFR backbone code (GitHub, pinned commit) and the
                          Glint360K checkpoints (Google Drive)
========================  ====================================================

The weights are released by their authors under their own terms, mostly for
non-commercial research only; :attr:`ModelSource.notice` summarises them and
``docs/models.md`` lists them in full.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache


@dataclass(frozen=True)
class Archive:
    """A downloadable archive that contains one or more model files.

    Attributes
    ----------
    urls
        Download URLs, tried in order.
    sha256
        Expected SHA-256 of the archive.
    size
        Size of the archive in bytes.
    """

    urls: tuple[str, ...]
    sha256: str
    size: int


@dataclass(frozen=True)
class RemoteFile:
    """One file under ``FACE1KB_MODELS_ROOT`` and where it comes from.

    Exactly one of ``urls``, ``archive`` (with ``member``) or ``gdrive_id`` names the
    source.

    Attributes
    ----------
    path
        POSIX path relative to ``FACE1KB_MODELS_ROOT``.
    sha256
        Expected SHA-256 of the file.
    size
        Size in bytes.
    urls
        Direct download URLs, tried in order.
    archive, member
        Zip archive that contains the file, and the member name inside it.
    gdrive_id
        Google Drive file id (downloaded with ``gdown``; a manual download is the
        documented fallback).
    manual_url
        Web page from which the file can be downloaded by hand.
    """

    path: str
    sha256: str
    size: int
    urls: tuple[str, ...] = ()
    archive: Archive | None = None
    member: str | None = None
    gdrive_id: str | None = None
    manual_url: str | None = None


@dataclass(frozen=True)
class ModelSource:
    """Provenance and licence terms of one evaluator.

    Attributes
    ----------
    name
        Registry name (e.g. ``lvface_l``).
    files
        Files the evaluator needs.
    homepage
        Official project page.
    weights_licence
        Licence / terms of use of the pretrained weights, as stated by the authors.
    code_licence
        Licence of the upstream code.
    training_data
        Training set of the weights.
    remarks
        Further remarks (e.g. manual-download instructions).
    extra
        Internal bookkeeping (not part of the provenance record).
    """

    name: str
    files: tuple[RemoteFile, ...]
    homepage: str
    weights_licence: str
    code_licence: str
    training_data: str
    remarks: str = ""
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def size(self) -> int:
        """Total size of the model files in bytes."""
        return sum(f.size for f in self.files)

    @property
    def notice(self) -> str:
        """One-paragraph licence notice printed before the files are fetched."""
        return (
            f"{self.name}: weights from {self.homepage} -- {self.weights_licence} "
            f"(code: {self.code_licence}; trained on {self.training_data}). The "
            "files are downloaded from the official source and are not part of "
            "face1kb; you are responsible for complying with their terms."
        )


# ----------------------------------------------------------------------- LVFace
LVFACE_REPO = "bytedance-research/LVFace"
LVFACE_REVISION = "b12702ab1f5c721748e054a66dc90e1edd1f0724"
_LVFACE = {
    "t": ("bf8da0e1e93c432d9a1d874a9ba0990f5859f970e8864b3990f2f33d11f9cdb3", 76653813),
    "s": (
        "cd09f27c82ce0a3633fb8b1966d779a7171b23aa4f14ca0de6edf9677573d119",
        304196926,
    ),
    "b": (
        "9d834ed8e927fd35b9123b2bf97c40aad05785b1f9ecfb1c4c1f6242d38d1382",
        455533594,
    ),
    "l": (
        "49389036a4a5b69e0efcddfe34839ac72c7a71ce6b4dc1b6821e2ac368c87063",
        1022938188,
    ),
}


def _hf_url(repo: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{path}"


def lvface_file(variant: str) -> RemoteFile:
    """ONNX model of LVFace-``variant`` (``t``, ``s``, ``b`` or ``l``)."""
    sha, size = _LVFACE[variant]
    stem = f"LVFace-{variant.upper()}_Glint360K"
    return RemoteFile(
        path=f"lvface/{stem}.onnx",
        sha256=sha,
        size=size,
        urls=(_hf_url(LVFACE_REPO, LVFACE_REVISION, f"{stem}/{stem}.onnx"),),
    )


# ------------------------------------------------------------ insightface ArcFace
ANTELOPEV2_ZIP = Archive(
    urls=(
        "https://github.com/deepinsight/insightface/releases/download/v0.7/"
        "antelopev2.zip",
    ),
    sha256="8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5",
    size=360662982,
)
GLINTR100 = RemoteFile(
    path="insightface/antelopev2/glintr100.onnx",
    sha256="4ab1d6435d639628a6f3e5008dd4f929edf4c4124b1a7169e1048f9fef534cdf",
    size=260665334,
    archive=ANTELOPEV2_ZIP,
    member="antelopev2/glintr100.onnx",
)

# ---------------------------------------------------------------------- CVLface
CVLFACE_REPOS = {
    "ir101": (
        "minchul/cvlface_adaface_ir101_webface12m",
        "54f602a0737bd1ee4a4e7e9fd089a485f397fefd",
    ),
    "vit_b": (
        "minchul/cvlface_adaface_vit_base_kprpe_webface12m",
        "daefd5012d369588bd214fbaf4cc6b1d286e7066",
    ),
}
# The files of each snapshot that the evaluator executes or reads. The transformers
# copy of the weights (model.safetensors) and the snapshot's own face aligner
# (pretrained_model/aligner.*) are not used and not downloaded.
_CVLFACE_BASE = (
    (
        "models/__init__.py",
        "dda1747063e80c8309221e0169d58d0ce75e3b7ca35c97c511688ca7b4645975",
        2002,
    ),
    (
        "models/base/__init__.py",
        "2d038b0a87d752b1ca646b5fcad78f756133d1eaf83e359b0cbe9114e834eeb8",
        5390,
    ),
    (
        "models/base/configs/example.yaml",
        "1895b0bb4e4461ab2e6e5cfc9f6b884c99667f283e135c1d21699af2fb1879f7",
        98,
    ),
    (
        "models/base/utils.py",
        "eb31982bdc79a059f088e43d1410b000cd9ca7fe802086c80b863ad1e4a51ead",
        3122,
    ),
    (
        "wrapper.py",
        "8676246c47090203367d757bd610ced89e002a59eaef29cd4be39e19b7d5181e",
        766,
    ),
)
_CVLFACE_FILES = {
    "ir101": _CVLFACE_BASE
    + (
        (
            "README.md",
            "8968dda62e80dc9036ec32cb34f9be166e3e183323188535f2d3e0fdfc83cfcb",
            2780,
        ),
        (
            "models/iresnet/__init__.py",
            "f87644c32110878c67e91dcfd82e2af394a019c204a17bf80511806044ac77f1",
            1794,
        ),
        (
            "models/iresnet/configs/v1_ir101.yaml",
            "0d771ed0c2a399bd5207a070e11a6bf9d3d941077de94b472b5d3376cccbcf7a",
            103,
        ),
        (
            "models/iresnet/configs/v1_ir18.yaml",
            "5e00c3002546cdb363c3d01bc2ed7b6e8b261afcd5a67e6a876ff4af5fb8fbbc",
            102,
        ),
        (
            "models/iresnet/configs/v1_ir50.yaml",
            "a3c4ef1133e56e5248c8303ec4640ffb8e09894ba95a5c23508426763456e0fd",
            102,
        ),
        (
            "models/iresnet/model.py",
            "e31be55c60b538c15151887e911b7535e2f0e1114a13427ba06483e1cd2a63f9",
            10912,
        ),
        (
            "pretrained_model/config.yaml",
            "de70ea65dbb7d0149f9e5e38476a029362fb474b78168225a2b94b6af36363e6",
            54,
        ),
        (
            "pretrained_model/model.pt",
            "e312d79222d28027f146ed495e182e48fe0dddf404cdbbdabcccdbdd07cc3758",
            261111273,
        ),
        (
            "pretrained_model/model.yaml",
            "29b9eb57d3e6364421b34312ec1cb44cfd3e2b7a3e4fb06c72d322ddeb338dae",
            150,
        ),
    ),
    "vit_b": _CVLFACE_BASE
    + (
        (
            "README.md",
            "98f50124e32bb63ed05e47ea80fb5ac1d20961bc7ea445765be7a8c6a7858b74",
            3102,
        ),
        (
            "models/vit_kprpe/RPE/KPRPE/dist.py",
            "90776a2f2b6a77608229b456a81271fccff93df8aa7566750aea25ff72dab4b6",
            4435,
        ),
        (
            "models/vit_kprpe/RPE/KPRPE/kprpe_shared.py",
            "30d374482c3921c6a90abcad4ea964a394b8a91de51283401f90fecf52361ce4",
            25726,
        ),
        (
            "models/vit_kprpe/RPE/KPRPE/relative_keypoints.py",
            "02f77f9ee001c4cea2e071d430e70ce5f9e7617c54258162a34a92b7f070bcf2",
            682,
        ),
        (
            "models/vit_kprpe/RPE/__init__.py",
            "d10b8a67955ec765dc9c98f0b265317389a591d58653896cba8b98c3b38a20be",
            1804,
        ),
        (
            "models/vit_kprpe/RPE/rpe_ops/README.md",
            "d68ccce3b882e9c632ca110226d1ed15fe93809e3fe65770113cf9b5e1a5b65f",
            86,
        ),
        (
            "models/vit_kprpe/RPE/rpe_ops/rpe_index.cpp",
            "ce3ed63af3de83bb090b04b252b89421029d31d232270a170a19f9b57c5f6bc3",
            5593,
        ),
        (
            "models/vit_kprpe/RPE/rpe_ops/rpe_index.py",
            "b3aafeff4ca2106f4131ed98074b69cd0cf4f43f0351663acd9db8180709eece",
            3214,
        ),
        (
            "models/vit_kprpe/RPE/rpe_ops/rpe_index_cuda.cu",
            "e49731a6cb766e1b1df1ecca67e0c41559f8e6fe66f31c5dc55d7ec6d9c569f3",
            5422,
        ),
        (
            "models/vit_kprpe/RPE/rpe_ops/setup.py",
            "5fba54fc1062920aeb80d1cea92f30ed80b39ecb612bc30a91435310f005a152",
            758,
        ),
        (
            "models/vit_kprpe/__init__.py",
            "8af029e06238bc983d50accc67c34f85f247d97493af9481acf029df2692f950",
            2229,
        ),
        (
            "models/vit_kprpe/configs/v1_base_kprpe_splithead_unshared.yaml",
            "e960265141b2f4d5189534a7fa7698c45aad81949913d697637052c49528267f",
            292,
        ),
        (
            "models/vit_kprpe/configs/v1_small_kprpe_splithead_unshared.yaml",
            "9c34b6e62b59f16cbccde958ea33ee89bf25e0f6d9d7f29b834f7a2edcc0f911",
            293,
        ),
        (
            "models/vit_kprpe/rpe_options.py",
            "89b237db33d51039a4ffa005ff5e3e30a76b20c1c135af39b7c62a1cd9fa0ace",
            5556,
        ),
        (
            "models/vit_kprpe/vit.py",
            "43d32e885efdf071ac84965548b6ae05abd60960d21d25ea9bf36f467d0aef65",
            11677,
        ),
        (
            "pretrained_model/config.yaml",
            "a8dc49d9bb9ce79d1f9020230e09e57c13ef5c325ed0ac737f78cdc833556f02",
            71,
        ),
        (
            "pretrained_model/model.pt",
            "04b4bee1de7cefa9e97900f8449fca906d8afbab2029bd39cc5049d33e927ed9",
            460381841,
        ),
        (
            "pretrained_model/model.yaml",
            "67b9a4d1a873311cf0d70811ad70d608387225c005c0fa167f4622c0db82d697",
            366,
        ),
    ),
}


def cvlface_dir(variant: str) -> str:
    """Snapshot folder of a CVLface variant, relative to ``FACE1KB_MODELS_ROOT``."""
    repo, _ = CVLFACE_REPOS[variant]
    return f"cvlface/{repo.split('/', 1)[1]}"


def cvlface_files(variant: str) -> tuple[RemoteFile, ...]:
    """Pinned snapshot files of a CVLface variant (``ir101`` or ``vit_b``)."""
    repo, rev = CVLFACE_REPOS[variant]
    root = cvlface_dir(variant)
    return tuple(
        RemoteFile(
            path=f"{root}/{rel}",
            sha256=sha,
            size=size,
            urls=(_hf_url(repo, rev, rel),),
        )
        for rel, sha, size in sorted(_CVLFACE_FILES[variant])
    )


# ----------------------------------------------------------------------- TopoFR
TOPOFR_REPO = "DanJun6737/TopoFR"
TOPOFR_COMMIT = "5960a33220968e6f9f74adcffb3f1a4ca31f9a0a"
#: Folder of the fetched TopoFR ``backbones`` package, relative to MODELS_ROOT.
TOPOFR_CODE_DIR = f"topofr/src-{TOPOFR_COMMIT[:12]}/backbones"
_TOPOFR_CODE = (
    (
        "__init__.py",
        "7a616efdd3f55f963eb590a829035c516ff1756c955a8d1eca405e73e6280efc",
        7741,
    ),
    (
        "iresnet.py",
        "623be381ac59efed84861455295aaacf0749a457f05024fd12b6c40f23bb6bb7",
        8040,
    ),
    (
        "iresnet2060.py",
        "2bb41694c2af1b532ed78ce5474eb90065354baed1851f3cdc76a52a2893daa4",
        6708,
    ),
    (
        "mobilefacenet.py",
        "95a1746ca8c8b88460adf4fb0b113a4a0292481019eb5157df3e1cd9eacb8881",
        5591,
    ),
)
TOPOFR_CODE_FILES = tuple(
    RemoteFile(
        path=f"{TOPOFR_CODE_DIR}/{name}",
        sha256=sha,
        size=size,
        urls=(
            f"https://raw.githubusercontent.com/{TOPOFR_REPO}/{TOPOFR_COMMIT}/"
            f"backbones/{name}",
        ),
    )
    for name, sha, size in _TOPOFR_CODE
)
_TOPOFR_WEIGHTS = {
    "r50": (
        "Glint360K_R50_TopoFR_9727.pt",
        "1R_ffZ2GpvNrwG5ZM76LO32KTol-hXNQx",
        "15f07d919bd0b1443a481fc4e39f0689e773881836da7e959eca9ddb09e31310",
        912424777,
    ),
    "r100": (
        "Glint360K_R100_TopoFR_9760.pt",
        "1vQBGXc_nXytEx8fpV9jykeLdxD45cE8B",
        "f92e5e61b326495d32803156ab60aa58abf7e17f84e83d38b5ff6bf1691bdbfe",
        998958213,
    ),
    "r200": (
        "Glint360K_R200_TopoFR_9784.pt",
        "1DXvcksXIaIXoNWxTXPWhLQaKL_aPaBAR",
        "bf0fda461721273279f39688f834c81d4ec5e4b3465cd8946d324f555b025111",
        1214223369,
    ),
}


def topofr_weights(variant: str) -> RemoteFile:
    """Glint360K checkpoint of TopoFR-``variant`` (``r50``, ``r100``, ``r200``)."""
    filename, gid, sha, size = _TOPOFR_WEIGHTS[variant]
    return RemoteFile(
        path=f"topofr/{filename}",
        sha256=sha,
        size=size,
        gdrive_id=gid,
        manual_url=f"https://drive.google.com/file/d/{gid}/view",
    )


# --------------------------------------------------------------------- EdgeFace
EDGEFACE_REPOS = {
    "edgeface_xxs": "Idiap/EdgeFace-XXS",
    "edgeface_xs": "Idiap/EdgeFace-XS-GAMMA",
    "edgeface_s": "Idiap/EdgeFace-S-GAMMA",
    "edgeface_base": "Idiap/EdgeFace-Base",
}
_EDGEFACE_SIZES = {
    "edgeface_xxs.pt": 5032125,
    "edgeface_xs_gamma_06.pt": 7170425,
    "edgeface_s_gamma_05.pt": 14695737,
    "edgeface_base.pt": 72972261,
}


def edgeface_file(name: str) -> RemoteFile:
    """EdgeFace checkpoint of ``name``, as pinned by the codec's identity loss.

    The specification (file name, SHA-256, URLs) is taken from
    :data:`face1kb.codec.identity_loss.EDGEFACE_WEIGHTS`, so the evaluators and the
    codec training share one copy of the weights.
    """
    from face1kb.codec.identity_loss import (  # noqa: PLC0415
        EDGEFACE_ARCHS,
        EDGEFACE_WEIGHTS,
    )

    spec = EDGEFACE_WEIGHTS[EDGEFACE_ARCHS[name]]
    return RemoteFile(
        path=f"{spec.subdir}/{spec.filename}",
        sha256=spec.sha256,
        size=_EDGEFACE_SIZES.get(spec.filename, 0),
        urls=tuple(spec.urls),
    )


# ------------------------------------------------------------------ the roster
_INSIGHTFACE_TERMS = (
    "insightface pretrained models: non-commercial research purposes only"
)
_GLINT = "Glint360K"


def _build() -> dict[str, ModelSource]:
    out: dict[str, ModelSource] = {}
    for v in ("t", "s", "b", "l"):
        name = f"lvface_{v}"
        out[name] = ModelSource(
            name=name,
            files=(lvface_file(v),),
            homepage=f"https://huggingface.co/{LVFACE_REPO}",
            weights_licence=(
                "non-commercial research purposes only (LVFace README; the Hugging "
                "Face card is tagged MIT, which covers the code)"
            ),
            code_licence="MIT",
            training_data=_GLINT,
        )
    out["arcface_antelopev2"] = ModelSource(
        name="arcface_antelopev2",
        files=(GLINTR100,),
        homepage="https://github.com/deepinsight/insightface (release v0.7)",
        weights_licence=_INSIGHTFACE_TERMS,
        code_licence="MIT",
        training_data=_GLINT,
        remarks=(
            "Only glintr100.onnx (ResNet-100 ArcFace) is kept from antelopev2.zip; "
            "the archive is deleted after extraction."
        ),
    )
    for v, label in (("ir101", "IR-101"), ("vit_b", "ViT-B + KP-RPE")):
        name = f"cvlface_{v}"
        repo, _ = CVLFACE_REPOS[v]
        out[name] = ModelSource(
            name=name,
            files=cvlface_files(v),
            homepage=f"https://huggingface.co/{repo}",
            weights_licence=(
                "no explicit licence; the model card asks users to follow the licence "
                "of the training data (WebFace260M terms: non-commercial research "
                "only)"
            ),
            code_licence="MIT (github.com/mk-minchul/CVLface)",
            training_data="WebFace12M",
            remarks=f"AdaFace {label}; the snapshot's code is executed at load time.",
        )
    for name, repo in EDGEFACE_REPOS.items():
        out[name] = ModelSource(
            name=name,
            files=(),  # resolved lazily from face1kb.codec.identity_loss
            homepage=f"https://huggingface.co/{repo}",
            weights_licence="CC BY-NC-SA 4.0 (Idiap Research Institute)",
            code_licence="BSD-3-Clause (vendored in face1kb.third_party.edgeface)",
            training_data="WebFace260M (WebFace12M/4M subsets)",
            extra={"edgeface": name},
        )
    for v in ("r50", "r100", "r200"):
        name = f"topofr_{v}"
        out[name] = ModelSource(
            name=name,
            files=TOPOFR_CODE_FILES + (topofr_weights(v),),
            homepage=f"https://github.com/{TOPOFR_REPO}",
            weights_licence=(
                "no licence stated by the authors (training data: research use only)"
            ),
            code_licence=(
                "no licence file; the four backbones/*.py files are fetched from "
                f"commit {TOPOFR_COMMIT[:12]}, never redistributed"
            ),
            training_data=_GLINT,
            remarks=(
                "The checkpoint is hosted on Google Drive and fetched with gdown; if "
                "that fails, download it by hand (see the printed instructions)."
            ),
        )
    return out


_SOURCES = _build()


@cache
def model_source(name: str) -> ModelSource:
    """Return the :class:`ModelSource` of a built-in evaluator.

    Raises
    ------
    KeyError
        If ``name`` is not one of the 14 built-in evaluators.
    """
    src = _SOURCES[name]
    if "edgeface" in src.extra:
        src = ModelSource(
            name=src.name,
            files=(edgeface_file(src.extra["edgeface"]),),
            homepage=src.homepage,
            weights_licence=src.weights_licence,
            code_licence=src.code_licence,
            training_data=src.training_data,
            remarks=src.remarks,
            extra=dict(src.extra),
        )
    return src


def builtin_names() -> tuple[str, ...]:
    """Names of the built-in evaluators that have a pinned source."""
    return tuple(_SOURCES)
