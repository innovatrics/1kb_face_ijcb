# SPDX-License-Identifier: MIT
"""Registry of face-recognition evaluators and the user hooks to extend it."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .embedders import (
    INPUT_SIZE,
    Embedder,
    Normalize,
    OnnxEmbedder,
    TorchEmbedder,
    resolve_device,
)

#: ArcFace ResNet-100 (insightface ``antelopev2`` / ``glintr100``), the reference
#: matcher of the paper.
ARCFACE = "arcface_antelopev2"
#: LVFace-L (ViT-L, Glint360K).
LVFACE_L = "lvface_l"
#: TopoFR ResNet-100 (Glint360K).
TOPOFR_R100 = "topofr_r100"
#: EdgeFace-XS (the lightweight probe; also the FAST codec's training loss model).
EDGEFACE_XS = "edgeface_xs"
#: The four anchor matchers of the headline results, in the paper's order.
ANCHORS: tuple[str, ...] = (ARCFACE, LVFACE_L, TOPOFR_R100, EDGEFACE_XS)
#: The held-out verification matcher of the paper's codec evaluation: neither an
#: anchor nor used in codec training (the codecs train against EdgeFace).
HELDOUT = "cvlface_ir101"
#: Default public matcher for identity-cosine (id-cos) scores. The paper's id-cos
#: columns were computed with a proprietary matcher that is not distributed, so they
#: cannot be regenerated exactly; scores from this default differ from them. Plug in
#: your own model with :func:`register_onnx` / :func:`register_torch` and pass its
#: name instead.
IDCOS_DEFAULT = HELDOUT


@dataclass(frozen=True)
class ModelSpec:
    """A registered evaluator.

    Attributes
    ----------
    name
        Registry name.
    family
        ``lvface``, ``arcface``, ``cvlface``, ``edgeface``, ``topofr`` (built-in) or
        ``onnx``, ``torch``, ``custom`` (user-registered).
    label
        Display name used in tables and figures.
    factory
        ``factory(device, download, verify) -> Embedder``.
    builtin
        Whether the model is one of the 14 evaluators of the paper.
    variant
        Family variant (built-in models).
    """

    name: str
    family: str
    label: str
    factory: Callable[[str, bool, bool], Embedder]
    builtin: bool = False
    variant: str | None = None


def _builtin(name: str, family: str, variant: str, label: str) -> ModelSpec:
    def factory(device: str, download: bool, verify: bool) -> Embedder:
        from . import families  # noqa: PLC0415

        loader = getattr(families, f"load_{family}")
        return loader(name, variant, device, download, verify)

    return ModelSpec(name, family, label, factory, builtin=True, variant=variant)


# The 14-model roster in the paper's display order (grouped by family / strength).
_ROSTER = (
    _builtin("arcface_antelopev2", "arcface", "antelopev2", "ArcFace R100"),
    _builtin("topofr_r50", "topofr", "r50", "TopoFR-R50"),
    _builtin("topofr_r100", "topofr", "r100", "TopoFR-R100"),
    _builtin("topofr_r200", "topofr", "r200", "TopoFR-R200"),
    _builtin("lvface_t", "lvface", "t", "LVFace-T"),
    _builtin("lvface_s", "lvface", "s", "LVFace-S"),
    _builtin("lvface_b", "lvface", "b", "LVFace-B"),
    _builtin("lvface_l", "lvface", "l", "LVFace-L"),
    _builtin("cvlface_vit_b", "cvlface", "vit_b", "CVLface ViT-B"),
    _builtin("cvlface_ir101", "cvlface", "ir101", "CVLface IR-101"),
    _builtin("edgeface_xxs", "edgeface", "xxs", "EdgeFace-XXS"),
    _builtin("edgeface_xs", "edgeface", "xs", "EdgeFace-XS"),
    _builtin("edgeface_s", "edgeface", "s", "EdgeFace-S"),
    _builtin("edgeface_base", "edgeface", "base", "EdgeFace-Base"),
)
#: Names of the 14 built-in evaluators, in the paper's display order.
ROSTER: tuple[str, ...] = tuple(s.name for s in _ROSTER)
#: Display names of the built-in evaluators.
MODEL_LABELS: dict[str, str] = {s.name: s.label for s in _ROSTER}
#: All registered evaluators (built-in and user-registered), by name.
REGISTRY: dict[str, ModelSpec] = {s.name: s for s in _ROSTER}


#: Environment variable naming plugin modules that register evaluators.
PLUGINS_ENV = "FACE1KB_FR_PLUGINS"
# Plugin specs imported in this process (each one is imported once).
_PLUGINS_LOADED: set[str] = set()


def _plugin_specs(spec: str | Sequence[str] | None) -> list[str]:
    if spec is None:
        from face1kb import config  # noqa: PLC0415

        spec = getattr(config, "FR_PLUGINS", None)
        if spec is None:
            spec = os.environ.get(PLUGINS_ENV, "")
    if not isinstance(spec, str):  # a sequence of specs
        spec = ",".join(spec)
    return [s.strip() for s in spec.split(",") if s.strip()]


def load_plugins(spec: str | Sequence[str] | None = None) -> list[str]:
    """Import the plugin modules that register your own evaluators.

    A plugin is a Python module that calls :func:`register_onnx`,
    :func:`register_torch` or :func:`register_embedder` when it is imported. Naming
    it in ``FACE1KB_FR_PLUGINS`` makes the models it registers available in every
    process, including worker processes and command-line tools, without changing
    their code: :func:`load` and :func:`available` call this function first.

    Parameters
    ----------
    spec
        Comma-separated module names (``mypkg.matchers``) and/or paths of ``.py``
        files, or a sequence of them. Default: the ``FACE1KB_FR_PLUGINS``
        environment variable.

    Returns
    -------
    list of str
        The plugin specs imported by this call (each spec is imported once per
        process).
    """
    done = []
    for item in _plugin_specs(spec):
        if item in _PLUGINS_LOADED:
            continue
        if item.endswith(".py") or os.sep in item or "/" in item:
            path = Path(item).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"FR plugin {item!r} not found")
            mod_name = f"_face1kb_fr_plugin_{len(_PLUGINS_LOADED)}_{path.stem}"
            mod_spec = importlib.util.spec_from_file_location(mod_name, path)
            if mod_spec is None or mod_spec.loader is None:
                raise ValueError(
                    f"FR plugin {item!r} is not a Python file (expected a .py path "
                    "or a module name)"
                )
            module = importlib.util.module_from_spec(mod_spec)
            sys.modules[mod_name] = module
            try:
                mod_spec.loader.exec_module(module)
            except BaseException:
                del sys.modules[mod_name]
                raise
        else:
            importlib.import_module(item)
        _PLUGINS_LOADED.add(item)
        done.append(item)
    return done


def available() -> list[str]:
    """Sorted names of all registered evaluators (after :func:`load_plugins`)."""
    load_plugins()
    return sorted(REGISTRY)


def load(
    name: str,
    device: str | None = None,
    *,
    download: bool = True,
    verify: bool = True,
) -> Embedder:
    """Instantiate a registered evaluator.

    Parameters
    ----------
    name
        Registry name (see :data:`ROSTER` and :func:`available`).
    device
        ``"cpu"``, ``"cuda"`` or ``"cuda:<index>"``; default ``"cuda"`` when a GPU is
        available, else ``"cpu"``.
    download
        Download missing model files from their official source (built-in models).
        With ``False`` a missing file raises ``FileNotFoundError``.
    verify
        Check the SHA-256 of the model files (once per process and file).

    Returns
    -------
    Embedder
        Object with ``embed(crops) -> (N, D)`` float32 raw embeddings; torch-based
        evaluators also expose the network as ``.model``.
    """
    if name not in REGISTRY:
        load_plugins()
    try:
        spec = REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown FR model {name!r}; available: {available()} (register your own "
            f"with face1kb.fr.register_onnx / register_torch, e.g. in a module named "
            f"in {PLUGINS_ENV})"
        ) from None
    return spec.factory(resolve_device(device), download, verify)


def _add(spec: ModelSpec, overwrite: bool) -> ModelSpec:
    old = REGISTRY.get(spec.name)
    if old is not None:
        if old.builtin:
            raise ValueError(
                f"{spec.name!r} is a built-in evaluator; choose another name"
            )
        if not overwrite:
            raise ValueError(f"{spec.name!r} is already registered (overwrite=True)")
    REGISTRY[spec.name] = spec
    return spec


def _check_sha256(path: Path, sha256: str | None) -> None:
    if sha256 is None:
        return
    from .download import sha256_file  # noqa: PLC0415

    got = sha256_file(path)
    if got != sha256.lower():
        raise RuntimeError(f"{path} has sha256 {got}, expected {sha256}")


def register_onnx(
    name: str,
    path: str | Path,
    *,
    input_name: str | None = None,
    color: str = "RGB",
    normalize: Normalize = "[-1,1]",
    size: int = INPUT_SIZE,
    output_name: str | None = None,
    layout: str = "NCHW",
    sha256: str | None = None,
    label: str | None = None,
    overwrite: bool = False,
) -> ModelSpec:
    """Register your own ONNX face-recognition model as an evaluator.

    After registration, ``face1kb.fr.load(name)`` returns an embedder with the same
    contract as the built-in ones, so the model can be used wherever the
    experiments accept a matcher name.

    Parameters
    ----------
    name
        Registry name (must not clash with a built-in evaluator).
    path
        ONNX model file (checked when the model is loaded).
    input_name
        Model input to feed (default: the only input).
    color
        Channel order the model expects: ``"RGB"`` or ``"BGR"``.
    normalize
        ``"[-1,1]"`` (``(x / 255 - 0.5) / 0.5``), ``"[0,1]"`` (``x / 255``),
        ``"none"`` (0..255), ``(mean, std)`` in pixel units (scalars or per-channel
        triples, in the model's channel order), or a callable on the float32 batch.
    size
        Input side length; crops are resized bilinearly if it is not 112.
    output_name
        Model output holding the embedding (default: the first output).
    layout
        ``"NCHW"`` (default) or ``"NHWC"``.
    sha256
        Optional expected SHA-256 of the file, verified at load time.
    label
        Display name (default: ``name``).
    overwrite
        Replace an earlier user registration of the same name.

    Examples
    --------
    >>> from face1kb import fr
    >>> fr.register_onnx("my_matcher", "/path/to/model.onnx", color="RGB",
    ...                  normalize="[0,1]")           # doctest: +SKIP
    >>> emb = fr.load("my_matcher").embed(crops)      # doctest: +SKIP
    """
    path = Path(path)

    def factory(device: str, download: bool, verify: bool) -> Embedder:
        if not path.is_file():
            raise FileNotFoundError(f"{name}: ONNX model {path} not found")
        if verify:
            _check_sha256(path, sha256)
        return OnnxEmbedder(
            path,
            name=name,
            device=device,
            input_name=input_name,
            output_name=output_name,
            color=color,
            normalize=normalize,
            size=size,
            layout=layout,
            family="onnx",
        )

    return _add(ModelSpec(name, "onnx", label or name, factory), overwrite)


def _takes_positional_arg(factory: Callable[..., object]) -> bool:
    """Whether ``factory`` has a required positional parameter (the device)."""
    try:
        params = inspect.signature(factory).parameters.values()
    except (TypeError, ValueError):  # builtins without a signature
        return False
    kinds = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    return any(p.kind in kinds and p.default is inspect.Parameter.empty for p in params)


def register_torch(
    name: str,
    factory: Callable[..., object],
    *,
    color: str = "RGB",
    normalize: Normalize = "[-1,1]",
    size: int = INPUT_SIZE,
    forward_kwargs: dict | None = None,
    label: str | None = None,
    overwrite: bool = False,
) -> ModelSpec:
    """Register your own torch face-recognition network as an evaluator.

    Parameters
    ----------
    name
        Registry name (must not clash with a built-in evaluator).
    factory
        Callable returning the ``nn.Module`` (e.g. the module class); it is called
        with the device string if it has a required positional parameter, else
        with no arguments. The module is moved to
        the device and set to eval mode. Its forward receives a float32
        ``(N, 3, size, size)`` tensor; a tuple or list output uses its first
        element.
    color, normalize, size
        Preprocessing, as for :func:`register_onnx`.
    forward_kwargs
        Extra keyword arguments of each forward call (e.g. ``{"phase": "infer"}``).
    label
        Display name (default: ``name``).
    overwrite
        Replace an earlier user registration of the same name.
    """
    pass_device = _takes_positional_arg(factory)

    def build(device: str, download: bool, verify: bool) -> Embedder:
        module = factory(device) if pass_device else factory()
        return TorchEmbedder(
            module,
            name=name,
            device=device,
            color=color,
            normalize=normalize,
            size=size,
            forward_kwargs=forward_kwargs,
        )

    return _add(ModelSpec(name, "torch", label or name, build), overwrite)


def register_embedder(
    name: str,
    factory: Callable[[str], object],
    *,
    label: str | None = None,
    overwrite: bool = False,
) -> ModelSpec:
    """Register any object with an ``embed(crops) -> (N, D)`` method.

    ``factory(device)`` must return an object that follows the embedder contract
    (112 x 112 RGB uint8 crops in, float32 raw embeddings out).
    """

    def build(device: str, download: bool, verify: bool):
        emb = factory(device)
        if not callable(getattr(emb, "embed", None)):
            raise TypeError(f"{name}: factory returned {type(emb)} without .embed()")
        return emb

    return _add(ModelSpec(name, "custom", label or name, build), overwrite)


def unregister(name: str) -> None:
    """Remove a user-registered evaluator (built-in ones cannot be removed)."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise KeyError(name)
    if spec.builtin:
        raise ValueError(f"{name!r} is a built-in evaluator")
    del REGISTRY[name]


def label(name: str) -> str:
    """Display name of a registered evaluator."""
    spec = REGISTRY.get(name)
    return spec.label if spec is not None else name


def torch_builder(name: str) -> Callable[[bool], object]:
    """``builder(pretrained) -> nn.Module`` for a built-in EdgeFace or TopoFR model.

    Use it to make TopoFR available to the codec training (identity loss or
    side-stream anchor)::

        from face1kb.codec.identity_loss import register_torch_fr
        register_torch_fr("topofr_r100", "topofr", fr.torch_builder("topofr_r100"))
    """
    spec = REGISTRY.get(name)
    if spec is None or not spec.builtin or spec.family not in ("edgeface", "topofr"):
        raise ValueError(f"{name!r} is not a built-in EdgeFace or TopoFR model")

    from . import families  # noqa: PLC0415

    if spec.family == "topofr":
        return lambda pretrained=True: families.build_topofr_net(
            spec.variant, pretrained=pretrained
        )

    def edgeface(pretrained: bool = True):
        if pretrained:
            return families.build_edgeface_net(name)
        from face1kb.codec.identity_loss import (  # noqa: PLC0415
            EDGEFACE_ARCHS,
            build_edgeface,
        )

        return build_edgeface(EDGEFACE_ARCHS[name], pretrained=False)

    return edgeface
