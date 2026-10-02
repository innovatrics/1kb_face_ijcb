# EdgeFace backbone (vendored)

This directory vendors the two files of the EdgeFace repository that are needed to
instantiate the EdgeFace network architectures. They are used by

* the identity side-stream of face1kb-ACCURATE, whose frozen anchor is
  `edgeface_s_gamma_05` (EdgeFace-S, low-rank factor 0.5), and
* the in-loop identity loss used for training (EdgeFace-XS for FAST, EdgeFace-S for
  ACCURATE).

| | |
|---|---|
| Upstream | <https://github.com/otroshi/edgeface> |
| Commit | `ce86851cfc37979a9cd2558598d0e9bc592cbba3` (the `backbones/` directory is unchanged since `2a8a92e327dd4eeb701195e8f40d3f757dd737d6`) |
| Files | `backbones/__init__.py` -> `__init__.py`, `backbones/timmfr.py` -> `timmfr.py` (verbatim, unmodified) |
| Code licence | BSD 3-Clause, see [LICENSE](LICENSE) (copied from the upstream repository) |
| Authors | Anjith George, Christophe Ecabert, Hatef Otroshi Shahreza, Ketan Kotwal, Sébastien Marcel (Idiap Research Institute) |

SHA-256 of the vendored files:

```text
f5b480e7bfa19214cc27b5fa092d2948dda444c9c3ea9edec726c02ea1f7884a  __init__.py
5f2cec2f38c7f9c320abf12df86d7f46727e08c24e678cf668345473e5a40f12  timmfr.py
```

Only code is vendored here. The pretrained EdgeFace weights are **not** part of this
directory: they are released by Idiap under CC BY-NC-SA 4.0 and are downloaded on
demand (with SHA-256 verification) by `face1kb.codec.identity_loss` from the official
Idiap model repositories on Hugging Face (`Idiap/EdgeFace-XS-GAMMA`,
`Idiap/EdgeFace-S-GAMMA`). The face1kb-ACCURATE weight file bundles a copy of the
EdgeFace-S weights (bit-identical to `edgeface_s_gamma_05.pt`); see
`weights/README.md`.

The architectures are built on `timm` (`edgenext_x_small`, `edgenext_small`); the
state-dict key names come from timm and were verified with `timm==1.0.24`.

If you use EdgeFace, please cite:

```bibtex
@article{edgeface,
  title={Edgeface: Efficient face recognition model for edge devices},
  author={George, Anjith and Ecabert, Christophe and Shahreza, Hatef Otroshi and Kotwal, Ketan and Marcel, Sebastien},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2024}
}
```
