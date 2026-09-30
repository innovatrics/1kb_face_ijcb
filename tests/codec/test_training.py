# SPDX-License-Identifier: MIT
"""Training-path logic on CPU: loss schedule, data sources, sampling, checkpoints."""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from face1kb.codec import data as D  # noqa: E402
from face1kb.codec import identity_loss as IL  # noqa: E402

N_SUBJECTS = 70  # enough for both splits with val_mod=8
IMGS_PER_SUBJECT = 3


def _write_folder(root, size=112):
    from PIL import Image

    rng = np.random.default_rng(0)
    for s in range(N_SUBJECTS):
        d = root / f"s{s:03d}"
        d.mkdir(parents=True)
        for i in range(IMGS_PER_SUBJECT):
            arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{i}.png")
    return root


@pytest.fixture()
def folder(tmp_path):
    return _write_folder(tmp_path / "faces")


def _take(ds, n):
    return list(itertools.islice(iter(ds), n))


# ------------------------------------------------------------------ loss schedule
def test_phase_weights_breakpoints():
    losses = pytest.importorskip("face1kb.codec.losses")
    base = {"R": 1.0, "D": 2.0, "id": 0.5}
    total = 1000
    for step in (0, 10, 19):
        assert losses.phase_weights(step, total, base)["id"] == 0.0
    assert losses.phase_weights(20, total, base)["id"] == 0.0
    mid = losses.phase_weights(110, total, base)["id"]
    assert mid == pytest.approx(0.5 * (0.11 - 0.02) / 0.18)
    assert losses.phase_weights(200, total, base)["id"] == 0.5
    assert losses.phase_weights(999, total, base)["id"] == 0.5
    w = losses.phase_weights(0, total, base)
    assert w["R"] == 1.0 and w["D"] == 2.0  # other terms active from step 0
    assert base["id"] == 0.5  # not mutated


# ------------------------------------------------------------------ data sources
def test_image_folder_source_and_list_cache(folder, tmp_path):
    lst = tmp_path / "list.txt"
    a = D.ImageFolderSource(folder, list_file=lst)
    assert lst.is_file() and len(a) == N_SUBJECTS * IMGS_PER_SUBJECT
    assert a.rels == sorted(a.rels)
    b = D.ImageFolderSource(folder, list_file=lst)  # read back from the cache
    assert b.rels == a.rels
    assert a[0] == {"image": str(folder / a.rels[0]), "subject_id": "s000"}


def test_stale_list_fails_fast(folder, tmp_path):
    lst = tmp_path / "stale.txt"
    lst.write_text("s000/0.jpg\ns000/1.jpg\n")
    with pytest.raises(FileNotFoundError, match="does not match"):
        D.ImageFolderSource(folder, list_file=lst)


def test_empty_folder_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        D.ImageFolderSource(tmp_path / "empty")


# ------------------------------------------------------------------ sampling
def test_random_dataset_is_deterministic_and_split_disjoint(folder):
    tr1 = _take(D.RandomFaceDataset(folder, "train", val_mod=8, seed=3), 40)
    tr2 = _take(D.RandomFaceDataset(folder, "train", val_mod=8, seed=3), 40)
    assert all(torch.equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(tr1, tr2))
    other = _take(D.RandomFaceDataset(folder, "train", val_mod=8, seed=4), 40)
    assert any(a[1] != b[1] for a, b in zip(tr1, other))
    val = _take(D.RandomFaceDataset(folder, "val", val_mod=8, seed=3), 20)
    tr_ids = {i for _, i in tr1}
    val_ids = {i for _, i in val}
    assert tr_ids and val_ids and not tr_ids & val_ids
    t = tr1[0][0]
    assert t.shape == (3, 112, 112) and t.dtype == torch.float32
    assert 0.0 <= float(t.min()) and float(t.max()) <= 1.0


def test_hf_source_matches_image_folder(folder, tmp_path):
    datasets = pytest.importorskip("datasets")
    src = D.ImageFolderSource(folder)
    rows = [src[i] for i in range(len(src))]
    hf = datasets.Dataset.from_dict(
        {
            "image": [{"bytes": None, "path": r["image"]} for r in rows],
            "subject_id": [r["subject_id"] for r in rows],
        },
        features=datasets.Features(
            {"image": datasets.Image(), "subject_id": datasets.Value("string")}
        ),
    )
    hf_dir = tmp_path / "hf"
    hf.save_to_disk(str(hf_dir))
    a = _take(D.RandomFaceDataset(folder, "train", val_mod=8, seed=1), 25)
    b = _take(D.RandomFaceDataset(hf_dir, "train", val_mod=8, seed=1), 25)
    assert all(torch.equal(x[0], y[0]) and x[1] == y[1] for x, y in zip(a, b))


def test_unreadable_rows_raise_instead_of_hanging(
    folder, tmp_path, monkeypatch, caplog
):
    lst = tmp_path / "list.txt"
    src = D.ImageFolderSource(folder, list_file=lst)
    for rel in src.rels[D.LIST_CHECK_ENTRIES :]:
        (folder / rel).write_bytes(b"not an image")
    monkeypatch.setattr(D, "MAX_CONSECUTIVE_FAILURES", 50)
    ds = D.RandomFaceDataset(folder, "train", val_mod=8, list_file=lst)
    with caplog.at_level(logging.WARNING, logger="face1kb.codec.data"):
        with pytest.raises(RuntimeError, match="could not be loaded"):
            _take(ds, 10_000)
    assert "skipping unreadable row" in caplog.text


def test_empty_split_raises(folder, monkeypatch):
    monkeypatch.setattr(D, "MAX_CONSECUTIVE_REJECTIONS", 500)
    # val_mod=1: every subject is held out, so the train split is empty.
    ds = D.RandomFaceDataset(folder, "train", val_mod=1)
    with pytest.raises(RuntimeError, match="split"):
        _take(ds, 1)


def test_rare_corrupt_rows_are_skipped(folder):
    src = D.ImageFolderSource(folder)
    tensors = {rel: D.to_tensor01(str(folder / rel)) for rel in src.rels}
    stream = _take(D.RandomFaceDataset(folder, "train", val_mod=8, seed=5), 60)
    bad = next(r for r, t in tensors.items() if torch.equal(t, stream[2][0]))
    (folder / bad).write_bytes(b"corrupt")
    expected = [x for x in stream if not torch.equal(x[0], tensors[bad])]
    got = _take(D.RandomFaceDataset(folder, "train", val_mod=8, seed=5), len(expected))
    # The same random stream, with the draws of the corrupt row dropped.
    assert len(expected) < len(stream)
    assert all(torch.equal(a[0], b[0]) and a[1] == b[1] for a, b in zip(got, expected))


# ------------------------------------------------------------------ pretrained FR
def test_fetch_pretrained_rejects_wrong_hash(tmp_path):
    spec = IL.PretrainedWeights(
        filename="w.pt", sha256="0" * 64, urls=(), subdir="edgeface"
    )
    (tmp_path / "edgeface").mkdir()
    (tmp_path / "edgeface" / "w.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="sha256"):
        IL.fetch_pretrained(spec, models_root=tmp_path)


def test_fetch_pretrained_accepts_matching_cache(tmp_path):
    p = tmp_path / "edgeface" / "w.pt"
    p.parent.mkdir()
    p.write_bytes(b"weights")
    spec = IL.PretrainedWeights("w.pt", IL.sha256_file(p), urls=())
    assert IL.fetch_pretrained(spec, models_root=tmp_path) == p


def test_edgeface_urls_are_pinned():
    for spec in IL.EDGEFACE_WEIGHTS.values():
        assert len(spec.sha256) == 64
        assert any(
            "/resolve/" in u and len(u.split("/resolve/")[1].split("/")[0]) == 40
            for u in spec.urls
        )


# ------------------------------------------------------------------ checkpoints
def test_export_metadata_and_warmstart_from_safetensors(tmp_path):
    pytest.importorskip("pytorch_msssim")
    from face1kb.codec import api
    from face1kb.codec import train as T

    net = torch.nn.Linear(4, 2)
    p = tmp_path / "abl.safetensors"
    T.save_safetensors(net, p, "accurate", 12, no_side=True, anchor="edgeface_s")
    sd, meta = api.load_state_dict_file(p)
    assert meta["face1kb.no_side"] == "true" and meta["face1kb.step"] == "12"
    with pytest.raises(ValueError, match="non-default architecture"):
        api.load("accurate", device="cpu", weights_path=p)
    ck = T._load_checkpoint(p, "cpu")
    assert set(ck) == {"net"} and torch.equal(ck["net"]["weight"], net.weight)

    q = tmp_path / "fast.safetensors"
    T.save_safetensors(net, q, "fast", 3)
    assert "face1kb.no_side" not in api.load_state_dict_file(q)[1]


def test_train_cli_defaults():
    pytest.importorskip("pytorch_msssim")
    from face1kb.codec import train as T

    a = T.build_parser().parse_args(["--data", "d", "--ckpt", "c"])
    assert (a.steps, a.stop_at, a.batch, a.lr, a.aux_lr) == (
        600_000,
        None,
        32,
        1e-4,
        1e-3,
    )
    assert T.RECIPES["fast"]["res_weights"] == (1, 4, 3, 3, 5)
    assert T.RECIPES["accurate"]["buckets"] == (64, 128, 192, 256)
    assert a.anchor_train_mode is False


class _NotATensor:
    """A picklable object that the restricted (weights-only) unpickler refuses."""


def test_load_checkpoint_is_weights_only(tmp_path):
    pytest.importorskip("pytorch_msssim")
    import pickle

    from face1kb.codec import train as T

    net = torch.nn.Linear(2, 2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.SequentialLR(
        opt,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, total_iters=2),
            torch.optim.lr_scheduler.CosineAnnealingLR(opt, 10, eta_min=1e-5),
        ],
        milestones=[2],
    )
    ok = tmp_path / "last.pt"
    torch.save(
        {
            "net": net.state_dict(),
            "opt": opt.state_dict(),
            "aux_opt": opt.state_dict(),
            "sched": sched.state_dict(),
            "step": 3,
            "variant": "fast",
            "sched_kind": "warmup_cosine",
            "extend_from": 0,
        },
        ok,
    )
    ck = T._load_checkpoint(ok, "cpu")
    assert ck["step"] == 3 and set(ck["net"]) == {"weight", "bias"}
    bad = tmp_path / "bad.pt"
    torch.save({"net": net.state_dict(), "extra": _NotATensor()}, bad)
    with pytest.raises(pickle.UnpicklingError):
        T._load_checkpoint(bad, "cpu")


def test_side_stream_anchor_mode():
    pytest.importorskip("compressai")
    from face1kb.codec.side_stream import IdentitySideStream

    bn_anchor = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.BatchNorm1d(4))
    side = IdentitySideStream(
        anchor_factory=lambda name, pretrained: ("edgeface", bn_anchor)
    )
    side.train()
    assert side.training and side.proj.training and not side.anchor.training
    side.freeze_anchor_mode = False  # the behaviour of the paper's trainer
    side.train()
    assert side.anchor.training
    side.eval()
    assert not side.anchor.training
