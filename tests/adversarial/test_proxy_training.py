# SPDX-License-Identifier: MIT
"""Li-AE proxy training recipe: identity selection, prototypes, training, CLI (CPU)."""

from __future__ import annotations

import io

import numpy as np
import pytest

torch = pytest.importorskip("torch")
Image = pytest.importorskip("PIL.Image")

from face1kb.adversarial import proxy as P  # noqa: E402
from face1kb.adversarial import train as T  # noqa: E402

from ._synthetic import crops  # noqa: E402


def png_bytes(a: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def img(k: int, size: int = 112) -> np.ndarray:
    return np.full((size, size, 3), k % 256, np.uint8)


def test_decode_crop_inputs(tmp_path):
    a = crops(1, 112)[0]
    p = tmp_path / "a.png"
    Image.fromarray(a).save(p)
    for src in (a, png_bytes(a), p, str(p), Image.fromarray(a)):
        assert np.array_equal(T.decode_crop(src), a)
    small = T.decode_crop(crops(1, 64)[0])
    assert small.shape == (112, 112, 3) and small.dtype == np.uint8
    want = np.asarray(Image.fromarray(crops(1, 64)[0]).resize((112, 112), 2))
    assert np.array_equal(small, want)  # PIL bilinear
    with pytest.raises(ValueError):
        T.decode_crop(np.zeros((8, 8), np.uint8))


def test_select_identities_rules():
    # subject "a": 5 images, "b": 2 (below min_per_id), "c": 6 (capped at 4),
    # "d": 4 with one undecodable image, "e": would come after the stop
    records = (
        [("a", img(i)) for i in range(5)]
        + [("b", img(10 + i)) for i in range(2)]
        + [("c", img(20 + i)) for i in range(6)]
        + [("d", b"not an image")]
        + [("d", img(30 + i)) for i in range(4)]
        + [("e", img(40 + i)) for i in range(4)]
    )
    images, labels, subjects = T.select_identities(
        records, n_ids=3, per_id=4, min_per_id=3
    )
    # stop once 3 subjects are full (a, c, d); b is skipped (2 < 3); e never read
    assert subjects == ["a", "c", "d"]
    assert labels.tolist() == [0] * 4 + [1] * 4 + [2] * 4
    assert [int(x[0, 0, 0]) for x in images] == [
        0,
        1,
        2,
        3,
        20,
        21,
        22,
        23,
        30,
        31,
        32,
        33,
    ]
    assert images.dtype == np.uint8 and images.shape == (12, 112, 112, 3)


def test_select_identities_keeps_partial_subjects_and_stops_early():
    def stream():
        yield from [("p", img(1)), ("p", img(2)), ("p", img(3))]  # partial (3 < 4)
        yield from [("q", img(5 + i)) for i in range(4)]
        raise AssertionError("read past the stop")

    images, labels, subjects = T.select_identities(
        stream(), n_ids=1, per_id=4, min_per_id=2
    )
    # the first n_ids subjects with >= min_per_id crops, in order of appearance
    assert subjects == ["p"] and len(images) == 3


def test_select_identities_max_rows_and_resize():
    records = [("a", crops(1, 64)[0])] * 3 + [("b", img(1))] * 3
    images, labels, subjects = T.select_identities(
        records, n_ids=5, per_id=3, min_per_id=1, max_rows=4
    )
    assert subjects == ["a", "b"] and labels.tolist() == [0, 0, 0, 1]
    assert images.shape[1:] == (112, 112, 3)
    with pytest.raises(ValueError):
        T.select_identities([("a", img(0))], min_per_id=2)


def test_iter_image_folder(tmp_path):
    for s in ("s2", "s1"):
        (tmp_path / s).mkdir()
        for n in ("b.png", "a.jpg", "notes.txt"):
            (tmp_path / s / n).write_bytes(b"x")
    got = [(s, p.name) for s, p in T.iter_image_folder(tmp_path)]
    assert got == [("s1", "a.jpg"), ("s1", "b.png"), ("s2", "a.jpg"), ("s2", "b.png")]
    only = [s for s, _ in T.iter_image_folder(tmp_path, subjects=["s2"])]
    assert only == ["s2", "s2"]


def test_prototypes():
    x = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).view(4, 3, 2, 2)
    y = torch.tensor([0, 1, 0, 1])
    proto = T.prototypes(x, y)
    assert torch.equal(proto[0], (x[0] + x[2]) / 2)
    assert torch.equal(proto[1], (x[1] + x[3]) / 2)


def tiny_set():
    images = np.concatenate([crops(3, 112), crops(3, 112)[:, ::-1]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    return np.ascontiguousarray(images), labels


def test_train_proxy_seeded_and_isolated():
    images, labels = tiny_set()
    state = torch.random.get_rng_state()
    net, hist = T.train_proxy(
        images, labels, epochs=2, batch_size=4, width=4, device="cpu"
    )
    assert torch.equal(state, torch.random.get_rng_state())
    assert not net.training and net.width == 4
    assert len(hist) == 2 and all(np.isfinite(hist))
    net2, hist2 = T.train_proxy(
        images, labels, epochs=2, batch_size=4, width=4, device="cpu"
    )
    assert hist == hist2
    for k, v in net.state_dict().items():
        assert torch.equal(v, net2.state_dict()[k]), k


def test_train_proxy_matches_reference_loop():
    """The optimisation equals the plain recipe loop (global seed, CPU)."""
    images, labels = tiny_set()
    net, hist = T.train_proxy(
        images, labels, epochs=3, batch_size=4, lr=1e-3, width=4, device="cpu"
    )

    torch.manual_seed(0)
    x = torch.from_numpy(images.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    y = torch.from_numpy(labels)
    proto = torch.zeros(2, 3, 112, 112)
    for c in range(2):
        proto[c] = x[y == c].mean(0)
    tgt = proto[y]
    ref = P.ProxyAE(width=4).train()
    opt = torch.optim.Adam(ref.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 3)
    g = torch.Generator(device="cpu").manual_seed(0)
    ref_hist = []
    for _ in range(3):
        perm = torch.randperm(6, generator=g)
        tot = 0.0
        for s in range(0, 6, 4):
            idx = perm[s : s + 4]
            loss = torch.nn.functional.mse_loss(ref(x[idx])[0], tgt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        ref_hist.append(tot / 6)
    assert hist == ref_hist
    for k, v in ref.state_dict().items():
        assert torch.equal(v, net.state_dict()[k]), k


def test_cli(tmp_path):
    root = tmp_path / "faces"
    for s in range(3):
        (root / f"id{s}").mkdir(parents=True)
        for i, a in enumerate(crops(2, 112)):
            Image.fromarray(np.roll(a, 7 * s, axis=1)).save(root / f"id{s}/{i}.png")
    out = tmp_path / "proxy.safetensors"
    args = "--ids 2 --per-id 2 --min-per-id 2 --epochs 1 --batch-size 2 --width 4"
    args += " --device cpu --source synthetic"
    T.main(["--images-root", str(root), "--out", str(out), *args.split()])
    _, meta = P.read_proxy_file(out)
    assert meta["face1kb.n_ids"] == "2" and meta["face1kb.n_imgs"] == "4"
    assert meta["face1kb.epochs"] == "1" and meta["face1kb.source"] == "synthetic"
    assert P.load_proxy(out).width == 4
