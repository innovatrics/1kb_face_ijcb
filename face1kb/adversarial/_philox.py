# SPDX-License-Identifier: MIT
"""Device-independent re-computation of torch's CUDA ``uniform_`` stream.

On CUDA, ``torch.Tensor.uniform_`` draws from the counter-based Philox-4x32-10
generator (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3", SC 2011) in
a grid-stride kernel. Which random number lands on which tensor element depends on
the launch geometry, i.e. on the number of streaming multiprocessors of the GPU, so
the same seed gives different random starts on different GPU models.
:func:`cuda_uniform` recomputes, with NumPy, the values that the kernel writes on a
GPU that launches ``blocks`` blocks of 256 threads, so a random start of the paper can
be reproduced on any GPU or on the CPU.

The emulated behaviour (float32 tensors, torch 2.x):

* the launch uses ``min(blocks, ceil(numel / 256))`` blocks of 256 threads, where
  ``blocks = SMs * (max threads per SM // 256)`` (see :func:`cuda_launch_blocks`);
* thread ``t`` seeds Philox with key ``seed``, subsequence ``t`` and the generator's
  current offset, and its ``k``-th call yields four 32-bit words; word ``i`` goes to
  element ``t + T * (4k + i)`` in memory order (``T`` = number of threads);
* a word ``x`` becomes ``u = x * 2**-32 + 2**-33`` in ``(0, 1]``, then
  ``fma(u, high - low, low)`` in float32, and ``high`` is mapped to ``low``;
* the generator offset then advances by ``4 * ceil(numel / (4 T))``.
"""

from __future__ import annotations

import numpy as np

_BLOCK = 256
_M0, _M1 = np.uint64(0xD2511F53), np.uint64(0xCD9E8D57)
_W0, _W1 = np.uint64(0x9E3779B9), np.uint64(0xBB67AE85)
_LO32 = np.uint64(0xFFFFFFFF)
_SHIFT = np.uint64(32)


def philox4x32_10(ctr: tuple[np.ndarray, ...], key: tuple[int, int]) -> np.ndarray:
    """Apply Philox-4x32-10 to arrays of counters; return ``(..., 4)`` uint32."""
    c0, c1, c2, c3 = (np.asarray(c, np.uint64) & _LO32 for c in ctr)
    k0, k1 = np.uint64(key[0]) & _LO32, np.uint64(key[1]) & _LO32
    for rnd in range(10):
        p0 = _M0 * c0
        p1 = _M1 * c2
        c0, c1, c2, c3 = (
            (p1 >> _SHIFT) ^ c1 ^ k0,
            p1 & _LO32,
            (p0 >> _SHIFT) ^ c3 ^ k1,
            p0 & _LO32,
        )
        if rnd < 9:
            k0, k1 = (k0 + _W0) & _LO32, (k1 + _W1) & _LO32
    return np.stack([c0, c1, c2, c3], axis=-1).astype(np.uint32)


def fma_f32(a: np.ndarray, b: np.float32, c: np.float32) -> np.ndarray:
    """Return float32 ``a * b + c`` rounded once, like CUDA ``fmaf``.

    The product of two float32 values is exact in float64 and the float64 sum rounds
    at most once. Rounding that sum to float32 is then correct unless it lies exactly
    on a float32 rounding midpoint (or in the float32 subnormal range); only those
    elements are corrected, with the exact rounding error of the sum (TwoSum).
    """
    p = np.asarray(a, np.float32).astype(np.float64) * np.float64(b)  # exact
    cc = np.float64(c)
    s = p + cc
    r = s.astype(np.float32)
    # float64 with 29 dropped mantissa bits equal to 1000...0: a float32 midpoint
    low = s.view(np.uint64) & np.uint64((1 << 29) - 1)
    check = (low == np.uint64(1 << 28)) | (np.abs(s) < 2.0**-125)
    if check.any():
        i = np.nonzero(check)
        si, pi = s[i], p[i]
        bb = si - pi
        err = (pi - (si - bb)) + (cc - bb)  # exact rounding error of the sum
        ri = r[i]
        rd = ri.astype(np.float64)
        up = np.nextafter(ri, np.float32(np.inf))
        down = np.nextafter(ri, np.float32(-np.inf))
        # ties-to-even chose ri; the exact value lies on the neighbour's side
        ri = np.where((2 * si == rd + up.astype(np.float64)) & (err > 0), up, ri)
        ri = np.where((2 * si == rd + down.astype(np.float64)) & (err < 0), down, ri)
        r[i] = ri
    return r


def launch_threads(numel: int, blocks: int) -> int:
    """Return the thread count of the grid-stride launch for ``numel`` values."""
    grid = min(int(blocks), (int(numel) + _BLOCK - 1) // _BLOCK)
    return grid * _BLOCK


def offset_increment(numel: int, blocks: int) -> int:
    """Return the Philox offset one ``uniform_`` call on ``numel`` values uses."""
    if numel <= 0:
        return 0
    t = launch_threads(numel, blocks)
    return ((int(numel) - 1) // (t * 4) + 1) * 4


def cuda_uniform(
    numel: int, low: float, high: float, seed: int, offset: int, blocks: int
) -> np.ndarray:
    """Return CUDA ``uniform_(low, high)`` values of a float32 tensor (memory order).

    Parameters
    ----------
    numel
        Number of tensor elements.
    low, high
        Bounds of ``uniform_``.
    seed
        Seed of the generator (``manual_seed``).
    offset
        Philox offset of the generator before the call (0 after seeding; advance it
        by :func:`offset_increment` after every call).
    blocks
        Launch blocks of the emulated GPU (see :func:`cuda_launch_blocks`).

    Returns
    -------
    np.ndarray
        ``(numel,)`` float32 values; element ``i`` is the value of the element at
        memory position ``i`` of a dense tensor.
    """
    numel = int(numel)
    if numel <= 0:
        return np.empty(0, np.float32)
    if offset % 4:
        raise ValueError("the Philox offset of torch's generator is a multiple of 4")
    t = launch_threads(numel, blocks)
    calls = (numel - 1) // (t * 4) + 1
    k = np.arange(calls, dtype=np.uint64)[:, None]
    tid = np.arange(t, dtype=np.uint64)[None, :]
    seed = int(seed) & 0xFFFFFFFFFFFFFFFF
    words = philox4x32_10(
        (
            np.broadcast_to(k + np.uint64(offset // 4), (calls, t)),
            np.zeros((calls, t), np.uint64),
            np.broadcast_to(tid & _LO32, (calls, t)),
            np.broadcast_to(tid >> _SHIFT, (calls, t)),
        ),
        (seed & 0xFFFFFFFF, seed >> 32),
    )  # (calls, t, 4): word i of call k of thread t -> element t + T * (4k + i)
    words = words.transpose(0, 2, 1).reshape(-1)[:numel]
    inv = np.float32(2.0**-32)
    u = words.astype(np.float32) * inv + np.float32(2.0**-33)
    lo, hi = np.float32(low), np.float32(high)
    v = fma_f32(u, np.float32(hi - lo), lo)
    return np.where(v == hi, lo, v).astype(np.float32)


def cuda_launch_blocks(device=None) -> int:
    """Return the launch blocks of torch's CUDA distribution kernels on ``device``.

    ``multiProcessorCount * (maxThreadsPerMultiProcessor // 256)``, e.g. 272 on a
    GeForce RTX 2080 Ti (68 SMs) and 288 on a Quadro RTX 6000 (72 SMs).
    """
    import torch  # noqa: PLC0415

    p = torch.cuda.get_device_properties(device)
    per_sm = int(p.max_threads_per_multi_processor) // _BLOCK
    return int(p.multi_processor_count) * per_sm
