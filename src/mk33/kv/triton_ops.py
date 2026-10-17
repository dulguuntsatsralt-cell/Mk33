"""Triton kernels for KV-cache compression.

``gather_kv_triton``
    One launch copies the kept rows of both K and V. PyTorch needs two ``gather``
    calls plus a materialized ``[B, H, kept, D]`` index tensor; the kernel reads each
    index once and uses it for both tensors. Eviction is memory-bound, so the kernel
    is judged by achieved bandwidth (see ``scripts/mk33/bench_kv_kernels.py``).

``key_norm_triton``
    Per-token key L2 norms with float32 accumulation, without materializing a float32
    copy of the keys as ``k.float().norm(-1)`` does.

Grid: one program per (batch * head, block of rows). Each program handles
``BLOCK_N`` rows of ``head_dim`` elements; ``BLOCK_D`` is head_dim rounded up to a
power of two, with masking. Both kernels accept arbitrary strides, so a cache layer
that was sliced or transposed does not need a copy first.

The kernels run on NVIDIA GPUs, and on CPU through Triton's interpreter
(``TRITON_INTERPRET=1``), which is how the tests check them without a GPU.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kv_kernel(
    K, V, IDX, OK, OV,
    H, NK, D,
    skb, skh, skt, skd,
    svb, svh, svt, svd,
    sib, sih, sin,
    sob, soh, sot, sod,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < NK
    mask = mask_n[:, None] & (offs_d[None, :] < D)

    rows = tl.load(IDX + b * sib + h * sih + offs_n * sin, mask=mask_n, other=0)
    k = tl.load(K + b * skb + h * skh + rows[:, None] * skt + offs_d[None, :] * skd, mask=mask)
    v = tl.load(V + b * svb + h * svh + rows[:, None] * svt + offs_d[None, :] * svd, mask=mask)
    dst = b * sob + h * soh + offs_n[:, None] * sot + offs_d[None, :] * sod
    tl.store(OK + dst, k, mask=mask)
    tl.store(OV + dst, v, mask=mask)


@triton.jit
def _key_norm_kernel(
    K, OUT,
    H, T, D,
    skb, skh, skt, skd,
    sob, soh, sot,
    BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    bh = tl.program_id(0)
    pid_n = tl.program_id(1)
    b = bh // H
    h = bh % H
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    mask_n = offs_n < T
    mask = mask_n[:, None] & (offs_d[None, :] < D)
    k = tl.load(K + b * skb + h * skh + offs_n[:, None] * skt + offs_d[None, :] * skd, mask=mask, other=0.0)
    k = k.to(tl.float32)
    norm = tl.sqrt(tl.sum(k * k, axis=1))
    tl.store(OUT + b * sob + h * soh + offs_n * sot, norm, mask=mask_n)


def _block_n(d: int) -> int:
    """Rows per program: a power of two (tl.arange needs one), about 4K elements per tile."""
    rows = 4096 // triton.next_power_of_2(max(d, 1))
    return max(16, min(128, rows))


def gather_kv_triton(k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert k.shape == v.shape and k.dim() == 4 and idx.dim() == 3
    B, H, _, D = k.shape
    nk = idx.shape[-1]
    ok = torch.empty((B, H, nk, D), dtype=k.dtype, device=k.device)
    ov = torch.empty_like(ok)
    if nk == 0:
        return ok, ov
    block_n = _block_n(D)
    grid = (B * H, triton.cdiv(nk, block_n))
    _gather_kv_kernel[grid](
        k, v, idx, ok, ov,
        H, nk, D,
        *k.stride(), *v.stride(), *idx.stride(), *ok.stride(),
        BLOCK_N=block_n, BLOCK_D=triton.next_power_of_2(D),
    )
    return ok, ov


def key_norm_triton(k: torch.Tensor) -> torch.Tensor:
    assert k.dim() == 4
    B, H, T, D = k.shape
    out = torch.empty((B, H, T), dtype=torch.float32, device=k.device)
    if T == 0:
        return out
    block_n = _block_n(D)
    grid = (B * H, triton.cdiv(T, block_n))
    _key_norm_kernel[grid](
        k, out, H, T, D, *k.stride(), *out.stride(),
        BLOCK_N=block_n, BLOCK_D=triton.next_power_of_2(D),
    )
    return out
