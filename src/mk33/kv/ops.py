"""Tensor operations behind KV-cache compression, with a PyTorch reference.

    gather_kv(k, v, idx)  keep, for every (batch, head), the cache rows listed in idx
    key_norm(k)           L2 norm of every cached key, in float32

Both take tensors laid out like a Hugging Face cache layer: ``[batch, kv_heads,
tokens, head_dim]``. ``idx`` is ``[batch, kv_heads, kept]`` and may differ per head.
On CUDA, the Triton kernels in ``triton_ops`` are used when Triton is installed
(``backend="auto"``); everywhere else the PyTorch versions run.
"""

from __future__ import annotations

import os

import torch


def gather_kv_torch(k: torch.Tensor, v: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    e = idx.unsqueeze(-1).expand(*idx.shape, k.shape[-1])
    return torch.gather(k, 2, e), torch.gather(v, 2, e)


def key_norm_torch(k: torch.Tensor) -> torch.Tensor:
    return k.float().norm(dim=-1)


def triton_available(device: torch.device) -> bool:
    if os.environ.get("MK33_NO_TRITON"):
        return False
    try:
        import triton  # noqa: F401
    except ImportError:
        return False
    return device.type == "cuda" or os.environ.get("TRITON_INTERPRET") == "1"


def _use_triton(t: torch.Tensor, backend: str) -> bool:
    if backend == "torch":
        return False
    if backend == "triton":
        return True
    return triton_available(t.device)


def gather_kv(k, v, idx, backend: str = "auto"):
    if _use_triton(k, backend):
        from .triton_ops import gather_kv_triton

        return gather_kv_triton(k, v, idx)
    return gather_kv_torch(k, v, idx)


def key_norm(k, backend: str = "auto"):
    if _use_triton(k, backend):
        from .triton_ops import key_norm_triton

        return key_norm_triton(k)
    return key_norm_torch(k)
