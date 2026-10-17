"""Triton kernels match the PyTorch reference exactly.

Runs on a GPU, or on CPU with ``TRITON_INTERPRET=1`` (set before Python starts; the CI
job does this). Skipped otherwise.
"""

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

CUDA = torch.cuda.is_available()
if not CUDA and os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("needs a GPU or TRITON_INTERPRET=1", allow_module_level=True)

from mk33.kv.ops import gather_kv_torch, key_norm_torch  # noqa: E402
from mk33.kv.triton_ops import gather_kv_triton, key_norm_triton  # noqa: E402

DEV = "cuda" if CUDA else "cpu"
DTYPES = [torch.float32, torch.float16] + ([torch.bfloat16] if CUDA else [])


def rand_idx(b, h, t, k, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.stack([torch.stack([torch.randperm(t, generator=g)[:k].sort().values for _ in range(h)])
                        for _ in range(b)]).to(DEV)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 2, 37, 64), (2, 3, 130, 24), (1, 8, 300, 128)])
def test_gather_matches_torch(dtype, shape):
    b, h, t, d = shape
    k = torch.randn(shape, dtype=dtype, device=DEV)
    v = torch.randn(shape, dtype=dtype, device=DEV)
    idx = rand_idx(b, h, t, t // 3 + 1)
    ref, got = gather_kv_torch(k, v, idx), gather_kv_triton(k, v, idx)
    assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1])


def test_gather_handles_strided_inputs_and_empty_selection():
    base = torch.randn(1, 50, 4, 32, device=DEV)  # [B, T, H, D] storage, viewed as [B, H, T, D]
    k = base.transpose(1, 2)
    v = torch.randn(1, 4, 50, 32, device=DEV)
    idx = rand_idx(1, 4, 50, 20)
    assert not k.is_contiguous()
    assert torch.equal(gather_kv_triton(k, v, idx)[0], gather_kv_torch(k, v, idx)[0])
    empty = gather_kv_triton(k, v, idx[..., :0])
    assert empty[0].shape == (1, 4, 0, 32)


@pytest.mark.parametrize("dtype", DTYPES)
def test_key_norm_matches_torch(dtype):
    k = torch.randn(2, 4, 77, 96, dtype=dtype, device=DEV) * 3
    assert torch.allclose(key_norm_triton(k), key_norm_torch(k), rtol=1e-5, atol=1e-5)


def test_generator_output_is_identical_with_either_backend():
    pytest.importorskip("transformers")
    from mk33.kv import KVPolicy
    from mk33.kv.generate import CompressedGenerator
    from test_kv import ids, tiny_model

    m = tiny_model().to(DEV)
    outs = []
    for backend in ("torch", "triton"):
        pol = KVPolicy("keynorm", budget=40, sinks=4, recent=8, backend=backend)
        outs.append(CompressedGenerator(m, pol, chunk=16).generate(ids(150), max_new_tokens=20)[0])
    assert outs[0] == outs[1]
