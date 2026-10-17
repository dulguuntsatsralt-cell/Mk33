"""Benchmark the KV-cache compression kernels: PyTorch vs Triton.

    python scripts/mk33/bench_kv_kernels.py --out bench/kv_kernels.json

Needs an NVIDIA GPU. For each cache length it times, per layer:

    gather    keep ``keep`` of the rows of K and V (per-head indices)
    key_norm  per-token key L2 norms

and reports milliseconds and achieved memory bandwidth. Eviction is memory-bound, so
bandwidth against the GPU's peak is the number to read. The bytes counted are the
minimum each operation must move: for gather, the kept rows read and written for K
and V plus the int64 indices read once; for key_norm, the keys read and the norms
written. ``torch.gather`` reads its index once per element, not once per row, which
is one reason it moves more bytes than this minimum.

Correctness is checked against the PyTorch reference before any timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mk33.kv.ops import gather_kv_torch, key_norm_torch  # noqa: E402


def timed(fn, iters: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="2048,8192,32768,131072")
    ap.add_argument("--heads", type=int, default=8, help="KV heads per layer")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--keep", type=float, default=0.25, help="fraction of rows kept")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if not torch.cuda.is_available():
        print("needs an NVIDIA GPU (on Colab: Runtime > Change runtime type > GPU)")
        return 1
    from mk33.kv.triton_ops import gather_kv_triton, key_norm_triton

    dtype = getattr(torch, a.dtype)
    esize = torch.tensor([], dtype=dtype).element_size()
    rows = []
    print(f"GPU: {torch.cuda.get_device_name()}, {a.dtype}, batch {a.batch}, {a.heads} KV heads, dim {a.dim}, "
          f"keep {a.keep:.0%}\n")
    print("| tokens | op | torch ms | triton ms | speedup | torch GB/s | triton GB/s |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for t in (int(x) for x in a.lengths.split(",")):
        shape = (a.batch, a.heads, t, a.dim)
        k = torch.randn(shape, dtype=dtype, device="cuda")
        v = torch.randn(shape, dtype=dtype, device="cuda")
        nk = max(1, int(t * a.keep))
        idx = torch.rand(a.batch, a.heads, t, device="cuda").topk(nk, dim=-1).indices.sort(dim=-1).values

        ref, got = gather_kv_torch(k, v, idx), gather_kv_triton(k, v, idx)
        assert torch.equal(ref[0], got[0]) and torch.equal(ref[1], got[1]), "gather mismatch"
        assert torch.allclose(key_norm_torch(k), key_norm_triton(k), rtol=1e-3, atol=1e-3), "key_norm mismatch"

        cases = {
            "gather": (lambda: gather_kv_torch(k, v, idx), lambda: gather_kv_triton(k, v, idx),
                       2 * 2 * a.batch * a.heads * nk * a.dim * esize + idx.numel() * 8),
            "key_norm": (lambda: key_norm_torch(k), lambda: key_norm_triton(k),
                         k.numel() * esize + a.batch * a.heads * t * 4),
        }
        for op, (f_torch, f_triton, nbytes) in cases.items():
            ms_t, ms_k = timed(f_torch, a.iters), timed(f_triton, a.iters)
            row = {"tokens": t, "op": op, "torch_ms": ms_t, "triton_ms": ms_k, "speedup": ms_t / ms_k,
                   "torch_gbs": nbytes / ms_t / 1e6, "triton_gbs": nbytes / ms_k / 1e6, "bytes": nbytes}
            rows.append(row)
            print(f"| {t} | {op} | {ms_t:.4f} | {ms_k:.4f} | {row['speedup']:.2f}x | "
                  f"{row['torch_gbs']:.0f} | {row['triton_gbs']:.0f} |")
        del k, v, idx
        torch.cuda.empty_cache()

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"gpu": torch.cuda.get_device_name(), "args": vars(a), "rows": rows},
                                          indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
