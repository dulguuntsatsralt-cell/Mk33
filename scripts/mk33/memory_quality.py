"""RAG vs KV-cache eviction at an equal memory budget, with confidence intervals.

    python scripts/mk33/memory_quality.py --model Qwen/Qwen2.5-3B-Instruct --out results/memq_qwen

For several windows of a long corpus (default: this repository's paper, docs, source
and tests, about 95K tokens), it scores the last ``--continuation`` tokens of each
window given the ``--context`` tokens before them, under every method and budget:

    kv:random, kv:streaming, kv:keynorm, kv:h2o
        the model reads the whole context; its KV cache is cut to ``budget`` tokens
    text:recent
        the model reads only the first 4 and the last ``budget - 4`` tokens (truncation)
    text:retrieve
        the model reads the first 4 tokens, the last ``--recent`` tokens, and the earlier
        chunks most relevant to them, chosen by the same BM25 retriever as Mk33's
        ``retrieve`` mode (RAG over the agent's own history)

Every method leaves ``budget`` context tokens in the cache, and the continuation is
appended without further compression, so memory is equal across methods.

Results are written after every measurement (``rows.jsonl``) and a rerun skips what is
done, so a Colab disconnect costs one measurement. ``summary.md`` reports, per method
and budget, the loss increase over the full cache and the paired difference against
random eviction, each with a bootstrap 95% interval over windows.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from mk33.kv import KVPolicy  # noqa: E402
from mk33.kv.client import HFChatClient, set_attention  # noqa: E402
from mk33.kv.generate import CompressedGenerator  # noqa: E402
from mk33.kv.textmem import text_context  # noqa: E402

KV = ("random", "streaming", "keynorm", "h2o")


def corpus_text(root: Path = ROOT) -> str:
    files = [root / "paper" / "main.tex"]
    for pattern in ("docs/mk33/*.md", "src/mk33/**/*.py", "tests/mk33/*.py"):
        files += sorted(root.glob(pattern))
    return "\n\n".join(f"# file: {f.relative_to(root)}\n{f.read_text(errors='ignore')}"
                       for f in files if f.is_file())


def windows(n_tokens: int, need: int, count: int) -> list[int]:
    """Start offsets of up to ``count`` non-overlapping windows, spread over the corpus."""
    slots = n_tokens // need
    if slots == 0:
        return []
    if count >= slots:
        return [i * need for i in range(slots)]
    step = slots / count
    return [int(i * step) * need for i in range(count)]


def bootstrap(xs: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(xs, k=len(xs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def summarize(rows: list[dict], model: str) -> tuple[dict, str]:
    full = {r["window"]: r["nll"] for r in rows if r["method"] == "full"}
    per: dict[tuple[str, int], dict[int, list[float]]] = {}
    for r in rows:
        if r["method"] == "full" or r["window"] not in full:
            continue
        per.setdefault((r["method"], r["budget"]), {}).setdefault(r["window"], []).append(r["nll"])
    # one number per window: mean over seeds
    delta = {k: {w: statistics.mean(v) - full[w] for w, v in d.items()} for k, d in per.items()}
    methods = sorted({m for m, _ in delta}, key=lambda m: (not m.startswith("kv"), m))
    budgets = sorted({b for _, b in delta})
    out = {"model": model, "windows": len(full), "full_nll_mean": statistics.mean(full.values()) if full else None,
           "cells": []}
    lines = [f"# RAG vs KV-cache eviction: {model}", "",
             f"{len(full)} windows. Loss increase over the full cache (nats; lower is better), mean and "
             "bootstrap 95% interval over windows. 'vs random' is the paired difference against kv:random "
             "(negative = better than random); 'yes' when its interval excludes zero.", "",
             "| method | budget | vs full | 95% CI | vs random | 95% CI | better than random? |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for b in budgets:
        rnd = delta.get(("kv:random", b), {})
        for m in methods:
            d = delta.get((m, b))
            if not d:
                continue
            xs = list(d.values())
            lo, hi = bootstrap(xs)
            cell = {"method": m, "budget": b, "n": len(xs), "vs_full": statistics.mean(xs), "vs_full_ci": [lo, hi]}
            vr = ""
            if m != "kv:random" and rnd:
                diffs = [d[w] - rnd[w] for w in d if w in rnd]
                if diffs:
                    dlo, dhi = bootstrap(diffs)
                    cell.update({"vs_random": statistics.mean(diffs), "vs_random_ci": [dlo, dhi],
                                 "better_than_random": dhi < 0, "worse_than_random": dlo > 0})
                    verdict = "yes" if dhi < 0 else ("no, worse" if dlo > 0 else "unclear")
                    vr = f"{statistics.mean(diffs):+.3f} | [{dlo:+.3f}, {dhi:+.3f}] | {verdict}"
            out["cells"].append(cell)
            lines.append(f"| {m} | {b} | {statistics.mean(xs):+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                         + (vr or "(control) | | ") + " |")
    return out, "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--continuation", type=int, default=1024)
    ap.add_argument("--budgets", default="1024,2048,4096")
    ap.add_argument("--methods", default="kv:random,kv:streaming,kv:keynorm,kv:h2o,text:recent,text:retrieve")
    ap.add_argument("--seeds", type=int, default=2, help="seeds for kv:random")
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--recent", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=512, help="prefill chunk")
    ap.add_argument("--h2o-chunk", type=int, default=128, help="smaller chunks for h2o (eager attention)")
    ap.add_argument("--retrieve-chunk", type=int, default=128, help="text:retrieve chunk size in tokens")
    ap.add_argument("--text", help="corpus file (default: this repository)")
    ap.add_argument("--device")
    ap.add_argument("--dtype", default="auto")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {(r["window"], r["method"], r["budget"], r["seed"]) for r in rows}

    client = HFChatClient(a.model, device=a.device, dtype=a.dtype)
    model, tok = client.model, client.tokenizer
    text = Path(a.text).read_text(errors="ignore") if a.text else corpus_text()
    all_ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    need = a.context + a.continuation
    starts = windows(len(all_ids), need, a.windows)
    if not starts:
        print(f"corpus has {len(all_ids)} tokens; need {need} per window")
        return 1
    decode = lambda ids: tok.decode(ids, skip_special_tokens=True)  # noqa: E731
    budgets = [int(b) for b in a.budgets.split(",")]
    methods = a.methods.split(",")
    print(f"{a.model}: {len(starts)} windows of {a.context}+{a.continuation} tokens "
          f"(corpus {len(all_ids)} tokens), budgets {budgets}")

    def record(**row):
        rows.append(row)
        with rows_path.open("a") as fh:
            fh.write(json.dumps(row) + "\n")

    def run_kv(ids, name, budget, seed):
        pol = KVPolicy(name, budget=budget, sinks=a.sinks, recent=a.recent, seed=seed)
        set_attention(model, "eager" if pol.needs_attention else "sdpa")
        chunk = a.h2o_chunk if pol.needs_attention else a.chunk
        nll, st = CompressedGenerator(model, pol, chunk=chunk).continuation_nll(
            ids, a.context, compress_continuation=False)
        return nll, st.peak_cache_tokens

    for w, start in enumerate(starts):
        ids = all_ids[start:start + need].unsqueeze(0)
        t_window = time.time()
        jobs = [("full", 0, 0)]
        for b in budgets:
            for m in methods:
                seeds = range(a.seeds) if m == "kv:random" else [0]
                jobs += [(m, b, s) for s in seeds]
        for m, b, seed in jobs:
            if (w, m, b, seed) in done:
                continue
            t0 = time.time()
            if m == "full":
                set_attention(model, "sdpa")
                nll, st = CompressedGenerator(model, KVPolicy("full"), chunk=a.chunk).continuation_nll(ids, a.context)
                peak = st.peak_cache_tokens
            elif m.startswith("kv:"):
                nll, peak = run_kv(ids, m[3:], b, seed)
            elif m.startswith("text:"):
                ctx, _ = text_context(ids[0, :a.context].tolist(), b, m[5:], decode, sinks=a.sinks,
                                      recent=a.recent, chunk=a.retrieve_chunk)
                new = torch.cat([torch.tensor([ctx]), ids[:, a.context:]], dim=1)
                set_attention(model, "sdpa")
                nll, st = CompressedGenerator(model, KVPolicy("full"), chunk=a.chunk).continuation_nll(new, len(ctx))
                peak = st.peak_cache_tokens
            else:
                raise SystemExit(f"unknown method {m}")
            record(window=w, start=int(start), method=m, budget=b, seed=seed, nll=nll, peak_cache_tokens=peak,
                   seconds=round(time.time() - t0, 2))
        print(f"window {w + 1}/{len(starts)} done in {time.time() - t_window:.0f}s")
        write_summary(out, rows, a.model)  # refreshed after every window
    print()
    print(write_summary(out, rows, a.model))
    return 0


def write_summary(out: Path, rows: list[dict], model: str) -> str:
    summary, md = summarize(rows, model)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(md)
    return md


if __name__ == "__main__":
    sys.exit(main())
