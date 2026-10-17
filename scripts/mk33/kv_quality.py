"""Quick quality check for KV-cache policies before running the agent bench.

    python scripts/mk33/kv_quality.py --model Qwen/Qwen2.5-1.5B-Instruct --smoke
    python scripts/mk33/kv_quality.py --model Qwen/Qwen2.5-1.5B-Instruct --out bench/kv_quality.json

``--smoke`` sends one agent-style request with the bench's tool schemas and prints the
parsed tool call, to confirm the model's chat template and tool-call format work with
``HFChatClient`` before a long run.

Without ``--smoke`` it measures, for every policy and budget, the perplexity of the
last ``--continuation`` tokens of a long document given the ``--context`` tokens
before it (default document: the Mk33 paper's LaTeX source plus the scaffold code,
which looks a lot like an agent's context: prose, code and logs). The gap to the
full cache, as a function of budget, tells you which budgets are worth running in the
agent grid. Random eviction is the floor a real policy has to beat.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mk33.kv import KVPolicy  # noqa: E402
from mk33.kv.client import HFChatClient, set_attention  # noqa: E402
from mk33.kv.generate import CompressedGenerator  # noqa: E402


def default_text() -> str:
    parts = [ROOT / "paper" / "main.tex"] + sorted((ROOT / "src" / "mk33" / "scaffold").glob("*.py"))
    return "\n\n".join(p.read_text(errors="ignore") for p in parts if p.exists())


def smoke(client: HFChatClient) -> int:
    from mk33.scaffold.tools import TOOL_SCHEMAS

    tools = [TOOL_SCHEMAS[n] for n in ("bash", "read_file", "edit_file", "submit")]
    msgs = [{"role": "system", "content": "You improve solution.py over many rounds. Call exactly one tool."},
            {"role": "user", "content": "# Task\nMinimize RMSE on the regression task. Start by reading README.md."}]
    t0 = time.time()
    reply = client.chat(msgs, tools=tools, temperature=0.0, max_tokens=256)
    print(json.dumps(reply, indent=2))
    print(f"\n{time.time() - t0:.1f}s, prompt {client.stats[-1].prompt_tokens} tokens")
    if not reply["tool_calls"]:
        print("No tool call parsed. Check that the model supports tool calling with its chat template.")
        return 1
    print("Tool call parsed: the model works with HFChatClient.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--text", help="file to score (default: paper + scaffold source)")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--continuation", type=int, default=1024)
    ap.add_argument("--policies", default="full,random,streaming,keynorm,h2o")
    ap.add_argument("--budgets", default="512,1024,2048,4096")
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--recent", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--device")
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    client = HFChatClient(a.model, device=a.device, dtype=a.dtype)
    if a.smoke:
        return smoke(client)

    model, tok = client.model, client.tokenizer
    text = Path(a.text).read_text(errors="ignore") if a.text else default_text()
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    need = a.context + a.continuation
    if ids.shape[1] < need:
        print(f"document has {ids.shape[1]} tokens; need {need}. Lower --context or pass a longer --text.")
        return 1
    ids = ids[:, :need]
    print(f"{a.model}: context {a.context}, continuation {a.continuation} tokens\n")
    print("| policy | budget | NLL | perplexity | vs full | peak cache MB | seconds |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    rows, full_nll = [], None
    for name in a.policies.split(","):
        budgets = [0] if name == "full" else [int(b) for b in a.budgets.split(",")]
        for b in budgets:
            pol = KVPolicy(name, budget=b or need, sinks=a.sinks, recent=a.recent)
            set_attention(model, "eager" if pol.needs_attention else "sdpa")
            t0 = time.time()
            nll, st = CompressedGenerator(model, pol, chunk=a.chunk).continuation_nll(ids, a.context)
            secs = time.time() - t0
            if name == "full":
                full_nll = nll
            delta = "" if full_nll is None else f"{nll - full_nll:+.4f}"
            row = {"policy": name, "budget": b or None, "nll": nll, "ppl": math.exp(nll),
                   "peak_cache_mb": st.peak_cache_bytes / 2**20, "seconds": secs}
            rows.append(row)
            print(f"| {name} | {b or 'all'} | {nll:.4f} | {math.exp(nll):.3f} | {delta} | "
                  f"{row['peak_cache_mb']:.1f} | {secs:.1f} |")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"model": a.model, "args": vars(a), "rows": rows}, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
