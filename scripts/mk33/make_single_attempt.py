#!/usr/bin/env python3
"""Build the token-matched single-attempt training set for ablation A7.

1. Collect single-attempt rollouts on the same environments. The A7 config limits the
   rollout budget to one round:
       mk33 rollout --config configs/mk33/ablations/A7_single_attempt.yaml ...
   then run ``mk33 build-sft`` on them to get the pool of windows.
2. Run this script to subsample those windows until their token count matches the
   improvement windows used by the main model.

Usage:
    python scripts/mk33/make_single_attempt.py \
        --reference data/mk33/windows.jsonl \
        --pool data/mk33/windows_single_attempt_pool.jsonl \
        --out data/mk33/windows_single_attempt.jsonl
"""

from __future__ import annotations

import argparse
import json
import random

from mk33.tokens import approx_tokens


def window_tokens(w: dict) -> int:
    return sum(approx_tokens(str(m.get("content") or "")) for m in w["messages"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ref = [json.loads(l) for l in open(a.reference) if l.strip()]
    pool = [json.loads(l) for l in open(a.pool) if l.strip()]
    target = sum(window_tokens(w) for w in ref)
    random.Random(a.seed).shuffle(pool)
    out, total = [], 0
    for w in pool:
        if total >= target:
            break
        out.append(w)
        total += window_tokens(w)
    with open(a.out, "w") as fh:
        for w in out:
            fh.write(json.dumps(w, ensure_ascii=False) + "\n")
    status = "matched" if total >= target else "pool too small"
    print(json.dumps({"target_tokens": target, "selected_tokens": total, "windows": len(out), "status": status}))


if __name__ == "__main__":
    main()
