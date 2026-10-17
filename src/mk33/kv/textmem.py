"""Text-level memory at a token budget, for comparison with KV-cache eviction.

KV policies shorten what the model *remembers*: it reads the whole context, then
forgets keys and values. Text-level methods shorten what the model *reads*: the
context is cut to ``budget`` tokens before the model sees it, and the result is
encoded from scratch. At the same budget both leave the same number of tokens in the
cache, so their quality can be compared directly.

    recent     the first ``sinks`` tokens + the most recent tokens (what ``truncate`` does)
    retrieve   the first ``sinks`` tokens + the ``recent`` most recent tokens + the earlier
               chunks most relevant to those recent tokens, in their original order. Uses
               the same BM25 retriever as Mk33's ``retrieve`` context mode.

No PyTorch needed: everything works on lists of token ids and a ``decode`` function.
"""

from __future__ import annotations

from typing import Callable

from ..scaffold.memory import BM25Retriever, MemoryItem

TEXT_MODES = ("recent", "retrieve")


def text_context(ids: list[int], budget: int, mode: str, decode: Callable[[list[int]], str],
                 sinks: int = 4, recent: int = 256, chunk: int = 128) -> tuple[list[int], dict]:
    """Shorten ``ids`` to exactly ``budget`` tokens (or fewer, if it is already shorter)."""
    if mode not in TEXT_MODES:
        raise ValueError(f"unknown text mode {mode!r}; choose from {', '.join(TEXT_MODES)}")
    n = len(ids)
    if n <= budget:
        return list(ids), {"retrieved_chunks": 0, "recent_tokens": n}
    s = min(sinks, budget)
    if mode == "recent":
        return list(ids[:s]) + list(ids[n - (budget - s):]), {"retrieved_chunks": 0, "recent_tokens": budget - s}

    r = min(recent, budget - s)
    room = budget - s - r
    lo, hi = s, n - r  # the retrievable middle
    spans = [(a, min(a + chunk, hi)) for a in range(lo, hi, chunk)]
    retriever = BM25Retriever()
    retriever.add([MemoryItem(i, 0, decode(list(ids[a:b])), b - a) for i, (a, b) in enumerate(spans)])
    hits = [it.turn_index for it, _ in retriever.search(decode(list(ids[n - r:])), len(spans))]
    matched = len(hits)
    # Chunks with no keyword overlap rank after every hit, most recent first.
    hit_set = set(hits)
    order = hits + [i for i in reversed(range(len(spans))) if i not in hit_set]

    keep: dict[int, int] = {}  # chunk index -> tokens kept from its start
    left = room
    for i in order:
        if left <= 0:
            break
        a, b = spans[i]
        take = min(b - a, left)  # the last pick may be partial, so the budget is exact
        keep[i] = take
        left -= take
    out = list(ids[:s])
    for i in sorted(keep):
        a, _ = spans[i]
        out.extend(ids[a:a + keep[i]])
    out.extend(ids[n - r:])
    assert len(out) == budget
    return out, {"retrieved_chunks": len(keep), "keyword_matches": matched, "recent_tokens": r}
