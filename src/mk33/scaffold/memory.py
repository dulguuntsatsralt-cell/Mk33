"""Retrieval memory for turns that no longer fit in the context window.

When a context mode with retrieval compacts the context, the verbatim turns it drops
are not lost: each one becomes a memory item. At every refresh (each new round, and
every compaction) the most relevant items are retrieved and placed in the context
prefix, under a fixed token budget.

Two retrievers are provided:

    BM25Retriever       standard library only; the default, so CPU runs need nothing extra
    EmbeddingRetriever  cosine similarity over embeddings from any function, e.g. an
                        OpenAI-compatible ``/embeddings`` endpoint (``openai_embedder``)

The retrieved block sits in the prefix, which is frozen between events, so a refresh
is a context event exactly like a compaction: it starts a new training window and
invalidates the model's KV cache for everything after the system prompt. That cost is
part of what the bench measures.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from ..tokens import approx_tokens

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?")
_STOP = frozenset(
    "a an the to of and or with for in on is are was be it this that as at by from not no "
    "i we you if then else so but".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased words and numbers; identifiers like ``max_depth`` stay whole and also split."""
    out: list[str] = []
    for w in _WORD.findall(text.lower()):
        if w in _STOP:
            continue
        out.append(w)
        if "_" in w:
            out.extend(p for p in w.split("_") if p and p not in _STOP)
    return out


@dataclass
class MemoryItem:
    turn_index: int
    round: int
    text: str
    tokens: int


class Retriever(Protocol):
    def add(self, items: Sequence[MemoryItem]) -> None: ...

    def search(self, query: str, k: int) -> list[tuple[MemoryItem, float]]: ...


@dataclass
class BM25Retriever:
    """Okapi BM25 over memory items. Ties break toward the more recent turn."""

    k1: float = 1.5
    b: float = 0.75
    items: list[MemoryItem] = field(default_factory=list)
    _tf: list[Counter] = field(default_factory=list)
    _df: Counter = field(default_factory=Counter)
    _len_sum: int = 0

    def add(self, items: Sequence[MemoryItem]) -> None:
        for it in items:
            tf = Counter(tokenize(it.text))
            self.items.append(it)
            self._tf.append(tf)
            self._df.update(tf.keys())
            self._len_sum += sum(tf.values())

    def search(self, query: str, k: int) -> list[tuple[MemoryItem, float]]:
        if not self.items or k <= 0:
            return []
        n = len(self.items)
        avg = self._len_sum / n or 1.0
        q = set(tokenize(query))
        scored = []
        for it, tf in zip(self.items, self._tf):
            dl = sum(tf.values())
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[w] + 0.5) / (self._df[w] + 0.5))
                s += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / avg))
            if s > 0:
                scored.append((it, s))
        scored.sort(key=lambda x: (-x[1], -x[0].turn_index))
        return scored[:k]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class EmbeddingRetriever:
    """Dense retrieval: ``embed`` maps a list of texts to a list of vectors."""

    embed: Callable[[list[str]], list[list[float]]]
    items: list[MemoryItem] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def add(self, items: Sequence[MemoryItem]) -> None:
        items = list(items)
        if items:
            self.vectors.extend(self.embed([it.text for it in items]))
            self.items.extend(items)

    def search(self, query: str, k: int) -> list[tuple[MemoryItem, float]]:
        if not self.items or k <= 0:
            return []
        qv = self.embed([query])[0]
        scored = [(it, _cosine(qv, v)) for it, v in zip(self.items, self.vectors)]
        scored.sort(key=lambda x: (-x[1], -x[0].turn_index))
        return scored[:k]


def openai_embedder(model: str, base_url: str | None = None, api_key_env: str = "MODEL_API_KEY",
                    batch: int = 64) -> Callable[[list[str]], list[list[float]]]:
    """Embedding function backed by any OpenAI-compatible ``/embeddings`` endpoint."""
    import os

    from openai import OpenAI  # lazy: optional dependency

    client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"))

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            resp = client.embeddings.create(model=model, input=texts[i:i + batch])
            out.extend(d.embedding for d in resp.data)
        return out

    return embed


def message_text(msg: dict[str, Any]) -> str:
    parts = [str(msg.get("content") or "")]
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        parts.append(f"{fn.get('name', '')}({fn.get('arguments', '')})")
    return "\n".join(p for p in parts if p)


def render_turn(turn_index: int, round_: int, messages: list[dict[str, Any]], max_chars: int) -> str:
    """One memory item: the agent's reasoning, its tool call, and the start and end of the result.

    Long observations keep their head and tail, where commands echo what ran and where
    scores and tracebacks usually end up.
    """
    reasoning, call, result = "", "", ""
    for m in messages:
        if m.get("role") == "assistant":
            reasoning = str(m.get("content") or "")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function", {})
                args = fn.get("arguments", "")
                if not isinstance(args, str):
                    args = json.dumps(args, ensure_ascii=False, sort_keys=True)
                call = f"{fn.get('name', '')} {args}"
        else:
            result = str(m.get("content") or "")
    head = f"[round {round_}, turn {turn_index}]"
    if call:
        head += f" {call}"
    body = []
    if reasoning:
        body.append(f"thought: {reasoning}")
    if result:
        body.append(f"result: {result}")
    text = head + ("\n" + "\n".join(body) if body else "")
    if len(text) > max_chars:
        keep = max(0, max_chars - 40)
        text = text[: keep * 2 // 3] + "\n[... trimmed ...]\n" + text[-(keep - keep * 2 // 3):]
    return text


@dataclass
class RetrievalMemory:
    retriever: Retriever = field(default_factory=BM25Retriever)
    k: int = 4
    budget_tokens: int = 2_000
    max_item_chars: int = 1_200
    count_tokens: Callable[[str], int] = approx_tokens
    stored: set[int] = field(default_factory=set)

    def __len__(self) -> int:
        return len(self.stored)

    def add_turn(self, turn_index: int, round_: int, messages: list[dict[str, Any]]) -> bool:
        if turn_index in self.stored:
            return False
        text = render_turn(turn_index, round_, messages, self.max_item_chars)
        self.retriever.add([MemoryItem(turn_index, round_, text, self.count_tokens(text))])
        self.stored.add(turn_index)
        return True

    def retrieve(self, query: str, exclude: set[int] | None = None) -> list[MemoryItem]:
        """Top items for ``query`` that fit the token budget, returned in turn order."""
        exclude = exclude or set()
        hits = self.retriever.search(query, self.k + len(exclude))
        chosen, used = [], 0
        for it, _ in hits:
            if it.turn_index in exclude:
                continue
            if used + it.tokens > self.budget_tokens:
                continue
            chosen.append(it)
            used += it.tokens
            if len(chosen) >= self.k:
                break
        return sorted(chosen, key=lambda it: it.turn_index)

    @staticmethod
    def render(items: list[MemoryItem]) -> str:
        lines = ["[Retrieved memory] Earlier turns that are no longer in context, chosen by relevance "
                 "to your latest work. They are excerpts; rerun a command if you need its full output."]
        for it in items:
            lines.append(it.text)
        return "\n\n".join(lines)
