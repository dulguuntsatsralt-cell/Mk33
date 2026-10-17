"""Round Ledger Compaction (paper Section 3.2).

The context is always assembled in this order:

    pinned (system prompt, task, skills, current best)
    ledger (one line per round, failures included)
    compaction note (if any turns were compacted)
    verbatim turns of the most recent rounds

When the context exceeds ``theta * window`` tokens, every verbatim turn older than
the last ``keep_rounds`` rounds is dropped; its information survives only in the
ledger. If the ledger itself exceeds ``ledger_cap``, its oldest non-best entries
are merged per strategy family.

``mode`` supports the A2 ablation and the memory study:
    "rlc"           the method above
    "truncate"      no ledger; drop the oldest turns until under theta * window
    "retrieve"      like truncate, but dropped turns go into a retrieval memory and the
                    most relevant ones are placed back in the prefix (see ``memory.py``)
    "rlc+retrieve"  the round ledger plus retrieval over the compacted turns
    "none"          no management (only valid when runs fit in the window)

With retrieval, the retrieved block is refreshed at every compaction and, when
``retrieve_refresh == "round"``, at the start of every round. A refresh changes the
prefix, so it is reported as an event (``kind="refresh"``) just like a compaction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..tokens import approx_tokens, message_tokens
from .ledger import RoundLedger
from .memory import (BM25Retriever, EmbeddingRetriever, MemoryItem, RetrievalMemory, Retriever,
                     message_text, openai_embedder)

Message = dict[str, Any]
PinnedFn = Callable[[], list[Message]]


@dataclass
class TurnMessages:
    turn_index: int
    round: int
    messages: list[Message]  # assistant (with tool call) + tool result


@dataclass
class CompactionEvent:
    first_kept_turn: int  # first verbatim turn index after the cut
    cutoff_round: int  # rounds < cutoff_round are now ledger-only
    ledger_merges: int = 0
    kind: str = "rlc"
    retrieved: list[int] = field(default_factory=list)  # turn indices placed back by retrieval


@dataclass
class RLCConfig:
    window: int = 128_000
    theta: float = 0.75
    keep_rounds: int = 2
    ledger_cap: int = 16_000
    mode: str = "rlc"
    # Retrieval memory (modes "retrieve" and "rlc+retrieve")
    retrieve_k: int = 4
    retrieve_budget: int = 2_000  # tokens reserved for the retrieved block
    retrieve_item_chars: int = 1_200
    retrieve_refresh: str = "round"  # "round" or "compaction"
    retriever: str = "bm25"  # "bm25" or "embed"
    embed_model: str = ""
    embed_base_url: str = ""

    @property
    def uses_ledger(self) -> bool:
        return self.mode in ("rlc", "rlc+retrieve")

    @property
    def uses_memory(self) -> bool:
        return self.mode in ("retrieve", "rlc+retrieve")


MODES = ("rlc", "truncate", "retrieve", "rlc+retrieve", "none")


def make_retriever(cfg: RLCConfig) -> Retriever:
    if cfg.retriever == "bm25":
        return BM25Retriever()
    if cfg.retriever == "embed":
        if not cfg.embed_model:
            raise ValueError("retriever 'embed' needs embed_model")
        return EmbeddingRetriever(openai_embedder(cfg.embed_model, cfg.embed_base_url or None))
    raise ValueError(f"unknown retriever {cfg.retriever}")


@dataclass
class RLCContext:
    """Builds the model context each turn.

    The prefix (pinned messages + ledger + retrieved memory) is frozen between events,
    so the context only grows by appending turns. This keeps KV caches valid and makes
    every segment between two events a single training window (see ``mk33.train``).
    New ledger entries reach the model immediately through the ``submit`` observation.
    """

    cfg: RLCConfig = field(default_factory=RLCConfig)
    count_tokens: Callable[[str], int] = approx_tokens
    cutoff_round: int = 0  # verbatim turns with round < cutoff_round are dropped
    dropped_turns: set[int] = field(default_factory=set)
    events: list[CompactionEvent] = field(default_factory=list)
    memory: RetrievalMemory | None = None
    _prefix: list[Message] | None = None
    _refresh_round: int = 0

    def __post_init__(self):
        if self.cfg.mode not in MODES:
            raise ValueError(f"unknown RLC mode {self.cfg.mode}")
        if self.cfg.uses_memory and self.memory is None:
            self.memory = RetrievalMemory(
                make_retriever(self.cfg), k=self.cfg.retrieve_k, budget_tokens=self.cfg.retrieve_budget,
                max_item_chars=self.cfg.retrieve_item_chars, count_tokens=self.count_tokens)

    def _tok(self, msgs: list[Message]) -> int:
        return sum(message_tokens(m, self.count_tokens) for m in msgs)

    def _visible(self, turns: list[TurnMessages]) -> list[TurnMessages]:
        return [t for t in turns if t.round >= self.cutoff_round and t.turn_index not in self.dropped_turns]

    def _make_prefix(self, pinned_fn: PinnedFn, ledger: RoundLedger | None,
                     retrieved: list[MemoryItem] | None = None) -> list[Message]:
        msgs = list(pinned_fn())
        if ledger is not None and self.cfg.uses_ledger and ledger.entries:
            msgs.append({"role": "user", "content": ledger.render()})
            if self.cutoff_round > 1:
                msgs.append({
                    "role": "user",
                    "content": f"[Context note] Turns from rounds 1..{self.cutoff_round - 1} were "
                    "compacted. Their results are in the round ledger above.",
                })
        if retrieved:
            msgs.append({"role": "user", "content": RetrievalMemory.render(retrieved)})
        return msgs

    def _assemble(self, turns: list[TurnMessages], prefix: list[Message] | None = None) -> list[Message]:
        msgs = list(self._prefix if prefix is None else prefix)
        for t in self._visible(turns):
            msgs.extend(t.messages)
        return msgs

    def _store_hidden(self, turns: list[TurnMessages]) -> None:
        if self.memory is None:
            return
        visible = {t.turn_index for t in self._visible(turns)}
        for t in turns:
            if t.turn_index not in visible:
                self.memory.add_turn(t.turn_index, t.round, t.messages)

    def _query(self, pinned_fn: PinnedFn, vis: list[TurnMessages]) -> str:
        """What the agent is working on now: its last two turns, else the task."""
        text = "\n".join(message_text(m) for t in vis[-2:] for m in t.messages)
        if text.strip():
            return text
        return "\n".join(str(m.get("content") or "") for m in pinned_fn() if m.get("role") == "user")

    def build(
        self,
        pinned_fn: PinnedFn,
        ledger: RoundLedger | None,
        turns: list[TurnMessages],
        current_round: int,
    ) -> tuple[list[Message], CompactionEvent | None]:
        cfg = self.cfg
        if self._prefix is None:
            self._prefix = self._make_prefix(pinned_fn, ledger)
            self._refresh_round = current_round
        msgs = self._assemble(turns)
        limit = int(cfg.theta * cfg.window)
        over = cfg.mode != "none" and self._tok(msgs) > limit
        refresh = (cfg.uses_memory and self.memory is not None and len(self.memory) > 0
                   and cfg.retrieve_refresh == "round" and current_round > self._refresh_round)
        if not over and not refresh:
            return msgs, None

        vis = self._visible(turns)
        merges = 0
        compacted = False
        if over and cfg.uses_ledger:
            new_cutoff = current_round - cfg.keep_rounds + 1
            ledger_over = ledger is not None and ledger.tokens() > cfg.ledger_cap
            over_window = self._tok(msgs) > cfg.window
            if new_cutoff > self.cutoff_round or ledger_over or over_window:
                compacted = True
                self.cutoff_round = max(self.cutoff_round, new_cutoff)
                if ledger_over:
                    merges = ledger.compact(cfg.ledger_cap)
        elif over:  # truncate / retrieve: drop the oldest turns, keeping room for the retrieved block
            compacted = True
            reserve = cfg.retrieve_budget if cfg.uses_memory else 0
            base = self._make_prefix(pinned_fn, None)
            while self._tok(self._assemble(turns, base)) > limit - reserve and len(vis) > 1:
                self.dropped_turns.add(vis[0].turn_index)
                vis = vis[1:]
        if not compacted and not refresh:
            return msgs, None  # nothing to compact yet; still fits the window

        self._store_hidden(turns)
        vis = self._visible(turns)
        retrieved: list[MemoryItem] = []
        if self.memory is not None and len(self.memory):
            retrieved = self.memory.retrieve(self._query(pinned_fn, vis), {t.turn_index for t in vis})
        self._prefix = self._make_prefix(pinned_fn, ledger if cfg.uses_ledger else None, retrieved)
        self._refresh_round = current_round
        msgs = self._assemble(turns)
        # Hard guard: a single huge round can still overflow the full window.
        while self._tok(msgs) > cfg.window and len(vis) > 1:
            self.dropped_turns.add(vis[0].turn_index)
            vis = vis[1:]
            msgs = self._assemble(turns)
        self._store_hidden(turns)

        # First verbatim turn kept; if none survive, the next turn to be recorded.
        first = vis[0].turn_index if vis else (turns[-1].turn_index + 1 if turns else 0)
        kind = cfg.mode if compacted else "refresh"
        event = CompactionEvent(first, self.cutoff_round, merges, kind, [it.turn_index for it in retrieved])
        self.events.append(event)
        return msgs, event
