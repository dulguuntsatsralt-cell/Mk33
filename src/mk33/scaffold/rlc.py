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

``mode`` supports the A2 ablation:
    "rlc"       the method above
    "truncate"  no ledger; drop the oldest turns until under theta * window
    "none"      no management (only valid when runs fit in the window)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..tokens import approx_tokens, message_tokens
from .ledger import RoundLedger

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


@dataclass
class RLCConfig:
    window: int = 128_000
    theta: float = 0.75
    keep_rounds: int = 2
    ledger_cap: int = 16_000
    mode: str = "rlc"


@dataclass
class RLCContext:
    """Builds the model context each turn.

    The prefix (pinned messages + ledger) is frozen between compaction events, so the
    context only grows by appending turns. This keeps KV caches valid and makes every
    segment between two events a single training window (see ``mk33.train``).
    New ledger entries reach the model immediately through the ``submit`` observation.
    """

    cfg: RLCConfig = field(default_factory=RLCConfig)
    count_tokens: Callable[[str], int] = approx_tokens
    cutoff_round: int = 0  # verbatim turns with round < cutoff_round are dropped
    dropped_turns: set[int] = field(default_factory=set)
    events: list[CompactionEvent] = field(default_factory=list)
    _prefix: list[Message] | None = None

    def _tok(self, msgs: list[Message]) -> int:
        return sum(message_tokens(m, self.count_tokens) for m in msgs)

    def _visible(self, turns: list[TurnMessages]) -> list[TurnMessages]:
        return [t for t in turns if t.round >= self.cutoff_round and t.turn_index not in self.dropped_turns]

    def _make_prefix(self, pinned_fn: PinnedFn, ledger: RoundLedger | None) -> list[Message]:
        msgs = list(pinned_fn())
        if ledger is not None and self.cfg.mode == "rlc" and ledger.entries:
            msgs.append({"role": "user", "content": ledger.render()})
            if self.cutoff_round > 1:
                msgs.append({
                    "role": "user",
                    "content": f"[Context note] Turns from rounds 1..{self.cutoff_round - 1} were "
                    "compacted. Their results are in the round ledger above.",
                })
        return msgs

    def _assemble(self, turns: list[TurnMessages]) -> list[Message]:
        msgs = list(self._prefix or [])
        for t in self._visible(turns):
            msgs.extend(t.messages)
        return msgs

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
        msgs = self._assemble(turns)
        limit = int(cfg.theta * cfg.window)
        if cfg.mode == "none" or self._tok(msgs) <= limit:
            return msgs, None
        if cfg.mode not in ("rlc", "truncate"):
            raise ValueError(f"unknown RLC mode {cfg.mode}")

        vis = self._visible(turns)
        merges = 0
        if cfg.mode == "rlc":
            new_cutoff = current_round - cfg.keep_rounds + 1
            ledger_over = ledger is not None and ledger.tokens() > cfg.ledger_cap
            over_window = self._tok(msgs) > cfg.window
            if new_cutoff <= self.cutoff_round and not ledger_over and not over_window:
                return msgs, None  # nothing to compact yet; still fits the window
            self.cutoff_round = max(self.cutoff_round, new_cutoff)
            if ledger_over:
                merges = ledger.compact(cfg.ledger_cap)
            self._prefix = self._make_prefix(pinned_fn, ledger)
            vis = self._visible(turns)
            msgs = self._assemble(turns)
            # Hard guard: a single huge round can still overflow the full window.
            while self._tok(msgs) > cfg.window and len(vis) > 1:
                self.dropped_turns.add(vis[0].turn_index)
                vis = vis[1:]
                msgs = self._assemble(turns)
        else:  # truncate: no ledger, drop oldest turns
            while self._tok(msgs) > limit and len(vis) > 1:
                self.dropped_turns.add(vis[0].turn_index)
                vis = vis[1:]
                msgs = self._assemble(turns)
            self._prefix = self._make_prefix(pinned_fn, None)
            msgs = self._assemble(turns)

        # First verbatim turn kept; if none survive, the next turn to be recorded.
        first = vis[0].turn_index if vis else (turns[-1].turn_index + 1 if turns else 0)
        event = CompactionEvent(first, self.cutoff_round, merges, cfg.mode)
        self.events.append(event)
        return msgs, event
