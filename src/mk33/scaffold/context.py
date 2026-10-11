"""Context construction shared by the live agent and the offline training replay.

Both call ``ContextBuilder`` with the same turns in the same order, so the context
the model trains on is exactly the context the scaffold builds at inference.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..trajectory import Turn
from .ledger import LedgerEntry, RoundLedger
from .rlc import CompactionEvent, RLCConfig, RLCContext, TurnMessages
from .tools import diff_stat

Message = dict[str, Any]

SYSTEM_PROMPT = """You are an agent that improves a solution over many rounds.

Each round: investigate, change the solution, then call `submit` to score it. The score and
logs come back to you. Use `run_job` and `wait_job` for long commands instead of polling.
Every `submit` needs a `hypothesis` (what you tried and why) and a one-line `lesson`.

A round ledger lists every past round, including failures. Older turns may be compacted
into the ledger when the context is full; the ledger is the record of what you tried.
Keep improving until the budget runs out. Rounds that lower the score are normal; diagnose
them and try again."""


def pinned_messages(system_prompt: str, task: str, skills: str, best_note: str) -> list[Message]:
    user = f"# Task\n{task}"
    if skills:
        user += f"\n\n# Skills\n{skills}"
    user += f"\n\n# Current best\n{best_note}"
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


def best_note(ledger: RoundLedger) -> str:
    b = ledger.best()
    if b is None or b.score is None:
        return "No scored submission yet."
    return f"Round {b.round}: score {b.score:.6g} (normalized {b.normalized:.4f})."


def call_id(turn_index: int) -> str:
    return f"call_{turn_index}"


def turn_messages(turn: Turn) -> list[Message]:
    assistant: Message = {"role": "assistant", "content": turn.reasoning}
    if turn.tool == "none":
        return [assistant, {"role": "user", "content": turn.observation}]
    cid = call_id(turn.index)
    assistant["tool_calls"] = [{"id": cid, "type": "function", "function": {
        "name": turn.tool, "arguments": json.dumps(turn.args, ensure_ascii=False, sort_keys=True)}}]
    return [assistant, {"role": "tool", "tool_call_id": cid, "content": turn.observation}]


class ContextBuilder:
    def __init__(self, task: str, skills: str = "", system_prompt: str = SYSTEM_PROMPT,
                 rlc: RLCConfig | None = None, count_tokens: Callable[[str], int] | None = None):
        kw = {"count_tokens": count_tokens} if count_tokens else {}
        self.task, self.skills, self.system_prompt = task, skills, system_prompt
        self.ledger = RoundLedger(**kw)
        self.rlc = RLCContext(rlc or RLCConfig(), **kw)
        self.turns: list[TurnMessages] = []
        self.round = 1
        self._round_diffs: list[str] = []

    def _pinned(self) -> list[Message]:
        return pinned_messages(self.system_prompt, self.task, self.skills, best_note(self.ledger))

    def next_context(self) -> tuple[list[Message], CompactionEvent | None]:
        return self.rlc.build(self._pinned, self.ledger, self.turns, self.round)

    def make_entry(self, turn: Turn) -> LedgerEntry | None:
        """Ledger entry this submit turn will produce (delta computed, not yet added)."""
        s = turn.submission
        if turn.tool != "submit" or s is None:
            return None
        e = LedgerEntry(
            round=s.round, wall_time_s=turn.elapsed_s, hypothesis=s.hypothesis,
            change=diff_stat(self._round_diffs + ([turn.diff] if turn.diff else [])),
            score=s.score if s.ok else None, normalized=s.normalized if s.ok else None,
            error_signature=None if s.ok else (s.error_signature or "failed"),
            lesson=s.lesson, family=str(turn.args.get("family", "") or ""),
        )
        prev = self.ledger.best_normalized()
        if e.normalized is not None and prev is not None:
            e.delta_vs_best = e.normalized - prev
        return e

    def record(self, turn: Turn) -> None:
        self.turns.append(TurnMessages(turn.index, turn.round, turn_messages(turn)))
        if turn.tool == "edit_file" and turn.diff:
            self._round_diffs.append(turn.diff)
        entry = self.make_entry(turn)
        if entry is not None:
            self.ledger.add(entry)
            self._round_diffs = []
            self.round = turn.submission.round + 1
