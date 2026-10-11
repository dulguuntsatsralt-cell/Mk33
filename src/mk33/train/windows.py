"""Offline RLC windowing (paper Section 3.2, "Train and test use the same contexts").

Each teacher trajectory is replayed through the same ``ContextBuilder`` the agent
uses. Every compaction event starts a new training window whose prefix is the
compacted context at that moment. Within a window the context only grows by
appending turns, so one sequence covers all of the window's turns. Loss applies
only to assistant turns inside the window, weighted by PALM.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from ..palm.weights import TurnWeight
from ..scaffold.context import SYSTEM_PROMPT, ContextBuilder, turn_messages
from ..scaffold.rlc import RLCConfig
from ..trajectory import Trajectory


@dataclass
class TrainingWindow:
    env_id: str
    window_index: int
    first_turn: int
    messages: list[dict[str, Any]] = field(default_factory=list)
    # One entry per message: loss weight for assistant targets, None for context.
    weights: list[float | None] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def weighted_turns(self) -> int:
        return sum(1 for w in self.weights if w)


def build_windows(
    traj: Trajectory,
    weights: list[TurnWeight],
    rlc: RLCConfig | None = None,
    count_tokens: Callable[[str], int] | None = None,
    system_prompt: str | None = None,
    drop_empty: bool = True,
) -> list[TrainingWindow]:
    wmap = {w.index: w.weight for w in weights}
    ctx = ContextBuilder(
        traj.task_statement, traj.pinned.get("skills", ""),
        system_prompt or traj.pinned.get("system") or SYSTEM_PROMPT, rlc, count_tokens,
    )
    windows: list[TrainingWindow] = []
    cur: TrainingWindow | None = None
    for turn in traj.turns:
        msgs, event = ctx.next_context()
        if cur is None or event is not None:
            cur = TrainingWindow(traj.env_id, len(windows), turn.index, list(msgs), [None] * len(msgs))
            windows.append(cur)
        tm = turn_messages(turn)
        cur.messages.extend(tm)
        cur.weights.extend([wmap.get(turn.index, 0.0)] + [None] * (len(tm) - 1))
        ctx.record(turn)
    if drop_empty:
        windows = [w for w in windows if w.weighted_turns > 0]
    return windows
