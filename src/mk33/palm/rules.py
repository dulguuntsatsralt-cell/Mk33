"""PALM zero-weight rules (paper Section 3.3).

R(i)  redundant:  a near-identical canonical action within the last W turns
                  (Jaccard of 5-gram shingles >= threshold) and no workspace change
                  since, including by turn i itself. Status checks are handled by P,
                  not R, because a repeated status check can return new information.
N(i)  null:       the normalized observation is empty, or identical to the last
                  observation from the same tool on the same target, with the
                  workspace unchanged.
P(i)  polling:    a status-only call that adds no new log lines since the last
                  status call on the same target.
"""

from __future__ import annotations

import re

from ..trajectory import Turn
from .canonical import canonical_action, jaccard, normalize_observation, shingles

_STATUS_BASH = re.compile(
    r"^\s*(?:sleep\s+\d+\s*(?:&&|;)\s*)?"
    r"(?:ps\b|pgrep\b|nvidia-smi\b|squeue\b|top\s+-b|watch\b|"
    r"tail\b.*\.(?:log|out|err|txt)\b|cat\b.*\.(?:log|out|err)\b|"
    r"ls\b.*\b(?:ckpt|checkpoints?|outputs?|logs?|runs?)\b|sleep\s+\d+\s*$)"
)
_EXIT = re.compile(r"^\[exit -?\d+\]\s*")
_EFFECT_TOOLS = {"edit_file", "submit", "run_job", "answer"}


def target_of(turn: Turn) -> str:
    a = turn.args
    for k in ("path", "url", "job_id", "query", "command"):
        if k in a:
            return f"{k}={a[k]}"
    return ""


def is_status_call(turn: Turn) -> bool:
    if turn.tool == "wait_job":
        return True
    if turn.tool == "bash":
        return bool(_STATUS_BASH.match(str(turn.args.get("command", ""))))
    return False


def _hash_before(turns: list[Turn], i: int) -> str:
    return turns[i - 1].workspace_hash if i > 0 else ""


def _unchanged_since(turns: list[Turn], j: int, i: int) -> bool:
    """True if the workspace after turn j equals the workspace before turn i."""
    return bool(turns[j].workspace_hash) and turns[j].workspace_hash == _hash_before(turns, i)


def is_redundant(turns: list[Turn], i: int, window: int = 20, threshold: float = 0.9) -> bool:
    t = turns[i]
    if t.tool in ("none",) or is_status_call(t):
        return False
    if i > 0 and t.workspace_hash != _hash_before(turns, i):
        return False  # the turn changed the workspace, so it was not a no-op
    sh_i = shingles(canonical_action(t.tool, t.args))
    for j in range(max(0, i - window), i):
        u = turns[j]
        if u.tool != t.tool:
            continue
        if jaccard(sh_i, shingles(canonical_action(u.tool, u.args))) >= threshold and _unchanged_since(turns, j, i):
            return True
    return False


def _obs_body(t: Turn) -> str:
    return normalize_observation(_EXIT.sub("", t.observation))


def is_null(turns: list[Turn], i: int) -> bool:
    t = turns[i]
    if t.tool in _EFFECT_TOOLS:
        return False
    unchanged_now = (t.workspace_hash == _hash_before(turns, i)) if i > 0 else False
    body = _obs_body(t)
    if not body:
        return unchanged_now or t.tool == "none"
    tgt = target_of(t)
    for j in range(i - 1, -1, -1):
        u = turns[j]
        if u.tool == t.tool and target_of(u) == tgt:
            return _obs_body(u) == body and _unchanged_since(turns, j, i)
    return False


def _status_lines(t: Turn) -> set[str]:
    lines = set(normalize_observation(_EXIT.sub("", t.observation)).splitlines())
    lines = {l for l in lines if not l.startswith("[job ")}
    lines.discard("(no new output)")
    return lines


def is_polling(turns: list[Turn], i: int) -> bool:
    t = turns[i]
    if not is_status_call(t):
        return False
    lines = _status_lines(t)
    if t.tool == "wait_job":
        # wait_job already returns only new log text.
        return t.timed_out and not lines
    if not lines:
        return True
    tgt = target_of(t)
    for j in range(i - 1, -1, -1):
        u = turns[j]
        if is_status_call(u) and u.tool == t.tool and target_of(u) == tgt:
            return not (lines - _status_lines(u))
    return False
