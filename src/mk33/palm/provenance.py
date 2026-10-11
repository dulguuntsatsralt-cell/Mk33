"""PALM full-credit set S+ (paper Section 3.3).

A submission at round t is *improving* if s~_t > B_{t-1} + eps, where B_0 = 0 (the
baseline on the normalized scale). A turn joins S+ if:

1. surviving edit:       an edit_file turn with at least one diff hunk present, line for
                         line, in the solution scored at a later improving submission;
2. improving submission: the submit call of an improving round;
3. supporting evidence:  an information turn whose observation or target is used by a
                         turn from rule 1 or 2 (shared path, URL, or identifier, or an
                         8-gram of the observation in the later turn's reasoning). One hop.
4. recovery:             the first edit after a regression or failed submission that
                         touches the file named in the error, when the agent regains its
                         pre-regression best within M rounds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..trajectory import Trajectory, Turn
from .canonical import canonical_action

_PATH = re.compile(r"(?:[\w.\-]+/)+[\w.\-]+|[\w\-]+\.(?:py|cpp|cc|h|hpp|yaml|yml|json|cfg|toml|sh|txt|csv|md)\b")
_URL = re.compile(r"https?://[^\s'\"<>)\]]+")
_IDENT = re.compile(r"\b(?:[A-Za-z_][A-Za-z0-9]*_[A-Za-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]+)\b")
_WORD = re.compile(r"[A-Za-z0-9_]+")
INFO_TOOLS = {"read_file", "bash", "web_search", "web_fetch", "wait_job"}


# ----------------------------------------------------------------- submissions
def improving_turns(traj: Trajectory, eps: float) -> dict[int, float]:
    """Map turn index of each improving submit -> previous best B_{t-1}."""
    best = 0.0
    out = {}
    for t in traj.turns:
        s = t.submission
        if t.tool != "submit" or s is None or not s.ok or s.normalized is None:
            continue
        if s.normalized > best + eps:
            out[t.index] = best
        best = max(best, s.normalized)
    return out


# ----------------------------------------------------------------------- hunks
@dataclass
class Hunk:
    path: str
    added: list[str]
    removed: list[str]


def parse_hunks(diff: str) -> list[Hunk]:
    hunks: list[Hunk] = []
    path = None
    cur: Hunk | None = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip().removeprefix("b/")
            cur = None
        elif line.startswith("--- "):
            continue
        elif line.startswith("@@"):
            cur = Hunk(path or "", [], [])
            hunks.append(cur)
        elif cur is not None:
            if line.startswith("+"):
                cur.added.append(line[1:])
            elif line.startswith("-"):
                cur.removed.append(line[1:])
    return [h for h in hunks if h.added or h.removed]


def _norm_lines(lines: list[str]) -> list[str]:
    return [l.rstrip() for l in lines if l.strip()]


def _contains_block(haystack: list[str], block: list[str]) -> bool:
    n = len(block)
    if n == 0:
        return False
    first = block[0]
    for i in range(len(haystack) - n + 1):
        if haystack[i] == first and haystack[i : i + n] == block:
            return True
    return False


def hunk_survives(h: Hunk, files: dict[str, str]) -> bool:
    content = files.get(h.path)
    if content is None:
        return False
    lines = _norm_lines(content.splitlines())
    added = _norm_lines(h.added)
    if added:
        return _contains_block(lines, added)
    removed = _norm_lines(h.removed)  # deletion-only hunk survives if the block stays gone
    return bool(removed) and not _contains_block(lines, removed)


# ------------------------------------------------------------------- evidence
def references(text: str) -> set[str]:
    return set(_PATH.findall(text)) | set(_URL.findall(text)) | {
        m for m in _IDENT.findall(text) if len(m) >= 6
    }


def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _WORD.findall(text.lower())
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def uses_evidence(info: Turn, user: Turn, stoplist: set[str], ngram: int = 8) -> bool:
    src = info.observation + "\n" + canonical_action(info.tool, info.args)
    dst = user.reasoning + "\n" + canonical_action(user.tool, user.args) + "\n" + (user.diff or "")
    shared = (references(src) & references(dst)) - stoplist
    if shared:
        return True
    return bool(_ngrams(info.observation, ngram) & _ngrams(user.reasoning, ngram))


# ------------------------------------------------------------------- recovery
def _error_file(sig: str | None, logs: str) -> str | None:
    for text in (sig or "", logs):
        m = re.search(r'File "([^"]+)"', text) or _PATH.search(text)
        if m:
            return m.group(1) if m.lastindex else m.group(0)
    return None


def _touches(turn: Turn, fname: str | None) -> bool:
    if fname is None:
        return True
    p = str(turn.args.get("path", ""))
    return p.endswith(fname) or fname.endswith(p) or p.split("/")[-1] == fname.split("/")[-1]


def recovery_turns(traj: Trajectory, eps: float, m_rounds: int = 3) -> set[int]:
    turns = traj.turns
    subs = [t for t in turns if t.tool == "submit" and t.submission is not None]
    out: set[int] = set()
    best = 0.0
    for k, st in enumerate(subs):
        s = st.submission
        assert s is not None
        prev_best = best
        regressed = (not s.ok) or (s.normalized is not None and s.normalized - prev_best < -eps)
        if s.ok and s.normalized is not None:
            best = max(best, s.normalized)
        if not regressed:
            continue
        later = subs[k + 1 : k + 1 + m_rounds]
        if not any(x.submission.ok and x.submission.normalized is not None
                   and x.submission.normalized >= prev_best for x in later):
            continue
        fname = _error_file(s.error_signature, s.logs) if not s.ok else None
        horizon = later[-1].index if later else len(turns)
        for t in turns[st.index + 1 : horizon + 1]:
            if t.tool == "edit_file" and t.diff and _touches(t, fname):
                out.add(t.index)
                break
    return out


# ------------------------------------------------------------------------ S+
@dataclass
class ProvenanceConfig:
    eps: float = 0.0
    m_rounds: int = 3
    evidence_window: int = 50
    ngram: int = 8


def provenance_set(traj: Trajectory, cfg: ProvenanceConfig | None = None) -> dict[int, str]:
    """Return {turn index: reason} for every turn in S+."""
    cfg = cfg or ProvenanceConfig()
    turns = traj.turns
    improving = improving_turns(traj, cfg.eps)
    reasons: dict[int, str] = {i: "improving_submission" for i in improving}

    improving_snaps = [(i, turns[i].submission.solution_files) for i in sorted(improving)]
    for t in turns:
        if t.tool != "edit_file" or not t.diff:
            continue
        hunks = parse_hunks(t.diff)
        for idx, files in improving_snaps:
            if idx > t.index and any(hunk_survives(h, files) for h in hunks):
                reasons.setdefault(t.index, "surviving_edit")
                break

    core = sorted(reasons)
    stop = references(traj.task_statement)
    for k in core:
        user = turns[k]
        for j in range(max(0, k - cfg.evidence_window), k):
            if j in reasons or turns[j].tool not in INFO_TOOLS:
                continue
            if uses_evidence(turns[j], user, stop, cfg.ngram):
                reasons[j] = "supporting_evidence"

    for i in recovery_turns(traj, cfg.eps, cfg.m_rounds):
        reasons.setdefault(i, "recovery")
    return reasons
