"""Round ledger (paper Section 3.2).

One entry per submission. The scaffold fills most fields; the agent writes only
``hypothesis`` and ``lesson`` through required arguments of ``submit``. Failed
rounds are never deleted, only merged once the ledger exceeds its token cap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..tokens import approx_tokens

_STOP = {
    "a", "an", "the", "to", "of", "and", "or", "with", "for", "in", "on", "will", "use", "using",
    "try", "trying", "by", "from", "is", "be", "that", "this", "it", "should", "can", "more", "less",
}


def strategy_family(hypothesis: str, n: int = 3) -> str:
    words = [w for w in re.findall(r"[a-z0-9]+", hypothesis.lower()) if w not in _STOP]
    return " ".join(words[:n]) or "misc"


def fmt_duration(s: float) -> str:
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"


@dataclass
class LedgerEntry:
    round: int
    wall_time_s: float = 0.0
    hypothesis: str = ""
    change: str = ""  # diff stat
    score: float | None = None
    normalized: float | None = None
    delta_vs_best: float | None = None
    error_signature: str | None = None
    lesson: str = ""
    family: str = ""
    extra: dict[str, str] = field(default_factory=dict)  # research: answer, confidence, evidence
    merged_rounds: list[int] = field(default_factory=list)  # non-empty for merged lines

    def __post_init__(self):
        if not self.family:
            self.family = strategy_family(self.hypothesis)

    def render(self) -> str:
        if self.merged_rounds:
            return f"R{','.join(map(str, self.merged_rounds))} [merged: {self.family}] {self.lesson}"
        parts = [f"R{self.round} [{fmt_duration(self.wall_time_s)}]"]
        if self.hypothesis:
            parts.append(f"hypothesis: {self.hypothesis}")
        if self.change:
            parts.append(f"change: {self.change}")
        if self.score is not None:
            s = f"score {self.score:.4g}"
            if self.normalized is not None:
                s += f" (norm {self.normalized:.3f}"
                if self.delta_vs_best is not None:
                    s += f", d_best {self.delta_vs_best:+.3f}"
                s += ")"
            parts.append(s)
        elif self.error_signature is None and not self.extra:
            parts.append("no score")
        if self.error_signature:
            parts.append(f"error: {self.error_signature}")
        for k, v in self.extra.items():
            parts.append(f"{k}: {v}")
        if self.lesson:
            parts.append(f"lesson: {self.lesson}")
        return " | ".join(parts)


class RoundLedger:
    def __init__(self, count_tokens: Callable[[str], int] = approx_tokens):
        self.entries: list[LedgerEntry] = []
        self.count_tokens = count_tokens

    def best(self) -> LedgerEntry | None:
        scored = [e for e in self.entries if e.normalized is not None and not e.merged_rounds]
        return max(scored, key=lambda e: e.normalized) if scored else None

    def best_normalized(self) -> float | None:
        vals = [e.normalized for e in self.entries if e.normalized is not None]
        return max(vals) if vals else None

    def add(self, entry: LedgerEntry) -> LedgerEntry:
        prev_best = self.best_normalized()
        if entry.normalized is not None and prev_best is not None:
            entry.delta_vs_best = entry.normalized - prev_best
        self.entries.append(entry)
        return entry

    def render(self) -> str:
        if not self.entries:
            return "Round ledger: (no submissions yet)"
        return "Round ledger:\n" + "\n".join(e.render() for e in self.entries)

    def tokens(self) -> int:
        return self.count_tokens(self.render())

    def compact(self, cap: int) -> int:
        """Merge oldest non-best entries, one line per strategy family, until under cap.

        Returns the number of merge operations performed.
        """
        merges = 0
        while self.tokens() > cap:
            best = self.best()
            cands = [e for e in self.entries if e is not best and not e.merged_rounds]
            if not cands:
                # Only merged lines left: merge the two oldest merged lines.
                merged = [e for e in self.entries if e.merged_rounds]
                if len(merged) < 2:
                    break
                group = merged[:2]
            else:
                fam = cands[0].family
                group = [e for e in cands if e.family == fam]
                if len(group) < 2:
                    group = cands[:2] if len(cands) >= 2 else group
                    if len(group) < 2:
                        break
            self._merge(group)
            merges += 1
        return merges

    def _merge(self, group: list[LedgerEntry]) -> None:
        rounds = sorted(r for e in group for r in (e.merged_rounds or [e.round]))
        fams = sorted({e.family for e in group})
        norms = [e.normalized for e in group if e.normalized is not None]
        errs = sum(1 for e in group if e.error_signature)
        lessons = [e.lesson for e in group if e.lesson]
        summary = []
        if norms:
            summary.append(f"best norm {max(norms):.3f}")
        if errs:
            summary.append(f"{errs} errored")
        if lessons:
            summary.append("lessons: " + "; ".join(lessons[-3:])[:300])
        merged = LedgerEntry(
            round=rounds[0], family=" / ".join(fams)[:80], lesson=", ".join(summary),
            normalized=None, merged_rounds=rounds,
        )
        first = self.entries.index(group[0])
        for e in group:
            self.entries.remove(e)
        self.entries.insert(first, merged)
