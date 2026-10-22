"""Process metrics (paper Section 4, "Process metrics").

| metric              | definition                                                        |
|---------------------|-------------------------------------------------------------------|
| productive rounds T*| rounds before the first run of M=5 rounds with no improving submit|
| mean gain r_bar     | mean normalized gain over improving rounds                        |
| best-so-far AUC     | area under the normalized best-so-far curve over the time budget  |
| recovery rate       | share of regressions followed by a new best within 3 rounds       |
| wasted-turn share   | share of turns flagged R, N, or P by the PALM rules               |
| re-try rate         | share of rounds (after the first) that repeat a known dead end:   |
|                     | a solution identical to an earlier submission, or an idea that    |
|                     | closely matches an earlier round that did not improve the score   |

Failed submissions count as non-improving rounds. The best starts at 0 (baseline).
The re-try rate is new in Mk33 (it is not an AREX-2 metric): it measures forgetting
directly, since an agent that remembers what failed should not try it again.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

from ..palm.rules import is_null, is_polling, is_redundant
from ..trajectory import Trajectory


def _round_scores(traj: Trajectory) -> list[tuple[float | None, float]]:
    """(normalized score or None if failed, elapsed seconds) per submit, in order."""
    out = []
    for t in traj.turns:
        if t.tool == "submit" and t.submission is not None:
            s = t.submission
            out.append((s.normalized if s.ok else None, t.elapsed_s))
    return out


def productive_rounds(scores: list[float | None], m: int = 5, eps: float = 0.0) -> int:
    best, streak = 0.0, 0
    for r, s in enumerate(scores, start=1):
        if s is not None and s > best + eps:
            best, streak = s, 0
        else:
            streak += 1
            if streak == m:
                return r - m
    return len(scores)


def mean_gain(scores: list[float | None], eps: float = 0.0) -> float:
    best, gains = 0.0, []
    for s in scores:
        if s is not None and s > best + eps:
            gains.append(s - best)
            best = s
    return sum(gains) / len(gains) if gains else 0.0


def best_so_far_auc(points: list[tuple[float | None, float]], budget_s: float) -> float:
    """Step-function area of best-so-far over [0, budget], divided by budget."""
    if budget_s <= 0:
        raise ValueError("budget must be positive")
    best, last_t, area = 0.0, 0.0, 0.0
    for s, t in sorted(points, key=lambda p: p[1]):
        t = min(t, budget_s)
        area += best * (t - last_t)
        last_t = t
        if s is not None:
            best = max(best, s)
    area += best * (budget_s - last_t)
    return area / budget_s


def recovery_rate(scores: list[float | None], eps: float = 0.0, within: int = 3) -> float | None:
    best, regressions, recovered = 0.0, 0, 0
    for k, s in enumerate(scores):
        prev = best
        if s is None or s < prev - eps:
            regressions += 1
            if any(x is not None and x > prev + eps for x in scores[k + 1 : k + 1 + within]):
                recovered += 1
        if s is not None:
            best = max(best, s)
    return recovered / regressions if regressions else None


def wasted_turn_share(traj: Trajectory) -> float:
    turns = traj.turns
    if not turns:
        return 0.0
    flagged = sum(
        1 for i in range(len(turns))
        if is_redundant(turns, i) or is_null(turns, i) or is_polling(turns, i)
    )
    return flagged / len(turns)


def _solution_hash(files: dict[str, str]) -> str | None:
    if not files:
        return None
    h = hashlib.sha256()
    for path in sorted(files):
        # Whitespace-only edits do not make a new solution.
        h.update(path.encode() + b"\0" + " ".join(files[path].split()).encode() + b"\0")
    return h.hexdigest()


def _idea_words(text: str) -> set[str]:
    from ..scaffold.memory import tokenize

    return set(tokenize(text))


def retry_rate(traj: Trajectory, eps: float = 0.0, idea_jaccard: float = 0.6,
               min_idea_words: int = 3) -> float | None:
    """Share of rounds 2..n that repeat a known dead end.

    A round is a re-try if (a) the scored solution is identical to one already submitted
    in an earlier round (an exact repeat can never be new information), or (b) its
    ``hypothesis`` shares at least ``idea_jaccard`` of its words with the hypothesis of an
    earlier round that failed or did not beat the best so far. Hypotheses shorter than
    ``min_idea_words`` words are too vague to compare and only count through (a).
    None when the run has fewer than two rounds.
    """
    subs = [t.submission for t in traj.turns if t.tool == "submit" and t.submission is not None]
    if len(subs) < 2:
        return None
    seen: set[str] = set()
    dead_ideas: list[set[str]] = []
    best, retries = 0.0, 0
    for k, s in enumerate(subs):
        h = _solution_hash(s.solution_files)
        words = _idea_words(s.hypothesis)
        if k > 0:
            same_solution = h is not None and h in seen
            same_idea = len(words) >= min_idea_words and any(
                len(words & d) / len(words | d) >= idea_jaccard for d in dead_ideas)
            retries += same_solution or same_idea
        if h is not None:
            seen.add(h)
        improved = s.ok and s.normalized is not None and s.normalized > best + eps
        if improved:
            best = s.normalized
        elif len(words) >= min_idea_words:
            dead_ideas.append(words)
    return retries / (len(subs) - 1)


@dataclass
class ProcessMetrics:
    rounds: int
    productive_rounds: int
    mean_gain: float
    best: float | None
    auc: float | None
    recovery_rate: float | None
    wasted_turn_share: float
    retry_rate: float | None = None


def process_metrics(traj: Trajectory, budget_s: float | None = None, m: int = 5) -> ProcessMetrics:
    eps = float(traj.metadata.get("eps", 0.0))
    pts = _round_scores(traj)
    scores = [s for s, _ in pts]
    return ProcessMetrics(
        rounds=len(scores),
        productive_rounds=productive_rounds(scores, m, eps),
        mean_gain=mean_gain(scores, eps),
        best=traj.best_normalized(),
        auc=best_so_far_auc(pts, budget_s) if budget_s else None,
        recovery_rate=recovery_rate(scores, eps),
        wasted_turn_share=wasted_turn_share(traj),
        retry_rate=retry_rate(traj, eps),
    )


def aggregate(metrics: list[ProcessMetrics]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if not metrics:
        return out
    for k in asdict(metrics[0]):
        vals = [getattr(m, k) for m in metrics if getattr(m, k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    return out
