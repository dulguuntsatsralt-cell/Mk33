"""Process metrics (paper Section 4, "Process metrics").

| metric              | definition                                                        |
|---------------------|-------------------------------------------------------------------|
| productive rounds T*| rounds before the first run of M=5 rounds with no improving submit|
| mean gain r_bar     | mean normalized gain over improving rounds                        |
| best-so-far AUC     | area under the normalized best-so-far curve over the time budget  |
| recovery rate       | share of regressions followed by a new best within 3 rounds       |
| wasted-turn share   | share of turns flagged R, N, or P by the PALM rules               |

Failed submissions count as non-improving rounds. The best starts at 0 (baseline).
"""

from __future__ import annotations

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


@dataclass
class ProcessMetrics:
    rounds: int
    productive_rounds: int
    mean_gain: float
    best: float | None
    auc: float | None
    recovery_rate: float | None
    wasted_turn_share: float


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
    )


def aggregate(metrics: list[ProcessMetrics]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if not metrics:
        return out
    for k in asdict(metrics[0]):
        vals = [getattr(m, k) for m in metrics if getattr(m, k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    return out
