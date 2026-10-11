"""Trajectory acceptance (paper Section 3.4).

A trajectory is accepted if all hold:
  1. best normalized score >= tau_d   (tau_ml = 0.9, tau_judge = 0.95)
  2. at least ``min_rounds`` scored rounds
  3. at least one improving round after the first scored round
  4. no process-audit flag
Individual rounds are never filtered. At most ``max_per_env`` trajectories are kept
per environment, preferring higher best scores.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ..envs.audit import AuditReport
from ..trajectory import Trajectory


@dataclass
class SelectionConfig:
    tau: dict[str, float] = field(default_factory=lambda: {"ml": 0.9, "judge": 0.95})
    min_rounds: int = 3
    require_iteration: bool = True
    max_per_env: int = 2


@dataclass
class Decision:
    accepted: bool
    reasons: list[str]
    best: float | None


def _improved_after_first(traj: Trajectory, eps: float) -> bool:
    scores = [s.normalized for s in traj.scored_submissions()]
    if len(scores) < 2:
        return False
    best = scores[0]
    for s in scores[1:]:
        if s > best + eps:
            return True
        best = max(best, s)
    return False


def accept(traj: Trajectory, cfg: SelectionConfig | None = None,
           audit: AuditReport | None = None) -> Decision:
    cfg = cfg or SelectionConfig()
    reasons = []
    best = traj.best_normalized()
    tau = cfg.tau.get(traj.domain)
    if tau is None:
        reasons.append(f"no threshold for domain {traj.domain}")
    elif best is None or best < tau:
        reasons.append(f"best {best} < tau {tau}")
    n = len(traj.scored_submissions())
    if n < cfg.min_rounds:
        reasons.append(f"{n} scored rounds < {cfg.min_rounds}")
    if cfg.require_iteration and not _improved_after_first(traj, float(traj.metadata.get("eps", 0.0))):
        reasons.append("no improving round after the first")
    if audit is not None and not audit.passed:
        reasons.append("audit: " + ", ".join(f.check for f in audit.flags))
    return Decision(not reasons, reasons, best)


def select(trajs: list[Trajectory], cfg: SelectionConfig | None = None,
           audits: dict[int, AuditReport] | None = None):
    """Return (accepted, decisions) with the per-environment cap applied."""
    cfg = cfg or SelectionConfig()
    audits = audits or {}
    decisions = [accept(t, cfg, audits.get(i)) for i, t in enumerate(trajs)]
    by_env: dict[str, list[int]] = defaultdict(list)
    for i, (t, d) in enumerate(zip(trajs, decisions)):
        if d.accepted:
            by_env[t.env_id].append(i)
    keep = set()
    for env, idx in by_env.items():
        idx.sort(key=lambda i: decisions[i].best or 0.0, reverse=True)
        keep.update(idx[: cfg.max_per_env])
        for i in idx[cfg.max_per_env :]:
            decisions[i] = Decision(False, [f"per-env cap {cfg.max_per_env}"], decisions[i].best)
    return [trajs[i] for i in sorted(keep)], decisions
