"""PALM turn weights (paper Section 3.3).

    w_i = 0       if R(i) or N(i) or P(i)
          1       if i in S+
          lambda  otherwise

``mode`` supports ablation A1:
    "palm"   the rule above
    "arex"   zero-weight rules only; every other assistant turn gets 1 (lambda = 1)
    "all"    every assistant turn gets 1 (no masking)
    "labels" weights from external labels (e.g. an LLM judge): {index: weight}
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..trajectory import Trajectory
from .provenance import ProvenanceConfig, provenance_set
from .rules import is_null, is_polling, is_redundant


@dataclass
class PALMConfig:
    mode: str = "palm"
    lam: float = 0.5
    window: int = 20
    jaccard: float = 0.9
    m_rounds: int = 3
    evidence_window: int = 50
    ngram: int = 8
    eps: float | None = None  # None -> trajectory.metadata["eps"] or 0.0
    labels: dict[int, float] = field(default_factory=dict)


@dataclass
class TurnWeight:
    index: int
    weight: float
    reasons: list[str]


def compute_weights(traj: Trajectory, cfg: PALMConfig | None = None) -> list[TurnWeight]:
    cfg = cfg or PALMConfig()
    turns = traj.turns
    if cfg.mode == "all":
        return [TurnWeight(t.index, 1.0, ["all"]) for t in turns]
    if cfg.mode == "labels":
        return [TurnWeight(t.index, float(cfg.labels.get(t.index, 0.0)), ["label"]) for t in turns]

    eps = cfg.eps if cfg.eps is not None else float(traj.metadata.get("eps", 0.0))
    splus = {}
    if cfg.mode == "palm":
        splus = provenance_set(traj, ProvenanceConfig(eps, cfg.m_rounds, cfg.evidence_window, cfg.ngram))
    elif cfg.mode != "arex":
        raise ValueError(f"unknown PALM mode {cfg.mode}")

    out = []
    for pos, t in enumerate(turns):
        zero = []
        if is_redundant(turns, pos, cfg.window, cfg.jaccard):
            zero.append("R")
        if is_null(turns, pos):
            zero.append("N")
        if is_polling(turns, pos):
            zero.append("P")
        if zero:
            out.append(TurnWeight(t.index, 0.0, zero))
        elif cfg.mode == "arex":
            out.append(TurnWeight(t.index, 1.0, ["arex"]))
        elif t.index in splus:
            out.append(TurnWeight(t.index, 1.0, [splus[t.index]]))
        else:
            out.append(TurnWeight(t.index, cfg.lam, ["lambda"]))
    return out


def summarize(weights: list[TurnWeight]) -> dict[str, float]:
    n = len(weights) or 1
    counts: dict[str, int] = {}
    for w in weights:
        for r in w.reasons:
            counts[r] = counts.get(r, 0) + 1
    out = {f"share_{k}": v / n for k, v in sorted(counts.items())}
    out["mean_weight"] = sum(w.weight for w in weights) / n
    out["share_zero"] = sum(1 for w in weights if w.weight == 0) / n
    return out
