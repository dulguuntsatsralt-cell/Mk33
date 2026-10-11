"""Noise-aware headroom filter (paper Section 3.1).

An environment is kept only if

    s_ref - b >= max(sigma_k * sigma_e, rel_margin * |s_ref|)

where sigma_e is the standard deviation of k reference runs. For judge problems
scored by pass fraction we instead require b <= judge_max_baseline and
s_ref >= judge_min_reference.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .spec import EnvSpec


@dataclass
class FilterConfig:
    sigma_k: float = 3.0
    rel_margin: float = 0.05
    min_reference_runs: int = 3
    judge_max_baseline: float = 0.6
    judge_min_reference: float = 0.95
    use_sigma: bool = True  # ablation A4
    use_rel_margin: bool = True  # ablation A4
    enabled: bool = True  # ablation A4 ("no filter")


@dataclass
class FilterResult:
    kept: bool
    reason: str
    gap: float | None = None
    required_gap: float | None = None
    sigma: float | None = None


def headroom_check(env: EnvSpec, cfg: FilterConfig | None = None) -> FilterResult:
    cfg = cfg or FilterConfig()
    if env.baseline_score is None or not env.reference_scores:
        return FilterResult(False, "missing baseline or reference scores")
    if len(env.reference_scores) < cfg.min_reference_runs:
        return FilterResult(
            False, f"need >= {cfg.min_reference_runs} reference runs, got {len(env.reference_scores)}"
        )
    if not cfg.enabled:
        return FilterResult(True, "filter disabled")

    b = env.baseline_score
    ref = env.reference_score
    assert ref is not None
    sigma = statistics.stdev(env.reference_scores) if len(env.reference_scores) > 1 else 0.0

    if env.domain == "judge" and env.metadata.get("score_kind") == "pass_fraction":
        ok = b <= cfg.judge_max_baseline and ref >= cfg.judge_min_reference
        reason = "ok" if ok else (
            f"judge thresholds failed: baseline {b:.3f} (max {cfg.judge_max_baseline}), "
            f"reference {ref:.3f} (min {cfg.judge_min_reference})"
        )
        return FilterResult(ok, reason, gap=ref - b, sigma=sigma)

    gap = (ref - b) if env.scorer.higher_is_better else (b - ref)
    terms = []
    if cfg.use_sigma:
        terms.append(cfg.sigma_k * sigma)
    if cfg.use_rel_margin:
        terms.append(cfg.rel_margin * abs(ref))
    required = max(terms) if terms else 0.0
    # Strictly positive gap is always required: otherwise normalization is undefined.
    ok = gap > 0 and gap >= required
    reason = "ok" if ok else f"headroom {gap:.4g} < required {required:.4g}"
    return FilterResult(ok, reason, gap=gap, required_gap=required, sigma=sigma)
