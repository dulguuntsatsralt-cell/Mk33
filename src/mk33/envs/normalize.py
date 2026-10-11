"""Baseline-to-reference score normalization (paper Section 3.1).

    s_tilde = (s - b) / (s_ref - b)

with the sign flipped for metrics where lower is better. The baseline maps to 0
and the reference to 1; values above 1 beat the reference.
"""

from __future__ import annotations


def normalized_score(
    score: float, baseline: float, reference: float, higher_is_better: bool = True
) -> float:
    if reference == baseline:
        raise ValueError("reference equals baseline: environment has no headroom")
    if not higher_is_better:
        score, baseline, reference = -score, -baseline, -reference
    return (score - baseline) / (reference - baseline)


def normalized_noise(sigma: float, baseline: float, reference: float) -> float:
    """epsilon_e = sigma_e / |s_ref - b|: run-to-run noise on the normalized scale."""
    gap = abs(reference - baseline)
    if gap == 0:
        raise ValueError("reference equals baseline")
    return sigma / gap
