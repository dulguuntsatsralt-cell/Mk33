"""Check PALM rules against human labels (paper Section 3.3, "Checking the rules").

Labels per turn: "no_progress", "progress", or "neutral". We report precision and
recall of the zero-weight rules (against "no_progress") and of S+ (against
"progress"), plus Cohen's kappa between two annotators and between rules and the
adjudicated label.
"""

from __future__ import annotations

from collections import Counter

from .weights import TurnWeight

LABELS = ("no_progress", "neutral", "progress")


def rule_label(w: TurnWeight) -> str:
    if w.weight == 0:
        return "no_progress"
    if w.weight == 1 and w.reasons and w.reasons[0] not in ("arex", "all", "lambda"):
        return "progress"
    return "neutral"


def precision_recall(pred: list[bool], gold: list[bool]) -> dict[str, float]:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}


def cohen_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a, "need equal-length, non-empty label lists"
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def evaluate(weights: list[TurnWeight], gold: list[str],
             annotator_b: list[str] | None = None) -> dict[str, object]:
    pred = [rule_label(w) for w in weights]
    out: dict[str, object] = {
        "zero_rules": precision_recall([p == "no_progress" for p in pred], [g == "no_progress" for g in gold]),
        "splus": precision_recall([p == "progress" for p in pred], [g == "progress" for g in gold]),
        "kappa_rules_vs_gold": cohen_kappa(pred, gold),
    }
    if annotator_b is not None:
        out["kappa_annotators"] = cohen_kappa(gold, annotator_b)
        out["rules_below_humans"] = out["kappa_rules_vs_gold"] < out["kappa_annotators"]
    return out
