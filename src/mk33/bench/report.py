"""Summarize bench results into ``report.md``, ``summary.json`` and ``curves.csv``.

Scores are normalized (baseline 0, reference 1). For each pair of context modes the
report gives the mean paired difference over matching (environment, repeat) runs with
a seeded bootstrap 95% interval, so a difference is only called real when the interval
excludes zero.
"""

from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

METRICS = [
    ("best", "best score"),
    ("auc_rounds", "best-so-far AUC"),
    ("productive_rounds", "productive rounds T*"),
    ("mean_gain", "mean gain"),
    ("recovery_rate", "recovery rate"),
    ("wasted_turn_share", "wasted-turn share"),
    ("compactions", "compactions"),
]


def _mean_sd(xs: list[float]) -> str:
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.3f}"
    return f"{statistics.mean(xs):.3f} ± {statistics.stdev(xs):.3f}"


def _bootstrap_ci(diffs: list[float], n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def load_rows(out: str | Path) -> list[dict[str, Any]]:
    path = Path(out) / "results.jsonl"
    if path.exists():
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [json.loads(p.read_text()) for p in sorted(Path(out).glob("runs/*/*/*/result.json"))]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in rows if not r.get("error")]
    modes = sorted({r["mode"] for r in ok})
    envs = sorted({r["env_id"] for r in ok})
    by_mode = defaultdict(list)
    for r in ok:
        by_mode[r["mode"]].append(r)

    summary: dict[str, Any] = {
        "runs": len(rows), "failed_runs": len(rows) - len(ok), "modes": modes, "envs": envs,
        "model": next((r["model"] for r in rows), None), "per_mode": {}, "per_env": {}, "paired": [],
        "curves": {},
    }
    for m in modes:
        rs = by_mode[m]
        summary["per_mode"][m] = {
            "n": len(rs),
            "beat_reference_share": sum(r["beat_reference"] for r in rs) / len(rs),
            "audit_pass_share": sum(r["audit_passed"] for r in rs) / len(rs),
            **{k: [r.get(k) for r in rs] for k, _ in METRICS},
        }
        n_rounds = max(len(r["curve"]) for r in rs)
        summary["curves"][m] = [statistics.mean(r["curve"][i] for r in rs if len(r["curve"]) > i)
                                for i in range(n_rounds)]
    for e in envs:
        summary["per_env"][e] = {m: [r["best"] for r in by_mode[m] if r["env_id"] == e] for m in modes}
    for a, b in combinations(modes, 2):
        ra = {(r["env_id"], r["rep"]): r for r in by_mode[a]}
        rb = {(r["env_id"], r["rep"]): r for r in by_mode[b]}
        keys = sorted(set(ra) & set(rb))
        for metric in ("best", "auc_rounds"):
            diffs = [ra[k][metric] - rb[k][metric] for k in keys]
            if len(diffs) < 2:
                continue
            lo, hi = _bootstrap_ci(diffs)
            summary["paired"].append({
                "a": a, "b": b, "metric": metric, "n_pairs": len(diffs),
                "mean_diff": statistics.mean(diffs), "ci95": [lo, hi],
                "a_wins": sum(d > 0 for d in diffs), "b_wins": sum(d < 0 for d in diffs),
                "significant": lo > 0 or hi < 0,
            })
    return summary


def render_markdown(s: dict[str, Any], cfg: dict[str, Any] | None = None) -> str:
    modes = s["modes"]
    out = ["# Mk33 bench results\n",
           f"Model: `{s['model']}`. Runs: {s['runs']} ({s['failed_runs']} failed and excluded).",
           "Scores are normalized per environment: baseline = 0, reference = 1, above 1 beats the reference.\n"]
    if cfg:
        out.append(f"Budget per run: {cfg.get('max_rounds')} rounds, {cfg.get('max_tool_calls')} tool calls, "
                   f"{cfg.get('max_wall_s', 0) / 60:.0f} min. Context window {cfg.get('window')} tokens, "
                   f"compaction at {cfg.get('theta')}, keep last {cfg.get('keep_rounds')} rounds.\n")
    out.append("## By context mode (mean ± sd over runs)\n")
    out.append("| metric | " + " | ".join(modes) + " |")
    out.append("| --- |" + " --- |" * len(modes))
    for k, label in METRICS:
        out.append(f"| {label} | " + " | ".join(_mean_sd(s["per_mode"][m][k]) for m in modes) + " |")
    out.append("| beat reference | " + " | ".join(f"{s['per_mode'][m]['beat_reference_share']:.0%}"
                                                  for m in modes) + " |")
    out.append("| audit passed | " + " | ".join(f"{s['per_mode'][m]['audit_pass_share']:.0%}"
                                                for m in modes) + " |")
    out.append("\n## Best score by environment\n")
    out.append("| environment | " + " | ".join(modes) + " |")
    out.append("| --- |" + " --- |" * len(modes))
    for e, per in s["per_env"].items():
        out.append(f"| {e} | " + " | ".join(_mean_sd(per[m]) for m in modes) + " |")
    if s["paired"]:
        out.append("\n## Paired comparisons\n")
        out.append("| comparison | metric | pairs | mean diff | 95% CI | wins (a / b) | real difference? |")
        out.append("| --- | --- | --- | --- | --- | --- | --- |")
        for p in s["paired"]:
            lo, hi = p["ci95"]
            out.append(f"| {p['a']} − {p['b']} | {p['metric']} | {p['n_pairs']} | {p['mean_diff']:+.3f} | "
                       f"[{lo:+.3f}, {hi:+.3f}] | {p['a_wins']} / {p['b_wins']} | "
                       f"{'yes' if p['significant'] else 'no, CI includes 0'} |")
    out.append("\n## Best-so-far by round (mean over runs)\n")
    out.append("| round | " + " | ".join(modes) + " |")
    out.append("| --- |" + " --- |" * len(modes))
    n = max(len(c) for c in s["curves"].values()) if s["curves"] else 0
    for i in range(n):
        out.append(f"| {i + 1} | " + " | ".join(
            f"{s['curves'][m][i]:.3f}" if i < len(s["curves"][m]) else "" for m in modes) + " |")
    return "\n".join(out) + "\n"


def write_report(out: str | Path) -> dict[str, Any]:
    out = Path(out)
    rows = load_rows(out)
    if not rows:
        raise FileNotFoundError(f"no results under {out}")
    s = summarize(rows)
    cfg_path = out / "bench_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    (out / "report.md").write_text(render_markdown(s, cfg))
    modes = s["modes"]
    lines = ["round," + ",".join(modes)]
    n = max(len(c) for c in s["curves"].values()) if s["curves"] else 0
    for i in range(n):
        lines.append(f"{i + 1}," + ",".join(f"{s['curves'][m][i]:.4f}" if i < len(s["curves"][m]) else ""
                                            for m in modes))
    (out / "curves.csv").write_text("\n".join(lines) + "\n")
    return s
