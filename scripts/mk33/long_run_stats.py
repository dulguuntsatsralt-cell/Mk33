"""Numbers for the paper's long-run section, recomputed from the saved runs.

    python scripts/mk33/long_run_stats.py            # prints the table, writes summary.json

Reads docs/mk33/results/long-runs/run*/{truncate,rlc}/ (result.json and the gzipped
trajectory). For each run it reports the best normalized score, rounds, re-try rate,
the round of the last improvement, the first compaction, seconds per model call, and a
context replay: the contexts each mode would have sent, rebuilt with the same
ContextBuilder the agent uses, to measure prompt length and how much of each prompt
starts with exactly the same text as the previous prompt (the part a prefix cache such
as Ollama's can reuse). Token counts in the replay use the default chars/4 estimate.
"""

from __future__ import annotations

import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mk33.bench.run import BenchConfig  # noqa: E402
from mk33.scaffold.context import ContextBuilder  # noqa: E402
from mk33.trajectory import Trajectory  # noqa: E402

RUNS = ROOT / "docs/mk33/results/long-runs"
MODES = ("truncate", "rlc")


def config(run_dir: Path) -> BenchConfig:
    raw = json.loads((run_dir / "bench_config.json").read_text(encoding="utf-8"))
    fields = BenchConfig.__dataclass_fields__
    return BenchConfig(**{k: v for k, v in raw.items() if k in fields})


def serialize(messages: list[dict]) -> str:
    return "\n".join(json.dumps(m, ensure_ascii=False, sort_keys=True) for m in messages)


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def replay(traj: Trajectory, cfg: BenchConfig, mode: str) -> dict:
    """Rebuild the context before every model call; measure length and reuse."""
    ctx = ContextBuilder(traj.task_statement, traj.pinned.get("skills", ""),
                         traj.pinned.get("system", ContextBuilder.__init__.__defaults__[1]),
                         cfg.rlc_for(mode))
    lengths, reused, prev = [], [], None
    for turn in traj.turns:
        messages, _ = ctx.next_context()
        text = serialize(messages)
        lengths.append(len(text) / 4)
        if prev is not None:
            reused.append(common_prefix(prev, text) / len(text))
        prev = text
        ctx.record(turn)
    return {"mean_prompt_tokens": round(statistics.mean(lengths)),
            "prefix_reused": round(statistics.mean(reused), 3) if reused else None}


def first_compaction_round(traj: Trajectory) -> int | None:
    """Round in which the context was first compacted (events record the turn index)."""
    events = [e for e in traj.metadata.get("compactions") or [] if isinstance(e, dict) and "turn" in e]
    if not events:
        return None
    turn = min(e["turn"] for e in events)
    rounds = [t.round for t in traj.turns if t.index == turn]
    return rounds[0] if rounds else None


def last_gain_round(curve: list[float]) -> int | None:
    last, best = None, 0.0
    for i, v in enumerate(curve, start=1):
        if v > best + 1e-9:
            best, last = v, i
    return last


def seconds_per_call(traj: Trajectory) -> float:
    times = [t.elapsed_s for t in traj.turns if t.elapsed_s is not None]
    steps = [b - a for a, b in zip([0.0] + times, times)]
    return statistics.median(steps) if steps else float("nan")


FAILURE_KINDS = [  # first match wins; checked against the failed submission's error signature
    ("missing library", ("ModuleNotFoundError", "No module named")),
    ("escaped newlines in code", ("unexpected character after line continuation",)),
    ("syntax or indentation", ("SyntaxError", "IndentationError")),
    ("required function missing", ("has no attribute 'fit_predict'", "has no attribute 'solve'")),
    ("undefined name", ("NameError",)),
    ("wrong library API", ("Invalid parameter", "unexpected keyword argument", "got an unexpected")),
]


def failure_kind(signature: str) -> str:
    for kind, needles in FAILURE_KINDS:
        if any(n in signature for n in needles):
            return kind
    return "other"


def failures(traj: Trajectory) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in traj.turns:
        s = t.submission
        if s is not None and not s.ok:
            k = failure_kind(s.error_signature or s.logs or "")
            counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def one(run_dir: Path, mode: str) -> dict:
    d = run_dir / mode
    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
    with gzip.open(d / "trajectory.json.gz", "rt", encoding="utf-8") as fh:
        traj = Trajectory.from_dict(json.load(fh))
    row = {
        "run": int(run_dir.name.removeprefix("run")), "mode": mode, "best": round(res["best"] or 0.0, 3),
        "rounds": res["rounds"], "failed_rounds": res["failed_rounds"],
        "retry_rate": round(res["retry_rate"], 2) if res.get("retry_rate") is not None else None,
        "last_gain_round": last_gain_round(res.get("curve") or []),
        "first_compaction_round": first_compaction_round(traj),
        "compactions": res.get("compactions"), "llm_calls": res["llm_calls"],
        "s_per_call": round(seconds_per_call(traj)), "minutes": round(res["wall_s"] / 60),
        "stop": res["stop"], "text_tool_calls": res.get("text_tool_calls"),
        "failures": failures(traj),
    }
    row.update(replay(traj, config(run_dir), mode))
    return row


def main() -> int:
    rows = [one(r, m) for r in sorted(RUNS.glob("run*")) for m in MODES]
    head = "run mode      best  rounds retry lastgain firstcomp s/call min  prompt_tok reused stop"
    print(head)
    for r in rows:
        print(f"{r['run']:>3} {r['mode']:<9} {r['best']:5.3f} {r['rounds']:>6} {r['retry_rate']!s:>5} "
              f"{r['last_gain_round']!s:>8} {r['first_compaction_round']!s:>9} {r['s_per_call']:>6} "
              f"{r['minutes']:>4} {r['mean_prompt_tokens']:>10} {r['prefix_reused']:>6.0%} {r['stop']}")
    by = {(r["run"], r["mode"]): r for r in rows}
    runs = sorted({r["run"] for r in rows})
    speed = [by[(n, "truncate")]["s_per_call"] / by[(n, "rlc")]["s_per_call"] for n in runs]
    print("failed rounds by kind:")
    for r in rows:
        print(f"  run {r['run']} {r['mode']:<9} {r['failures']}")
    print("speedup truncate/rlc s per call:", ", ".join(f"{s:.1f}x" for s in speed))
    out = {"rows": rows, "speedup": [round(s, 2) for s in speed]}
    (RUNS / "summary.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {RUNS / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
