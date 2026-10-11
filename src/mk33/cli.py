"""``mk33`` command line.

    mk33 filter      headroom filter + decontamination over environment specs
    mk33 rollout     run the agent (teacher or trained model) on one environment
    mk33 audit       process audit of trajectories
    mk33 select      acceptance rule + per-environment cap
    mk33 weights     PALM weights and summary
    mk33 build-sft   offline RLC windowing -> training windows jsonl
    mk33 train       weighted SFT
    mk33 metrics     process metrics
    mk33 validate-palm  PALM rules vs human labels
    mk33 bench-prepare  build the small CPU-only bench environments
    mk33 bench-run      run the agent on them (any OpenAI-compatible model)
    mk33 bench-report   summarize bench results
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import load_config

DEFAULT_CONFIG = "configs/mk33/default.yaml"


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def _write_jsonl(rows, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def cmd_filter(a) -> int:
    from .envs import BenchmarkIndex, EnvSpec, decontaminate, headroom_check

    cfg = load_config(a.config)
    envs = [EnvSpec.from_dict(d) for d in _read_jsonl(a.envs)]
    kept, report = [], {"input": len(envs), "headroom_dropped": 0, "decontam_dropped": 0, "per_rule": {}}
    for e in envs:
        r = headroom_check(e, cfg.filter)
        if r.kept:
            kept.append(e)
        else:
            report["headroom_dropped"] += 1
    if cfg.decontam["enabled"] and a.benchmarks:
        idx = BenchmarkIndex.from_jsonl(a.benchmarks, n=cfg.decontam["n"],
                                        ngram_threshold=cfg.decontam["threshold"])
        kept, dropped, per_rule = decontaminate(kept, idx)
        report["decontam_dropped"] = len(dropped)
        report["per_rule"] = per_rule
    report["kept"] = len(kept)
    _write_jsonl([e.to_dict() for e in kept], a.out)
    print(json.dumps(report, indent=2))
    return 0


def cmd_rollout(a) -> int:
    from .envs import EnvSpec, IsolatedScorer
    from .llm import OpenAIChatClient
    from .scaffold import Agent, AgentConfig, Budget, DockerSandbox, LocalSandbox, ToolExecutor

    cfg = load_config(a.config)
    env = EnvSpec.load(a.env)
    sandbox = DockerSandbox(env.workspace_dir, a.sandbox_image) if a.sandbox_image else LocalSandbox(env.workspace_dir)
    ex = ToolExecutor(env, sandbox, IsolatedScorer(env), cfg.tools)
    client = OpenAIChatClient(a.model, a.base_url, temperature=a.temperature)
    budget = cfg.budget.get(env.domain, Budget())
    try:
        traj = Agent(client, ex, env, AgentConfig(budget=budget, rlc=cfg.rlc)).run()
    finally:
        sandbox.kill_all()
        if hasattr(sandbox, "close"):
            sandbox.close()
    traj.metadata.update({"teacher": a.model, "config": cfg.name, "temperature": a.temperature})
    if a.out.endswith(".jsonl"):  # append, so many rollouts can share one file
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
    else:
        traj.dump(a.out)
    print(f"wrote {a.out}: {len(traj.turns)} turns, best {traj.best_normalized()}")
    return 0


def cmd_audit(a) -> int:
    from .envs import EnvSpec, audit
    from .trajectory import iter_jsonl

    cfg = load_config(a.config)
    envs = {e["env_id"]: EnvSpec.from_dict(e) for e in _read_jsonl(a.envs)}
    rows = []
    for i, t in enumerate(iter_jsonl(a.trajs)):
        rep = audit(t, envs[t.env_id], cfg.audit)
        rows.append({"index": i, "env_id": t.env_id, "passed": rep.passed,
                     "flags": [asdict(f) for f in rep.flags]})
    _write_jsonl(rows, a.out)
    print(f"{sum(r['passed'] for r in rows)}/{len(rows)} passed")
    return 0


def cmd_select(a) -> int:
    from .envs.audit import AuditFlag, AuditReport
    from .selection import select
    from .trajectory import iter_jsonl, write_jsonl

    cfg = load_config(a.config)
    trajs = list(iter_jsonl(a.trajs))
    audits = {}
    if a.audits:
        for r in _read_jsonl(a.audits):
            audits[r["index"]] = AuditReport([AuditFlag(**f) for f in r["flags"]])
    accepted, decisions = select(trajs, cfg.selection, audits)
    write_jsonl(accepted, a.out)
    reasons: dict[str, int] = {}
    for d in decisions:
        for r in d.reasons:
            key = r.split(" ")[0] if not r.startswith("audit") else "audit"
            reasons[key] = reasons.get(key, 0) + 1
    print(json.dumps({"input": len(trajs), "accepted": len(accepted), "rejections": reasons}, indent=2))
    return 0


def _palm_cfg(cfg, labels_path):
    import copy

    pc = copy.deepcopy(cfg.palm)
    labels = {}
    if labels_path:
        for r in _read_jsonl(labels_path):
            labels[(r["env_id"], r["turn"])] = float(r["weight"])
    return pc, labels


def cmd_weights(a) -> int:
    from .palm import compute_weights, summarize
    from .trajectory import iter_jsonl

    cfg = load_config(a.config)
    pc, labels = _palm_cfg(cfg, a.labels)
    rows, all_w = [], []
    for t in iter_jsonl(a.trajs):
        if labels:
            pc.labels = {k[1]: v for k, v in labels.items() if k[0] == t.env_id}
        ws = compute_weights(t, pc)
        all_w += ws
        rows.append({"env_id": t.env_id, "weights": [asdict(w) for w in ws]})
    _write_jsonl(rows, a.out)
    print(json.dumps(summarize(all_w), indent=2))
    return 0


def cmd_build_sft(a) -> int:
    from .palm import compute_weights
    from .tokens import hf_counter
    from .train import build_windows
    from .trajectory import iter_jsonl

    cfg = load_config(a.config)
    pc, labels = _palm_cfg(cfg, a.labels)
    count = hf_counter(a.tokenizer) if a.tokenizer else None
    rows, n_traj = [], 0
    for t in iter_jsonl(a.trajs):
        if labels:
            pc.labels = {k[1]: v for k, v in labels.items() if k[0] == t.env_id}
        ws = compute_weights(t, pc)
        for w in build_windows(t, ws, cfg.rlc_train, count):
            rows.append(w.to_dict())
        n_traj += 1
    _write_jsonl(rows, a.out)
    print(f"{n_traj} trajectories -> {len(rows)} windows -> {a.out}")
    return 0


def cmd_train(a) -> int:
    from .train.sft import train

    cfg = load_config(a.config)
    if a.train_files:
        cfg.sft.train_files = a.train_files.split(",")
    if a.output_dir:
        cfg.sft.output_dir = a.output_dir
    train(cfg.sft)
    return 0


def cmd_metrics(a) -> int:
    from .metrics import aggregate, process_metrics
    from .trajectory import iter_jsonl

    ms = [process_metrics(t, a.budget_s) for t in iter_jsonl(a.trajs)]
    print(json.dumps({"n": len(ms), **aggregate(ms)}, indent=2))
    return 0


def cmd_validate_palm(a) -> int:
    from .palm import compute_weights
    from .palm.validate import evaluate
    from .trajectory import iter_jsonl

    cfg = load_config(a.config)
    trajs = {t.env_id: t for t in iter_jsonl(a.trajs)}
    labels = _read_jsonl(a.labels)  # {env_id, turn, label_a, label_b?}
    wcache = {k: {w.index: w for w in compute_weights(t, cfg.palm)} for k, t in trajs.items()}
    ws = [wcache[r["env_id"]][r["turn"]] for r in labels]
    gold = [r["label_a"] for r in labels]
    b = [r["label_b"] for r in labels] if all("label_b" in r for r in labels) else None
    print(json.dumps(evaluate(ws, gold, b), indent=2))
    return 0


def cmd_bench_prepare(a) -> int:
    from .bench import prepare

    tasks = a.tasks.split(",") if a.tasks else None
    for p in prepare(a.out, tasks, a.seed):
        e = p.env
        refs = ", ".join(f"{x:.4g}" for x in e.reference_scores)
        print(f"{e.env_id}: baseline {e.baseline_score:.4g}, reference [{refs}], "
              f"{'kept' if p.kept else 'DROPPED: ' + p.reason}")
    print(f"wrote {Path(a.out) / 'envs.jsonl'}")
    return 0


def cmd_bench_run(a) -> int:
    from .bench import BenchConfig, SubmitOnlyClient, run_grid, write_report
    from .envs import EnvSpec

    envs = [EnvSpec.from_dict(d) for d in _read_jsonl(a.envs)]
    if a.tasks:
        keep = set(a.tasks.split(","))
        envs = [e for e in envs if e.metadata.get("task") in keep]
    cfg = BenchConfig(modes=a.modes.split(","), repeats=a.repeats, max_tool_calls=a.max_tool_calls,
                      max_rounds=a.max_rounds, max_wall_s=a.max_minutes * 60, window=a.window,
                      workers=a.workers, force=a.force, no_think=a.no_think, rpm=a.rpm)
    if a.dry_run:
        model, factory = "dry-run", SubmitOnlyClient
    else:
        if not a.model:
            raise SystemExit("--model is required unless --dry-run")
        from .llm import OpenAIChatClient

        model = a.model
        factory = lambda: OpenAIChatClient(a.model, a.base_url, api_key_env=a.api_key_env,
                                           temperature=a.temperature, max_tokens=a.max_tokens,
                                           timeout=a.timeout)
    run_grid(envs, cfg, factory, a.out, model)
    write_report(a.out)
    print(f"report: {Path(a.out) / 'report.md'}")
    return 0


def cmd_bench_report(a) -> int:
    from .bench import write_report

    write_report(a.out)
    print((Path(a.out) / "report.md").read_text())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mk33", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **args):
        sp = sub.add_parser(name)
        sp.add_argument("--config", default=DEFAULT_CONFIG)
        for k, kw in args.items():
            sp.add_argument(f"--{k.replace('_', '-')}", **kw)
        sp.set_defaults(fn=fn)

    add("filter", cmd_filter, envs={"required": True}, benchmarks={}, out={"required": True})
    add("rollout", cmd_rollout, env={"required": True}, model={"required": True}, base_url={},
        temperature={"type": float, "default": 0.7}, sandbox_image={}, out={"required": True})
    add("audit", cmd_audit, envs={"required": True}, trajs={"required": True}, out={"required": True})
    add("select", cmd_select, trajs={"required": True}, audits={}, out={"required": True})
    add("weights", cmd_weights, trajs={"required": True}, labels={}, out={"required": True})
    add("build-sft", cmd_build_sft, trajs={"required": True}, labels={}, tokenizer={}, out={"required": True})
    add("train", cmd_train, train_files={"help": "comma-separated; overrides sft.train_files"},
        output_dir={})
    add("metrics", cmd_metrics, trajs={"required": True}, budget_s={"type": float})
    add("validate-palm", cmd_validate_palm, trajs={"required": True}, labels={"required": True})
    add("bench-prepare", cmd_bench_prepare, out={"default": "bench/envs"},
        tasks={"help": "comma-separated subset of regression,sentiment,forecast,knapsack"},
        seed={"type": int, "default": 0})
    add("bench-run", cmd_bench_run, envs={"default": "bench/envs/envs.jsonl"}, out={"default": "bench/results"},
        model={}, base_url={}, api_key_env={"default": "MODEL_API_KEY"},
        temperature={"type": float, "default": 0.7}, max_tokens={"type": int, "default": 4096},
        modes={"default": "rlc,truncate"}, repeats={"type": int, "default": 3}, tasks={},
        max_rounds={"type": int, "default": 15}, max_tool_calls={"type": int, "default": 80},
        max_minutes={"type": float, "default": 45}, window={"type": int, "default": 24000},
        workers={"type": int, "default": 4}, force={"action": "store_true"},
        dry_run={"action": "store_true", "help": "no API calls; resubmits the baseline"},
        no_think={"action": "store_true", "help": "turn off Qwen3 thinking (/no_think); much faster on CPU"},
        timeout={"type": float, "default": 300.0, "help": "seconds before one model call is abandoned"},
        rpm={"type": float, "default": 0.0, "help": "max model calls per minute (for free API tiers); 0 = no pacing"})
    add("bench-report", cmd_bench_report, out={"default": "bench/results"})

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
