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
    mk33 bench-collect  gather trajectories of finished bench runs (teacher data)
    mk33 bench-compare  compare several models' bench results, paired by task and repeat
    mk33 ask         answer a question with web search (any OpenAI-compatible model)
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
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


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
    search = fetch = None
    if a.web:
        from .web import make_fetch, make_search

        search, fetch = make_search(a.web), make_fetch()
    ex = ToolExecutor(env, sandbox, IsolatedScorer(env), cfg.tools, search=search, fetch=fetch)
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

    modes = a.modes.split(",")
    kv_modes = [m for m in modes if m.startswith("kv:")]
    if kv_modes and a.backend != "hf":
        raise SystemExit(f"{', '.join(kv_modes)} compress a local model's KV cache; add --backend hf --model <hf id>")
    envs = [EnvSpec.from_dict(d) for d in _read_jsonl(a.envs)]
    if a.tasks:
        keep = set(a.tasks.split(","))
        envs = [e for e in envs if e.metadata.get("task") in keep]
    workers = 1 if a.backend == "hf" else a.workers  # one GPU, one shared model
    cfg = BenchConfig(modes=modes, repeats=a.repeats, max_tool_calls=a.max_tool_calls,
                      max_rounds=a.max_rounds, max_wall_s=a.max_minutes * 60, window=a.window,
                      workers=workers, force=a.force, no_think=a.no_think, rpm=a.rpm,
                      retrieve_k=a.retrieve_k, retrieve_budget=a.retrieve_budget,
                      retrieve_refresh=a.retrieve_refresh, retriever=a.retriever,
                      embed_model=a.embed_model or "", embed_base_url=a.embed_base_url or "",
                      kv_budget=a.kv_budget or a.window, kv_sinks=a.kv_sinks, kv_recent=a.kv_recent,
                      kv_text_window=a.kv_text_window)
    if a.dry_run:
        model, factory = "dry-run", SubmitOnlyClient
    elif a.backend == "hf":
        if not a.model:
            raise SystemExit("--model <hugging face id or path> is required with --backend hf")
        from .kv.client import HFClientFactory

        model = a.model + (f" ({a.quantize})" if a.quantize else "")  # reports state the quantization
        factory = HFClientFactory(a.model, budget=a.kv_budget or a.window, sinks=a.kv_sinks, recent=a.kv_recent,
                                  chunk=a.kv_chunk, temperature=a.temperature, max_tokens=a.max_tokens,
                                  device=a.device, dtype=a.dtype, quantize=a.quantize)
    elif a.backend == "ollama":
        if not a.model:
            raise SystemExit("--model <ollama model name> is required with --backend ollama")
        from .llm import OllamaChatClient

        model = a.model
        factory = lambda: OllamaChatClient(a.model, a.base_url, temperature=a.temperature,
                                           max_tokens=a.max_tokens, timeout=a.timeout)
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
    print((Path(a.out) / "report.md").read_text(encoding="utf-8"))
    return 0


def cmd_bench_collect(a) -> int:
    from .bench import collect_trajectories

    counts = collect_trajectories(a.out, a.to, require_audit=not a.include_flagged)
    print(json.dumps(counts))
    return 0


def cmd_bench_compare(a) -> int:
    from .bench import compare_dirs, render_comparison

    dirs = dict(item.split("=", 1) for item in a.runs.split(","))
    c = compare_dirs(dirs, mode=a.mode)
    text = render_comparison(c)
    if a.save:
        Path(a.save).write_text(text, encoding="utf-8")
        Path(a.save).with_suffix(".json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    print(text)
    return 0


def cmd_ask(a) -> int:
    from .llm import OpenAIChatClient
    from .research import ask
    from .web import make_fetch, make_search

    search = make_search(a.web)
    client = OpenAIChatClient(a.model, a.base_url, api_key_env=a.api_key_env, temperature=a.temperature,
                              max_tokens=a.max_tokens, timeout=a.timeout)
    print(f"searching with {search.provider}...", file=sys.stderr)
    log = (lambda m: print(m, file=sys.stderr)) if not a.quiet else None
    r = ask(a.question, client, search, make_fetch(), max_tool_calls=a.max_tool_calls,
            max_minutes=a.max_minutes, window=a.window, log=log)
    if a.out and r.trajectory is not None:
        r.trajectory.dump(a.out)
    if r.answer is None:
        print(f"no answer (stopped: {r.stop}, {r.turns} steps)")
        return 1
    print(r.answer)
    if r.confidence is not None:
        print(f"confidence: {r.confidence:.2f}")
    for e in r.evidence:
        print(f"source: {e}")
    print(f"({r.searches} searches, {r.fetches} pages read, {r.turns} steps)", file=sys.stderr)
    return 0


def _safe_console() -> None:
    """Windows consoles and pipes may use a legacy code page; print UTF-8 instead of crashing
    on characters like the minus sign in reports or non-English text in agent output."""
    for stream in (sys.stdout, sys.stderr):
        enc = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        if enc != "utf8" and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def main(argv: list[str] | None = None) -> int:
    _safe_console()
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
        temperature={"type": float, "default": 0.7}, sandbox_image={}, out={"required": True},
        web={"choices": ["auto", "duckduckgo", "tavily", "brave"],
             "help": "enable web_search/web_fetch with this search provider"})
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
        modes={"default": "rlc,truncate",
               "help": "comma-separated: rlc, truncate, retrieve, rlc+retrieve, none, kv:<policy> "
                       "(policies: full, random, streaming, keynorm, h2o)"}, repeats={"type": int, "default": 3}, tasks={},
        max_rounds={"type": int, "default": 15}, max_tool_calls={"type": int, "default": 80},
        max_minutes={"type": float, "default": 45}, window={"type": int, "default": 24000},
        workers={"type": int, "default": 4}, force={"action": "store_true"},
        dry_run={"action": "store_true", "help": "no API calls; resubmits the baseline"},
        no_think={"action": "store_true", "help": "turn off Qwen3 thinking (/no_think); much faster on CPU"},
        timeout={"type": float, "default": 300.0, "help": "seconds before one model call is abandoned"},
        rpm={"type": float, "default": 0.0, "help": "max model calls per minute (for free API tiers); 0 = no pacing"},
        backend={"default": "openai", "choices": ["openai", "ollama", "hf"],
                 "help": "openai: any OpenAI-compatible endpoint; ollama: Ollama's native API, which also records "
                         "per-call prompt and reply time (cache reuse); hf: a local Hugging Face model (needed for kv: modes)"},
        device={"help": "hf backend: cuda or cpu (default: cuda if available)"},
        dtype={"default": "auto", "help": "hf backend: auto, bfloat16, float16 or float32"},
        quantize={"default": "", "choices": ["", "4bit", "8bit"],
                  "help": "hf backend: load the weights in 4 or 8 bits (bitsandbytes, CUDA only)"},
        retrieve_k={"type": int, "default": 4, "help": "retrieve modes: items placed back per refresh"},
        retrieve_budget={"type": int, "default": 2000, "help": "retrieve modes: tokens for the retrieved block"},
        retrieve_refresh={"default": "round", "choices": ["round", "compaction"]},
        retriever={"default": "bm25", "choices": ["bm25", "embed"]},
        embed_model={"help": "retriever embed: model name at the embeddings endpoint"},
        embed_base_url={"help": "retriever embed: OpenAI-compatible base URL (default: OpenAI)"},
        kv_budget={"type": int, "default": 0, "help": "kv modes: cache tokens kept per layer/head (0 = --window)"},
        kv_sinks={"type": int, "default": 4}, kv_recent={"type": int, "default": 256},
        kv_chunk={"type": int, "default": 512, "help": "kv modes: prefill chunk size"},
        kv_text_window={"type": int, "default": 32000,
                        "help": "kv modes: text history limit (the model's context length)"})
    add("bench-report", cmd_bench_report, out={"default": "bench/results"})
    add("bench-collect", cmd_bench_collect, out={"default": "bench/results"}, to={"required": True},
        include_flagged={"action": "store_true", "help": "keep runs the process audit flagged"})
    add("bench-compare", cmd_bench_compare,
        runs={"required": True, "help": "name=dir pairs, e.g. base=r/base,palm=r/palm"},
        mode={"help": "context mode to compare (default: the one mode all dirs share, else rlc)"},
        save={"help": "write the markdown table here (and .json next to it)"})
    add("ask", cmd_ask, model={"required": True}, base_url={}, api_key_env={"default": "MODEL_API_KEY"},
        web={"default": "auto", "choices": ["auto", "duckduckgo", "tavily", "brave"],
             "help": "search provider (auto: Tavily or Brave if their key is set, else DuckDuckGo)"},
        temperature={"type": float, "default": 0.3}, max_tokens={"type": int, "default": 2048},
        timeout={"type": float, "default": 300.0}, max_tool_calls={"type": int, "default": 30},
        max_minutes={"type": float, "default": 20}, window={"type": int, "default": 24000},
        out={"help": "save the full trajectory (JSON) here"}, quiet={"action": "store_true"})
    sub.choices["ask"].add_argument("question")

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
