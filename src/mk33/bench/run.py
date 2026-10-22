"""Run the Mk33 agent on the bench environments and record process metrics.

One run = one (environment, context mode, repeat). Each run gets a fresh copy of the
workspace, the same fixed tool set as the paper (minus web tools), the isolated
scorer, and a context window small enough that context management actually matters.

Comparing ``rlc`` with ``truncate`` at a fixed model is an inference-time version of
ablation A2 (``A2_rlc_test_only``): it asks whether the round ledger helps an
untrained model keep improving once old turns no longer fit.

The memory study adds two more families of modes (docs/mk33/MEMORY.md):

    retrieve, rlc+retrieve   dropped turns go into a retrieval memory (text level)
    kv:<policy>              the agent keeps its full text history and the local model's
                             KV cache is compressed to ``kv_budget`` tokens instead
                             (model level; needs ``backend="hf"``)

Setting ``window`` and ``kv_budget`` to the same number compares the two levels at an
equal KV-cache memory budget.

Runs are resumable: a run whose ``result.json`` exists is skipped unless ``force``.
"""

from __future__ import annotations

import copy
import json
import random
import shutil
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..envs.audit import AuditConfig, audit
from ..envs.scorer import IsolatedScorer
from ..envs.spec import EnvSpec
from ..llm import ChatClient
from ..metrics import process_metrics
from ..scaffold import (MODES, SYSTEM_PROMPT, Agent, AgentConfig, Budget, LocalSandbox, RLCConfig, ToolConfig,
                        ToolExecutor)
from ..scaffold.sandbox import shell_note
from ..trajectory import Trajectory

BENCH_TOOLS = ["bash", "read_file", "edit_file", "run_job", "wait_job", "submit"]


@dataclass
class BenchConfig:
    modes: list[str] = field(default_factory=lambda: ["rlc", "truncate"])
    repeats: int = 3
    max_tool_calls: int = 80
    max_rounds: int = 15
    max_wall_s: float = 45 * 60
    window: int = 24_000  # tokens; small on purpose so long runs must be compacted
    theta: float = 0.75
    keep_rounds: int = 2
    ledger_cap: int = 3_000
    bash_timeout_s: int = 120
    max_wait_s: int = 300
    workers: int = 4
    force: bool = False
    no_think: bool = False
    rpm: float = 0.0  # pace model calls to stay under a free tier's requests-per-minute limit
    # Retrieval memory (modes "retrieve", "rlc+retrieve")
    retrieve_k: int = 4
    retrieve_budget: int = 2_000
    retrieve_item_chars: int = 1_200
    retrieve_refresh: str = "round"
    retriever: str = "bm25"
    embed_model: str = ""
    embed_base_url: str = ""
    # KV-cache compression (modes "kv:<policy>", local models only)
    kv_budget: int = 0  # tokens kept per layer and head; 0 = same as window
    kv_sinks: int = 4
    kv_recent: int = 256
    kv_text_window: int = 32_000  # text context limit for kv modes (the model's own limit)

    def __post_init__(self):
        for m in self.modes:
            parse_mode(m)

    def rlc_for(self, mode: str) -> RLCConfig:
        ctx_mode, kv_policy = parse_mode(mode)
        return RLCConfig(
            window=self.kv_text_window if kv_policy else self.window, theta=self.theta,
            keep_rounds=self.keep_rounds, ledger_cap=self.ledger_cap, mode=ctx_mode,
            retrieve_k=self.retrieve_k, retrieve_budget=self.retrieve_budget,
            retrieve_item_chars=self.retrieve_item_chars, retrieve_refresh=self.retrieve_refresh,
            retriever=self.retriever, embed_model=self.embed_model, embed_base_url=self.embed_base_url,
        )


def parse_mode(mode: str) -> tuple[str, str | None]:
    """'rlc' -> ('rlc', None); 'kv:keynorm' -> ('truncate', 'keynorm').

    A kv mode keeps the whole text history (only cut at ``kv_text_window``, the model's
    limit) and lets the KV-cache policy do the forgetting.
    """
    if mode.startswith("kv:"):
        from ..kv.names import POLICIES  # no torch import

        policy = mode[3:]
        if policy not in POLICIES:
            raise ValueError(f"unknown KV policy in mode {mode!r}; choose from {', '.join(POLICIES)}")
        return "truncate", policy
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; choose from {', '.join(MODES)} or kv:<policy>")
    return mode, None


class RetryingClient:
    """Paces and retries API calls; counts calls.

    Free API tiers reject bursts with HTTP 429. Those are retried patiently (honoring a
    Retry-After header when the server sends one); other transient failures (network,
    5xx) get a few quick retries; client errors (400/401/402/403/404) fail at once.
    ``rpm`` > 0 spaces calls across all runs in this process to stay under a rate limit.
    """

    _pace_lock = threading.Lock()
    _next_slot = 0.0

    def __init__(self, inner: ChatClient, tries: int = 3, base_delay: float = 2.0,
                 rpm: float = 0.0, rate_limit_tries: int = 12, max_wait: float = 90.0):
        self.inner = inner
        self.tries = tries
        self.base_delay = base_delay
        self.rpm = rpm
        self.rate_limit_tries = rate_limit_tries
        self.max_wait = max_wait
        self.calls = 0
        self.retries = 0

    def _pace(self):
        if self.rpm <= 0:
            return
        gap = 60.0 / self.rpm
        with RetryingClient._pace_lock:
            now = time.time()
            slot = max(now, RetryingClient._next_slot)
            RetryingClient._next_slot = slot + gap
        if slot > now:
            time.sleep(slot - now)

    @staticmethod
    def _status(e):
        code = getattr(e, "status_code", None)
        if code is None and getattr(e, "response", None) is not None:
            code = getattr(e.response, "status_code", None)
        return code

    @staticmethod
    def _retry_after(e):
        resp = getattr(e, "response", None)
        try:
            v = resp.headers.get("retry-after") if resp is not None else None
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    def chat(self, messages, tools=None, **kw):
        quick = limited = 0
        while True:
            self._pace()
            try:
                self.calls += 1
                return self.inner.chat(messages, tools=tools, **kw)
            except Exception as e:  # network errors, rate limits, 5xx
                code = self._status(e)
                if code in (400, 401, 402, 403, 404):
                    raise
                if code == 429 or "rate limit" in str(e).lower():
                    limited += 1
                    if limited >= self.rate_limit_tries:
                        raise
                    wait = self._retry_after(e) or min(self.max_wait, 5 * 2 ** (limited - 1))
                else:
                    quick += 1
                    if quick >= self.tries:
                        raise
                    wait = self.base_delay * 2 ** (quick - 1)
                self.retries += 1
                time.sleep(wait * (0.75 + 0.5 * random.random()))


def best_so_far_curve(traj: Trajectory, rounds: int) -> list[float]:
    """Best normalized score after each round 1..rounds (baseline 0; carried forward)."""
    best, curve = 0.0, []
    subs = [t.submission for t in traj.turns if t.tool == "submit" and t.submission is not None]
    for r in range(rounds):
        if r < len(subs) and subs[r].ok and subs[r].normalized is not None:
            best = max(best, subs[r].normalized)
        curve.append(best)
    return curve


def run_one(env: EnvSpec, mode: str, rep: int, cfg: BenchConfig, client: ChatClient,
            out: Path, model: str) -> dict[str, Any]:
    run_dir = out / "runs" / env.env_id / mode / f"rep{rep}"
    result_path = run_dir / "result.json"
    if result_path.exists() and not cfg.force:
        row = json.loads(result_path.read_text(encoding="utf-8"))
        if not row.get("error"):
            return row
    if run_dir.exists():
        shutil.rmtree(run_dir)
    ws = run_dir / "workspace"
    shutil.copytree(env.workspace_dir, ws)
    env_run = copy.deepcopy(env)
    env_run.workspace_dir = str(ws.resolve())

    sandbox = LocalSandbox(ws)
    tools = ToolConfig(bash_timeout_s=cfg.bash_timeout_s, max_wait_s=cfg.max_wait_s, enabled=list(BENCH_TOOLS))
    ex = ToolExecutor(env_run, sandbox, IsolatedScorer(env_run), tools)
    agent_cfg = AgentConfig(
        budget=Budget(max_tool_calls=cfg.max_tool_calls, max_wall_s=cfg.max_wall_s, max_rounds=cfg.max_rounds),
        rlc=cfg.rlc_for(mode),
        system_prompt=SYSTEM_PROMPT + (f"\n\n{note}" if (note := shell_note(sandbox)) else "")
        + ("\n\n/no_think" if cfg.no_think else ""),
    )
    row: dict[str, Any] = {"env_id": env.env_id, "task": env.metadata.get("task"), "mode": mode, "rep": rep,
                           "model": model, "error": None}
    retrying = RetryingClient(client, rpm=cfg.rpm)
    t0 = time.time()
    traj = None
    try:
        traj = Agent(retrying, ex, env_run, agent_cfg).run()
    except Exception as e:  # keep the grid going; record what broke
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc()[-4000:]
    finally:
        sandbox.kill_all()
    row["wall_s"] = round(time.time() - t0, 1)
    row["llm_calls"], row["llm_retries"] = retrying.calls, retrying.retries
    if getattr(client, "stats", None):  # local model: KV-cache memory and speed
        from ..kv.client import summarize_stats

        row.update(summarize_stats(client.stats))

    if traj is not None:
        traj.metadata.update({"model": model, "mode": mode, "rep": rep, "bench": asdict(cfg)})
        run_dir.mkdir(parents=True, exist_ok=True)
        traj.dump(run_dir / "trajectory.json")
        m = process_metrics(traj, budget_s=None)
        rep_audit = audit(traj, env_run, AuditConfig())
        curve = best_so_far_curve(traj, cfg.max_rounds)
        subs = [t.submission for t in traj.turns if t.tool == "submit" and t.submission is not None]
        row.update({
            "best": m.best if m.best is not None else 0.0,
            "beat_reference": bool(m.best is not None and m.best >= 1.0),
            "rounds": m.rounds,
            "failed_rounds": sum(1 for s in subs if not s.ok),
            "productive_rounds": m.productive_rounds,
            "mean_gain": m.mean_gain,
            "recovery_rate": m.recovery_rate,
            "wasted_turn_share": m.wasted_turn_share,
            "retry_rate": m.retry_rate,
            "auc_rounds": sum(curve) / len(curve) if curve else 0.0,
            "curve": curve,
            "turns": len(traj.turns),
            "compactions": len(traj.metadata.get("compactions", [])),
            "stop": traj.metadata.get("stop"),
            "audit_passed": rep_audit.passed,
            "audit_flags": [f"{f.check}@{f.turn}: {f.detail}"[:200] for f in rep_audit.flags],
        })
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
    return row


def run_grid(envs: list[EnvSpec], cfg: BenchConfig, client_factory: Callable[[], ChatClient],
             out: str | Path, model: str, log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "bench_config.json").write_text(json.dumps({"model": model, **asdict(cfg)}, indent=2), encoding="utf-8")
    jobs = [(e, m, r) for r in range(cfg.repeats) for e in envs for m in cfg.modes]
    rows: list[dict[str, Any]] = []
    lock = threading.Lock()

    def work(job):
        e, m, r = job
        client = client_factory(m) if getattr(client_factory, "takes_mode", False) else client_factory()
        return run_one(e, m, r, cfg, client, out, model)

    with ThreadPoolExecutor(max_workers=max(1, cfg.workers)) as pool:
        futs = {pool.submit(work, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), start=1):
            e, m, r = futs[f]
            row = f.result()
            with lock:
                rows.append(row)
                status = row["error"] or f"best {row.get('best', 0):.3f}, {row.get('rounds', 0)} rounds"
                log(f"[{i}/{len(jobs)}] {e.env_id} {m} rep{r}: {status}")
    rows.sort(key=lambda x: (x["env_id"], x["mode"], x["rep"]))
    with open(out / "results.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows
