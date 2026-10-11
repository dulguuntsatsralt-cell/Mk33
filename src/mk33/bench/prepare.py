"""Materialize the bench tasks as environment specs.

    out/
      envs.jsonl                       one EnvSpec per task (baseline + reference scores filled in)
      <task>/workspace/                README.md, solution.py (baseline), public data
      <task>/scorer/score.py           scoring script (sha256 recorded in the spec)
      <task>/scorer/heldout/...        held-out split
      <task>/scorer/reference/seedK/   reference solution, run once per seed

The baseline and reference scores are measured by running the real scorer, so the
normalization (baseline -> 0, reference -> 1) and the noise estimate eps come from
actual runs, not from constants.
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from ..envs.filter import FilterConfig, headroom_check
from ..envs.scorer import IsolatedScorer
from ..envs.spec import EnvSpec, ScorerSpec, sha256_file
from .tasks import TASKS, TASKS_BY_ID, BenchTask

REFERENCE_SEEDS = (0, 1, 2)


@dataclass
class PreparedEnv:
    env: EnvSpec
    kept: bool
    reason: str


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def _raw_score(env: EnvSpec, solution_dir: Path) -> float:
    res = IsolatedScorer(env).score(solution_dir)
    if not res.ok or res.score is None:
        raise RuntimeError(f"{env.env_id}: scoring {solution_dir} failed: {res.error_signature}\n{res.logs}")
    return res.score


def prepare_task(task: BenchTask, out: Path, seed: int = 0) -> PreparedEnv:
    root = (out / task.task_id).resolve()
    if root.exists():
        shutil.rmtree(root)
    ws, sc = root / "workspace", root / "scorer"
    # The seed namespace keeps the project's original name on purpose: changing this string
    # would generate different environments and invalidate every measured baseline and run.
    rng = random.Random(f"openreflect-bench/{task.task_id}/{seed}")
    public, heldout = task.generate(rng)
    _write(ws, {"README.md": task.statement, "solution.py": task.baseline_py.lstrip(), **public})
    _write(sc, {"score.py": task.score_py.lstrip(), **heldout})

    env = EnvSpec(
        env_id=f"bench-{task.task_id}",
        domain="ml",
        source=f"mk33-bench/{task.task_id}@seed{seed}",
        task_statement=task.statement,
        workspace_dir=str(ws),
        scorer=ScorerSpec(
            command=[sys.executable, "score.py", "--solution", "{solution}"],
            script_path="score.py",
            script_sha256=sha256_file(sc / "score.py"),
            scorer_dir=str(sc),
            higher_is_better=task.higher_is_better,
            timeout_s=task.timeout_s,
        ),
        metadata={"bench": True, "task": task.task_id, "seed": seed},
    )
    env.baseline_score = _raw_score(env, ws)
    refs = []
    for s in REFERENCE_SEEDS:
        d = sc / "reference" / f"seed{s}"
        _write(d, {"solution.py": task.reference_py.lstrip(), "seed.txt": str(s)})
        refs.append(_raw_score(env, d))
    env.reference_scores = refs
    r = headroom_check(env, FilterConfig())
    return PreparedEnv(env, r.kept, r.reason)


def prepare(out: str | Path, tasks: list[str] | None = None, seed: int = 0) -> list[PreparedEnv]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    chosen = [TASKS_BY_ID[t] for t in tasks] if tasks else TASKS
    prepared = [prepare_task(t, out, seed) for t in chosen]
    with open(out / "envs.jsonl", "w", encoding="utf-8") as fh:
        for p in prepared:
            if p.kept:
                fh.write(json.dumps(p.env.to_dict()) + "\n")
    return prepared
