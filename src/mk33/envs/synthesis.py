"""Teacher-driven environment synthesis (paper Section 3.1, "Sources").

Pipeline for an ML repository:
  1. ``candidate_ok``: license, runnable entry point, existing metric, runtime budget.
  2. The teacher reads a repo digest and returns a JSON draft: task statement,
     metric name and direction, scoring script, baseline and reference commands.
  3. ``materialize`` writes the workspace and scorer directories and records the
     scoring script hash.
  4. Baseline and reference are run by the caller (k reference seeds) and the
     headroom filter decides whether the environment is kept.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..llm import ChatClient
from .spec import EnvSpec, ScorerSpec, sha256_file

PERMISSIVE_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "unlicense", "cc0-1.0"}


@dataclass
class RepoCandidate:
    url: str
    commit: str
    license: str
    has_entry_point: bool
    metric_in_code: bool
    est_gpu_hours: float
    digest: str = ""  # README + file tree + key files, prepared by the crawler
    dataset_urls: list[str] = field(default_factory=list)


@dataclass
class CandidateConfig:
    max_gpu_hours: float = 2.0
    licenses: frozenset[str] = frozenset(PERMISSIVE_LICENSES)


def candidate_ok(c: RepoCandidate, cfg: CandidateConfig | None = None) -> tuple[bool, str]:
    cfg = cfg or CandidateConfig()
    if c.license.lower() not in cfg.licenses:
        return False, f"license {c.license} not permissive"
    if not c.has_entry_point:
        return False, "no runnable entry point"
    if not c.metric_in_code:
        return False, "no metric computed in code"
    if c.est_gpu_hours > cfg.max_gpu_hours:
        return False, f"estimated {c.est_gpu_hours} GPU-h > {cfg.max_gpu_hours}"
    return True, "ok"


TEACHER_PROMPT = """You are building a training environment from a machine learning repository.

Repository: {url} @ {commit}

Repository digest:
{digest}

Write a JSON object with these keys:
- "task_statement": instructions asking an agent to improve the repository's metric. State the
  metric, its direction, the files the agent may change, and that the solution is scored on a
  hidden held-out split. Do not reveal the reference solution.
- "metric": metric name.
- "higher_is_better": true or false.
- "split": how to carve a held-out split from the data (deterministic, seeded).
- "score_py": a complete Python script. It takes --solution DIR and --data DIR, runs the
  solution's predict entry point on the held-out split, and prints {{"score": <float>}} as the
  final stdout line.
- "baseline": shell command for a simple baseline (default config, majority class, or similar).
- "reference": shell command that reproduces the repository's own result.

Return only the JSON object."""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("teacher returned no JSON object")
    return json.loads(m.group(0))


def draft_environment(c: RepoCandidate, teacher: ChatClient) -> dict:
    msg = TEACHER_PROMPT.format(url=c.url, commit=c.commit, digest=c.digest[:60000])
    out = teacher.chat([{"role": "user", "content": msg}], temperature=0.2)
    draft = _extract_json(out["content"])
    missing = {"task_statement", "metric", "higher_is_better", "score_py", "baseline", "reference"} - set(draft)
    if missing:
        raise ValueError(f"teacher draft missing keys: {sorted(missing)}")
    return draft


def materialize(c: RepoCandidate, draft: dict, root: str | Path, env_id: str) -> EnvSpec:
    root = Path(root) / env_id
    workspace = root / "workspace"
    scorer_dir = root / "scorer"
    workspace.mkdir(parents=True, exist_ok=True)
    scorer_dir.mkdir(parents=True, exist_ok=True)
    script = scorer_dir / "score.py"
    script.write_text(draft["score_py"])
    return EnvSpec(
        env_id=env_id,
        domain="ml",
        source=f"{c.url}@{c.commit}",
        task_statement=draft["task_statement"],
        workspace_dir=str(workspace),
        scorer=ScorerSpec(
            command=["python", "score.py", "--solution", "{solution}", "--data", "{scorer_dir}/heldout"],
            script_path="score.py",
            script_sha256=sha256_file(script),
            scorer_dir=str(scorer_dir),
            higher_is_better=bool(draft["higher_is_better"]),
        ),
        dataset_urls=list(c.dataset_urls),
        metadata={
            "metric": draft["metric"],
            "split": draft.get("split", ""),
            "baseline_cmd": draft["baseline"],
            "reference_cmd": draft["reference"],
            "reference_urls": [c.url],
        },
    )
