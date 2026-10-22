"""Shared fixtures: a toy environment and a scripted chat client."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from mk33.envs.spec import EnvSpec, ScorerSpec, sha256_file
from mk33.trajectory import Submission, Trajectory, Turn

SCORE_PY = textwrap.dedent('''
    import argparse, importlib.util, json, sys
    ap = argparse.ArgumentParser(); ap.add_argument("--solution"); a = ap.parse_args()
    spec = importlib.util.spec_from_file_location("sol", a.solution + "/solution.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    target = float(open("heldout/target.txt").read())
    print("evaluating")
    print(json.dumps({"score": -abs(m.VALUE - target)}))
''')


@pytest.fixture
def toy_env(tmp_path: Path) -> EnvSpec:
    ws = tmp_path / "workspace"
    sc = tmp_path / "scorer"
    (sc / "heldout").mkdir(parents=True)
    ws.mkdir()
    (ws / "solution.py").write_text("VALUE = 0\n", encoding="utf-8")
    (sc / "heldout" / "target.txt").write_text("42", encoding="utf-8")
    (sc / "score.py").write_text(SCORE_PY, encoding="utf-8")
    return EnvSpec(
        env_id="toy", domain="ml", source="https://github.com/example/toy@abc",
        task_statement="Set VALUE in solution.py to maximize the score.",
        workspace_dir=str(ws),
        scorer=ScorerSpec(
            command=[sys.executable, "score.py", "--solution", "{solution}"],
            script_path="score.py", script_sha256=sha256_file(sc / "score.py"),
            scorer_dir=str(sc), higher_is_better=True, timeout_s=60,
        ),
        baseline_score=-42.0, reference_scores=[0.0, -0.1, 0.1],
    )


class ScriptedClient:
    """Returns a fixed sequence of tool calls; records every context it receives."""

    def __init__(self, steps):
        self.steps = list(steps)
        self.contexts = []

    def chat(self, messages, tools=None, **kw):
        self.contexts.append([dict(m) for m in messages])
        if not self.steps:
            return {"content": "done", "tool_calls": [{"id": "x", "name": "answer", "arguments": {"answer": "stop"}}]}
        reasoning, name, args = self.steps.pop(0)
        return {"content": reasoning, "tool_calls": [{"id": "x", "name": name, "arguments": args}]}


def make_turn(i, tool, args=None, obs="", ws="h0", rnd=1, diff=None, sub=None, timed_out=False,
              reasoning="", elapsed=0.0):
    return Turn(index=i, round=rnd, reasoning=reasoning, tool=tool, args=args or {}, observation=obs,
                workspace_hash=ws, diff=diff, submission=sub, timed_out=timed_out, elapsed_s=elapsed)


def make_sub(rnd, norm, ok=True, files=None, err=None, logs=""):
    return Submission(round=rnd, score=norm, normalized=norm if ok else None, ok=ok,
                      hypothesis=f"h{rnd}", lesson=f"l{rnd}", error_signature=err, logs=logs,
                      solution_files=files or {})


def make_traj(turns, domain="ml", task="task", eps=0.0):
    return Trajectory(env_id="t", domain=domain, task_statement=task, turns=turns, metadata={"eps": eps})
