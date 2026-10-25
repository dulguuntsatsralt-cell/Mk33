"""The bench refuses package installs so every run uses the same libraries."""

import pytest

from conftest import make_traj  # noqa: F401  (shared fixtures)
from mk33.bench import BenchConfig, prepare
from mk33.bench.run import run_one
from mk33.scaffold import LocalSandbox, ToolConfig, ToolExecutor
from mk33.scaffold.tools import INSTALL_BLOCKED, is_package_install


@pytest.mark.parametrize("cmd", [
    "python -m pip install xgboost",          # what the 7B agent ran in round 9
    "pip install lightgbm", "pip3 install -q catboost==1.2", "uv pip install numpy",
    "conda install -y scipy", "mamba install pandas", "poetry add requests", "pipx install black",
    "apt-get install -y gcc", "npm i left-pad", "cd /tmp && python3 -m pip install --user x",
    "PIP_INDEX=x pip install y",
])
def test_detects_installs(cmd):
    assert is_package_install(cmd)


@pytest.mark.parametrize("cmd", [
    "python solution.py", "ls -la", "pip list", "pip show numpy", "python -c 'import numpy'",
    "grep -r install README.md", "cat pipeline.py", "echo done; ls", "python train.py --install-dir out",
])
def test_ordinary_commands_are_not_blocked(cmd):
    assert not is_package_install(cmd)


def test_executor_blocks_only_when_configured(toy_env):
    on = ToolExecutor(toy_env, LocalSandbox(toy_env.workspace_dir), None, ToolConfig(block_package_installs=True))
    r = on.call("bash", {"command": "python -m pip install xgboost"})
    assert r.observation == INSTALL_BLOCKED and on.blocked_installs == 1
    assert on.call("run_job", {"command": "pip install lightgbm"}).observation == INSTALL_BLOCKED
    assert on.blocked_installs == 2
    assert "[exit 0]" in on.call("bash", {"command": "echo ok"}).observation
    off = ToolExecutor(toy_env, LocalSandbox(toy_env.workspace_dir), None, ToolConfig())
    assert off.call("bash", {"command": "pip --version"}).observation.startswith("[exit")  # default: not blocked
    assert off.blocked_installs == 0


class Installer:
    def __init__(self):
        self.t = 0

    def chat(self, messages, tools=None, **kw):
        self.t += 1
        call = ({"name": "bash", "arguments": {"command": "python -m pip install xgboost"}} if self.t == 1 else
                {"name": "submit", "arguments": {"hypothesis": "baseline", "lesson": "no installs"}})
        return {"content": "", "tool_calls": [{"id": f"c{self.t}", **call}]}


def test_bench_blocks_installs_and_reports_them(tmp_path):
    env = prepare(tmp_path / "envs", tasks=["regression"])[0].env
    row = run_one(env, "rlc", 0, BenchConfig(modes=["rlc"], repeats=1, max_rounds=1, max_tool_calls=3),
                  Installer(), tmp_path / "out", "scripted")
    assert row["blocked_installs"] == 1 and row["rounds"] == 1


def test_bench_config_records_the_library_set(tmp_path):
    import json

    from mk33.bench import SubmitOnlyClient, run_grid

    env = prepare(tmp_path / "envs", tasks=["knapsack"])[0].env
    run_grid([env], BenchConfig(modes=["rlc"], repeats=1, max_rounds=1, workers=1), SubmitOnlyClient,
             tmp_path / "out", "dry-run", log=lambda m: None)
    e = json.loads((tmp_path / "out/bench_config.json").read_text(encoding="utf-8"))["environment"]
    assert e["python"] and "numpy" in e["libraries"] and "xgboost" in e["libraries"]
