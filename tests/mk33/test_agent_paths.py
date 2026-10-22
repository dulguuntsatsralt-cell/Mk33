"""Docker-style '/workspace/...' paths and unchanged resubmissions on a local sandbox."""

from conftest import make_traj, make_turn
from mk33.envs.audit import AuditConfig, audit
from mk33.envs.scorer import IsolatedScorer
from mk33.paths import agent_relative
from mk33.scaffold import LocalSandbox, ToolConfig, ToolExecutor


def test_agent_relative():
    assert agent_relative("/workspace/solution.py") == "solution.py"
    assert agent_relative("/workspace/") == "."
    assert agent_relative("/workspace") == "."
    assert agent_relative("solution.py") == "solution.py"
    assert agent_relative("/etc/passwd") == "/etc/passwd"
    assert agent_relative("/workspaces/x") == "/workspaces/x"  # only the exact alias


def executor(env):
    return ToolExecutor(env, LocalSandbox(env.workspace_dir), IsolatedScorer(env), ToolConfig())


def test_file_tools_accept_workspace_alias(toy_env):
    ex = executor(toy_env)
    out = ex.call("read_file", {"path": "/workspace/solution.py"}).observation
    assert "VALUE = 0" in out
    ex.call("edit_file", {"path": "/workspace/solution.py", "old_str": "VALUE = 0", "new_str": "VALUE = 42"})
    assert "VALUE = 42" in ex.call("read_file", {"path": "solution.py"}).observation
    # Leaving the workspace is still refused.
    assert "error" in ex.call("read_file", {"path": "/workspace/../../outside.txt"}).observation


def test_audit_does_not_flag_workspace_alias_but_flags_real_escapes(toy_env):
    turns = [make_turn(0, "read_file", {"path": "/workspace/solution.py"}),
             make_turn(1, "read_file", {"path": "/etc/passwd"})]
    flags = [f for f in audit(make_traj(turns), toy_env, AuditConfig()).flags if f.check == "rules.outside_workspace"]
    assert [f.turn for f in flags] == [1]


def test_unchanged_resubmission_is_scored_and_noted(toy_env):
    ex = executor(toy_env)
    first = ex.call("submit", {"hypothesis": "baseline", "lesson": "start"})
    assert "identical" not in first.observation
    again = ex.call("submit", {"hypothesis": "dynamic programming", "lesson": "described only"})
    assert again.submission is not None and again.submission.ok  # still scored, still a round
    assert "identical to the one submitted in round 1" in again.observation
    assert again.meta["unchanged_since"] == 1
    ex.call("edit_file", {"path": "solution.py", "old_str": "VALUE = 0", "new_str": "VALUE = 40"})
    changed = ex.call("submit", {"hypothesis": "closer value", "lesson": "edited"})
    assert "identical" not in changed.observation
