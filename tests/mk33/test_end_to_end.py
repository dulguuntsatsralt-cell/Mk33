"""End-to-end: scripted agent on a toy environment -> PALM -> selection -> windows -> metrics.

The key check is train/test consistency: every context the agent sent to the model
must equal a prefix of the training window that covers that turn.
"""

import json

from mk33.config import load_config
from mk33.envs import IsolatedScorer
from mk33.metrics import process_metrics
from mk33.palm import PALMConfig, compute_weights
from mk33.scaffold import Agent, AgentConfig, Budget, RLCConfig, ToolExecutor
from mk33.selection import accept
from mk33.train import build_windows
from mk33.trajectory import Trajectory
from conftest import ScriptedClient


def edit(old, new, why=""):
    return (why, "edit_file", {"path": "solution.py", "old_str": f"VALUE = {old}", "new_str": f"VALUE = {new}"})


def submit(h, l):
    return ("", "submit", {"hypothesis": h, "lesson": l})


STEPS = [
    ("look at the solution", "read_file", {"path": "solution.py"}),
    edit(0, 30, "VALUE = 0 in solution.py is far from optimal"), submit("raise VALUE to 30", "higher helps"),
    edit(30, 60), submit("overshoot to 60", "60 is too high"),
    ("", "run_job", {"command": "sleep 0.2; echo probe done"}),
    ("", "wait_job", {"job_id": "PLACEHOLDER", "timeout_s": 5}),
    edit(60, 41, "60 overshot; go back near 40"), submit("try 41", "close to target"),
    edit(41, 42), submit("try 42", "42 is optimal"),
    ("", "read_file", {"path": "solution.py"}),
    ("", "read_file", {"path": "solution.py"}),
]


class JobAwareClient(ScriptedClient):
    """Fills in the job id from the previous run_job observation."""

    def chat(self, messages, tools=None, **kw):
        if self.steps and self.steps[0][1] == "wait_job":
            last = messages[-1]["content"]
            jid = last.split("started job ")[-1].split()[0]
            self.steps[0] = ("", "wait_job", {"job_id": jid, "timeout_s": 5})
        return super().chat(messages, tools, **kw)


def run_agent(env, window=800):
    client = JobAwareClient(STEPS)
    ex = ToolExecutor(env, scorer=IsolatedScorer(env))
    cfg = AgentConfig(budget=Budget(max_tool_calls=50), rlc=RLCConfig(window=window, theta=0.75, keep_rounds=1,
                                                                       ledger_cap=400))
    return Agent(client, ex, env, cfg).run(), client, cfg


def test_full_pipeline(toy_env, tmp_path):
    traj, client, cfg = run_agent(toy_env)

    # --- the run itself
    subs = traj.scored_submissions()
    assert [round(s.normalized, 3) for s in subs] == [0.714, 0.571, 0.976, 1.0]
    assert traj.metadata["stop"] == "answer"
    assert "[ledger] R2" in traj.turns[4].observation
    assert traj.metadata["compactions"], "small window should force at least one compaction"
    assert traj.metadata["eps"] > 0

    # --- serialization round trip
    p = tmp_path / "t.json"
    traj.dump(p)
    traj2 = Trajectory.load(p)
    assert traj2.to_dict() == traj.to_dict()

    # --- PALM
    ws = {w.index: w for w in compute_weights(traj, PALMConfig())}
    by_tool = lambda name: [t.index for t in traj.turns if t.tool == name]
    edits, submits = by_tool("edit_file"), by_tool("submit")
    assert ws[edits[0]].reasons == ["surviving_edit"]  # VALUE = 30 is what round 1 scored
    assert ws[submits[0]].reasons == ["improving_submission"]
    assert ws[submits[1]].weight == 0.5  # regression submit
    assert ws[edits[1]].weight == 0.5  # the overshoot edit did not survive into an improving round
    assert ws[edits[2]].reasons[0] in ("surviving_edit", "recovery")
    assert ws[edits[3]].reasons == ["surviving_edit"]
    reads = by_tool("read_file")
    assert ws[reads[-1]].weight == 0  # duplicate read

    # --- selection
    d = accept(traj)
    assert d.accepted, d.reasons

    # --- metrics
    m = process_metrics(traj, budget_s=60)
    assert m.rounds == 4 and m.productive_rounds == 4 and m.recovery_rate == 1.0

    # --- training windows match the contexts the agent actually saw
    windows = build_windows(traj, list(ws.values()), cfg.rlc, drop_empty=False)
    assert len(windows) == len(traj.metadata["compactions"]) + 1
    starts = [w.first_turn for w in windows]
    for i, ctx in enumerate(client.contexts[: len(traj.turns)]):
        w = windows[max(k for k, s in enumerate(starts) if s <= i)]
        assert w.messages[: len(ctx)] == ctx, f"turn {i} context differs from its training window"
    # loss only on assistant messages, with the PALM weight of that turn
    for w in windows:
        for msg, wt in zip(w.messages, w.weights):
            if wt is not None:
                assert msg["role"] == "assistant"
    json.dumps([w.to_dict() for w in windows])  # serializable


def test_cli_offline_pipeline(toy_env, tmp_path, capsys):
    from mk33.cli import main

    traj, _, _ = run_agent(toy_env, window=128000)
    tj = tmp_path / "trajs.jsonl"
    tj.write_text(json.dumps(traj.to_dict()) + "\n", encoding="utf-8")
    envs = tmp_path / "envs.jsonl"
    envs.write_text(json.dumps(toy_env.to_dict()) + "\n", encoding="utf-8")
    cfgp = "configs/mk33/default.yaml"
    assert main(["filter", "--config", cfgp, "--envs", str(envs), "--out", str(tmp_path / "kept.jsonl")]) == 0
    assert main(["audit", "--config", cfgp, "--envs", str(envs), "--trajs", str(tj),
                 "--out", str(tmp_path / "audit.jsonl")]) == 0
    assert main(["select", "--config", cfgp, "--trajs", str(tj), "--audits", str(tmp_path / "audit.jsonl"),
                 "--out", str(tmp_path / "acc.jsonl")]) == 0
    assert main(["weights", "--config", cfgp, "--trajs", str(tmp_path / "acc.jsonl"),
                 "--out", str(tmp_path / "w.jsonl")]) == 0
    assert main(["build-sft", "--config", cfgp, "--trajs", str(tmp_path / "acc.jsonl"),
                 "--out", str(tmp_path / "windows.jsonl")]) == 0
    assert main(["metrics", "--trajs", str(tj), "--budget-s", "60"]) == 0
    out = capsys.readouterr().out
    assert '"accepted": 1' in out and "1 trajectories -> 1 windows" in out
    assert load_config(cfgp).rlc.window == 128000
