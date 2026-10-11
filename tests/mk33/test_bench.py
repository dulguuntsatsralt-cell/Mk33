"""Bench: environments are well-formed, the runner measures real improvement, and
context management is exercised when runs outgrow the window."""

import json

import pytest

from mk33.bench import TASKS, BenchConfig, SubmitOnlyClient, prepare, run_grid, summarize, write_report
from mk33.bench.report import render_markdown
from mk33.envs import IsolatedScorer, headroom_check

STRONG_KNAPSACK = '''
def solve(w, v, cap):
    best = [[0] * (cap + 1) for _ in range(len(w) + 1)]
    for i in range(len(w)):
        for c in range(cap + 1):
            best[i + 1][c] = best[i][c]
            if w[i] <= c and best[i][c - w[i]] + v[i] > best[i + 1][c]:
                best[i + 1][c] = best[i][c - w[i]] + v[i]
    c, chosen = cap, []
    for i in range(len(w), 0, -1):
        if best[i][c] != best[i - 1][c]:
            chosen.append(i - 1)
            c -= w[i - 1]
    return chosen
'''

GREEDY_KNAPSACK = '''
def solve(w, v, cap):
    chosen, used = [], 0
    for i in sorted(range(len(w)), key=lambda i: -v[i] / w[i]):
        if used + w[i] <= cap:
            chosen.append(i)
            used += w[i]
    return chosen
'''

BASELINE_KNAPSACK = TASKS[3].baseline_py.lstrip()


@pytest.fixture(scope="module")
def envs(tmp_path_factory):
    out = tmp_path_factory.mktemp("bench")
    prepared = prepare(out)
    return out, prepared


def test_every_task_passes_the_headroom_filter(envs):
    _, prepared = envs
    assert {p.env.metadata["task"] for p in prepared} == {t.task_id for t in TASKS}
    for p in prepared:
        assert p.kept, (p.env.env_id, p.reason)
        assert len(p.env.reference_scores) == 3
        assert headroom_check(p.env).kept


def test_prepare_is_deterministic(envs, tmp_path):
    _, first = envs
    second = prepare(tmp_path, ["knapsack"])
    a = next(p.env for p in first if p.env.metadata["task"] == "knapsack")
    assert second[0].env.baseline_score == a.baseline_score
    assert second[0].env.reference_scores == a.reference_scores


def test_heldout_data_stays_out_of_the_workspace(envs):
    out, prepared = envs
    for p in prepared:
        ws = {f.name for f in (out / p.env.metadata["task"] / "workspace").iterdir()}
        assert "heldout" not in ws and "score.py" not in ws and "reference" not in ws
        assert {"README.md", "solution.py"} <= ws


def test_normalization_matches_paper(envs, tmp_path):
    """Baseline scores 0; the exact DP scores the knapsack ceiling, above the reference."""
    _, prepared = envs
    env = next(p.env for p in prepared if p.env.metadata["task"] == "knapsack")
    (tmp_path / "solution.py").write_text(STRONG_KNAPSACK)
    r = IsolatedScorer(env).score(tmp_path)
    assert r.ok and r.score == pytest.approx(0.0, abs=1e-12)
    assert r.normalized > 1.0
    base = IsolatedScorer(env).score(env.workspace_dir)
    assert base.normalized == pytest.approx(0.0, abs=1e-9)


class ImprovingClient:
    """Scripted model: greedy in round 1, a regression in round 2, the exact DP in round 3.

    Long reasoning text makes the context outgrow a small window, forcing compaction.
    """

    def __init__(self, pad=0):
        self.pad = "think " * pad
        self.plan = [
            ("read_file", {"path": "solution.py"}),
            ("edit_file", {"path": "solution.py", "old_str": BASELINE_KNAPSACK, "new_str": GREEDY_KNAPSACK}),
            ("submit", {"hypothesis": "ratio greedy", "lesson": "greedy beats in-order"}),
            ("edit_file", {"path": "solution.py", "old_str": GREEDY_KNAPSACK, "new_str": BASELINE_KNAPSACK}),
            ("submit", {"hypothesis": "back to in-order", "lesson": "worse; revert"}),
            ("edit_file", {"path": "solution.py", "old_str": BASELINE_KNAPSACK, "new_str": STRONG_KNAPSACK}),
            ("submit", {"hypothesis": "exact DP", "lesson": "DP is optimal and fast enough"}),
        ]

    def chat(self, messages, tools=None, **kw):
        name, args = self.plan.pop(0) if self.plan else (
            "submit", {"hypothesis": "resubmit", "lesson": "no change"})
        return {"content": self.pad, "tool_calls": [{"id": "x", "name": name, "arguments": args}]}


def _knapsack(prepared):
    return [next(p.env for p in prepared if p.env.metadata["task"] == "knapsack")]


def test_runner_measures_improvement_and_recovery(envs, tmp_path):
    _, prepared = envs
    cfg = BenchConfig(modes=["rlc"], repeats=1, max_rounds=4, workers=1)
    rows = run_grid(_knapsack(prepared), cfg, lambda: ImprovingClient(), tmp_path, "scripted", log=lambda s: None)
    (row,) = rows
    assert row["error"] is None
    assert row["best"] > 1.0 and row["beat_reference"]
    assert row["curve"][0] > 0 and row["curve"][1] == row["curve"][0] and row["curve"][2] == row["best"]
    assert row["recovery_rate"] == 1.0  # round 2 regressed, round 3 set a new best
    assert row["audit_passed"]
    assert (tmp_path / "runs/bench-knapsack/rlc/rep0/trajectory.json").exists()


def test_small_window_forces_compaction_in_both_modes(envs, tmp_path):
    _, prepared = envs
    cfg = BenchConfig(modes=["rlc", "truncate"], repeats=1, max_rounds=6, window=2500, keep_rounds=1, workers=2)
    rows = run_grid(_knapsack(prepared), cfg, lambda: ImprovingClient(pad=150), tmp_path, "scripted",
                    log=lambda s: None)
    by_mode = {r["mode"]: r for r in rows}
    assert by_mode["rlc"]["compactions"] > 0
    assert by_mode["truncate"]["compactions"] > 0
    traj = json.loads((tmp_path / "runs/bench-knapsack/rlc/rep0/trajectory.json").read_text())
    assert traj["metadata"]["rlc"]["mode"] == "rlc"


def test_runs_resume_and_report(envs, tmp_path):
    _, prepared = envs
    cfg = BenchConfig(modes=["rlc", "truncate"], repeats=2, max_rounds=3, workers=2)
    envs_ = [p.env for p in prepared]
    run_grid(envs_, cfg, SubmitOnlyClient, tmp_path, "dry-run", log=lambda s: None)
    marker = tmp_path / "runs/bench-knapsack/rlc/rep0/result.json"
    stamp = marker.stat().st_mtime_ns
    rows = run_grid(envs_, cfg, SubmitOnlyClient, tmp_path, "dry-run", log=lambda s: None)
    assert marker.stat().st_mtime_ns == stamp  # skipped, not rerun
    assert len(rows) == 16 and all(r["best"] == 0.0 for r in rows)
    s = write_report(tmp_path)
    assert s["paired"] and not s["paired"][0]["significant"]
    md = (tmp_path / "report.md").read_text()
    assert "rlc − truncate" in md and (tmp_path / "curves.csv").exists()


def test_paired_bootstrap_detects_a_clear_difference():
    rows = []
    for env in ("e1", "e2", "e3"):
        for rep in range(4):
            for mode, best in (("rlc", 1.2 + 0.01 * rep), ("truncate", 0.8 + 0.01 * rep)):
                rows.append({"env_id": env, "rep": rep, "mode": mode, "model": "m", "error": None, "best": best,
                             "auc_rounds": best / 2, "beat_reference": best >= 1, "audit_passed": True,
                             "productive_rounds": 3, "mean_gain": 0.2, "recovery_rate": None,
                             "wasted_turn_share": 0.1, "compactions": 1, "curve": [best]})
    s = summarize(rows)
    best = next(p for p in s["paired"] if p["metric"] == "best")
    assert best["significant"] and best["mean_diff"] == pytest.approx(0.4)
    assert "yes" in render_markdown(s)


class RecordingClient(SubmitOnlyClient):
    def __init__(self):
        super().__init__()
        self.systems = []

    def chat(self, messages, tools=None, **kw):
        self.systems.append(messages[0]["content"])
        return super().chat(messages, tools, **kw)


def test_no_think_reaches_the_system_prompt(envs, tmp_path):
    _, prepared = envs
    rec = RecordingClient()
    cfg = BenchConfig(modes=["rlc"], repeats=1, max_rounds=2, workers=1, no_think=True)
    run_grid(_knapsack(prepared), cfg, lambda: rec, tmp_path, "dry-run", log=lambda s: None)
    assert rec.systems and all(s.rstrip().endswith("/no_think") for s in rec.systems)


class FlakyError(Exception):
    def __init__(self, code):
        super().__init__(f"Error code: {code}")
        self.status_code = code


class FlakyClient:
    def __init__(self, codes):
        self.codes = list(codes)
        self.n = 0

    def chat(self, messages, tools=None, **kw):
        self.n += 1
        if self.codes:
            raise FlakyError(self.codes.pop(0))
        return {"content": "", "tool_calls": []}


def test_rate_limits_are_retried_and_client_errors_are_not(monkeypatch):
    from mk33.bench import run as runmod
    monkeypatch.setattr(runmod.time, "sleep", lambda s: None)
    c = runmod.RetryingClient(FlakyClient([429, 429, 429, 500]))
    assert c.chat([]) == {"content": "", "tool_calls": []}
    assert c.retries == 4
    with pytest.raises(FlakyError):
        runmod.RetryingClient(FlakyClient([402])).chat([])
    with pytest.raises(FlakyError):
        runmod.RetryingClient(FlakyClient([500, 500, 500])).chat([])


def test_pacing_spaces_calls(monkeypatch):
    from mk33.bench import run as runmod
    slept = []
    monkeypatch.setattr(runmod.time, "sleep", lambda s: slept.append(s))
    runmod.RetryingClient._next_slot = 0.0
    c = runmod.RetryingClient(FlakyClient([]), rpm=60)
    for _ in range(3):
        c.chat([])
    assert len(slept) >= 1 and all(0 < s <= 2.0 for s in slept)
    runmod.RetryingClient._next_slot = 0.0
