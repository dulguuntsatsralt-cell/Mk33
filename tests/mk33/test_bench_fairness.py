"""Bench fixes from the first Kaggle grid: equal token budgets, bounded tool output,
the baseline floor, and a summary of why submissions failed."""

import json

import pytest

from mk33.bench import BenchConfig, prepare, run_grid
from mk33.bench.run import run_one


@pytest.fixture(scope="module")
def regression(tmp_path_factory):
    return prepare(tmp_path_factory.mktemp("envs"), tasks=["regression"])[0].env


class ReadCsvThenSubmit:
    """Reads the training CSV (about 80 KB of numbers), then submits."""

    def __init__(self):
        self.t = 0
        self.observations = []

    def chat(self, messages, tools=None, **kw):
        self.t += 1
        if messages[-1].get("role") == "tool":
            self.observations.append(messages[-1]["content"])
        call = ({"name": "read_file", "arguments": {"path": "train.csv"}} if self.t == 1 else
                {"name": "submit", "arguments": {"hypothesis": "baseline", "lesson": "read the data"}})
        return {"content": "", "tool_calls": [{"id": f"c{self.t}", **call}]}


class CountingClient(ReadCsvThenSubmit):
    """Exposes count_tokens like the local HF client; counts every character as a token."""

    calls = 0

    def count_tokens(self, text):
        CountingClient.calls += 1
        return len(text)


def test_tool_output_is_capped_relative_to_the_window(regression, tmp_path):
    client = ReadCsvThenSubmit()
    cfg = BenchConfig(modes=["truncate"], repeats=1, max_rounds=1, max_tool_calls=3, window=8000)
    run_one(regression, "truncate", 0, cfg, client, tmp_path, "scripted")
    read = client.observations[0]
    assert "chars truncated" in read
    assert len(read) <= (8000 // 8) * 4 + 200  # window // 8 tokens, at about 4 characters each


def test_local_model_tokenizer_is_used_for_the_window(regression, tmp_path):
    CountingClient.calls = 0
    cfg = BenchConfig(modes=["rlc"], repeats=1, max_rounds=1, max_tool_calls=3, window=8000)
    run_one(regression, "rlc", 0, cfg, CountingClient(), tmp_path, "scripted")
    assert CountingClient.calls > 0


class WorseThenCrash:
    """Submits a solution worse than the baseline, then one that crashes."""

    def __init__(self):
        self.t = 0

    def chat(self, messages, tools=None, **kw):
        self.t += 1
        steps = [
            {"name": "edit_file", "arguments": {"path": "solution.py", "old_str": "", "new_str":
                "def fit_predict(train_X, train_y, test_X):\n    return [100.0 for _ in test_X]\n"}},
            {"name": "submit", "arguments": {"hypothesis": "constant 100", "lesson": "much worse"}},
            {"name": "edit_file", "arguments": {"path": "solution.py", "old_str": "", "new_str":
                "import not_a_real_module\n"}},
            {"name": "submit", "arguments": {"hypothesis": "missing module", "lesson": "crashes"}},
        ]
        call = steps[min(self.t, len(steps)) - 1]
        return {"content": "", "tool_calls": [{"id": f"c{self.t}", **call}]}


def test_best_never_below_baseline_and_errors_are_summarized(regression, tmp_path):
    cfg = BenchConfig(modes=["rlc"], repeats=1, max_rounds=2, max_tool_calls=6, window=8000)
    row = run_one(regression, "rlc", 0, cfg, WorseThenCrash(), tmp_path, "scripted")
    assert row["best_submitted"] < 0  # the only scored submission was worse than the baseline
    assert row["best"] == 0.0         # but the agent still has the baseline
    assert list(row["errors"]) == ["ModuleNotFoundError: No module named 'not_a_real_module'"]
    saved = json.loads((tmp_path / "runs/bench-regression/rlc/rep0/result.json").read_text(encoding="utf-8"))
    assert saved["errors"] == row["errors"]


class TextCallClient:
    """Writes every call as JSON text, like the model after its first compaction."""

    def __init__(self):
        self.text_tool_calls = 0

    def chat(self, messages, tools=None, **kw):
        from mk33.toolcalls import calls_from_text, tool_names

        text = '```json\n{"name": "submit", "arguments": {"hypothesis": "baseline", "lesson": "text call"}}\n```'
        calls = calls_from_text(text, tool_names(tools))
        self.text_tool_calls += len(calls)
        return {"content": text, "tool_calls": calls}


def test_recovered_text_calls_are_run_and_reported(regression, tmp_path):
    cfg = BenchConfig(modes=["rlc"], repeats=1, max_rounds=2, max_tool_calls=4, window=8000)
    row = run_one(regression, "rlc", 0, cfg, TextCallClient(), tmp_path, "scripted")
    assert row["rounds"] == 2 and row["text_tool_calls"] == 2
