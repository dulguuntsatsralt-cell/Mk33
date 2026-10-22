"""Bench wiring for the memory study: retrieval modes, kv modes, the CLI guard and
the report's KV rows."""

import json

import pytest

from mk33.bench import BenchConfig, SubmitOnlyClient, prepare, run_grid, write_report
from mk33.bench.run import parse_mode
from test_bench import ImprovingClient


@pytest.fixture(scope="module")
def knapsack(tmp_path_factory):
    return [prepare(tmp_path_factory.mktemp("bench"), ["knapsack"])[0].env]


def test_parse_mode():
    assert parse_mode("rlc") == ("rlc", None)
    assert parse_mode("rlc+retrieve") == ("rlc+retrieve", None)
    assert parse_mode("kv:keynorm") == ("truncate", "keynorm")
    for bad in ("rag", "kv:snapkv"):
        with pytest.raises(ValueError):
            parse_mode(bad)
    with pytest.raises(ValueError):
        BenchConfig(modes=["rlc", "kv:nope"])


def test_rlc_for_maps_bench_settings():
    cfg = BenchConfig(window=8000, kv_text_window=30000, retrieve_k=3, retrieve_budget=900)
    r = cfg.rlc_for("retrieve")
    assert (r.mode, r.window, r.retrieve_k, r.retrieve_budget) == ("retrieve", 8000, 3, 900)
    k = cfg.rlc_for("kv:streaming")
    assert (k.mode, k.window) == ("truncate", 30000)  # text is only cut at the model's limit


def test_retrieval_modes_run_compact_and_retrieve(knapsack, tmp_path):
    cfg = BenchConfig(modes=["retrieve", "rlc+retrieve"], repeats=1, max_rounds=6, window=2500, keep_rounds=1,
                      workers=2, retrieve_budget=300, retrieve_item_chars=600)
    rows = run_grid(knapsack, cfg, lambda: ImprovingClient(pad=150), tmp_path, "scripted", log=lambda s: None)
    for row in rows:
        assert row["error"] is None and row["best"] > 1.0 and row["compactions"] > 0
        traj = json.loads((tmp_path / f"runs/bench-knapsack/{row['mode']}/rep0/trajectory.json").read_text(encoding="utf-8"))
        assert traj["metadata"]["rlc"]["mode"] == row["mode"]
        assert any(e["retrieved"] for e in traj["metadata"]["compactions"])


def test_kv_modes_need_the_hf_backend(tmp_path):
    from mk33.cli import main

    with pytest.raises(SystemExit, match="backend hf"):
        main(["bench-run", "--envs", str(tmp_path / "none.jsonl"), "--modes", "rlc,kv:streaming", "--dry-run",
              "--out", str(tmp_path / "o")])


def test_dry_run_cli_with_retrieval(knapsack, tmp_path):
    from mk33.cli import main

    envs = tmp_path / "envs.jsonl"
    envs.write_text(json.dumps(knapsack[0].to_dict()) + "\n", encoding="utf-8")
    assert main(["bench-run", "--envs", str(envs), "--dry-run", "--modes", "retrieve,rlc", "--repeats", "1",
                 "--max-rounds", "2", "--out", str(tmp_path / "out")]) == 0
    assert "retrieve" in (tmp_path / "out/report.md").read_text(encoding="utf-8")


class KVScriptFactory:
    """HF backend with a tiny random model and a scripted tokenizer (no downloads)."""

    takes_mode = True

    def __init__(self):
        from test_kv import ScriptTokenizer, tiny_model

        self.model, self.Tok = tiny_model(), ScriptTokenizer

    def __call__(self, mode):
        from mk33.kv import KVPolicy
        from mk33.kv.client import HFChatClient

        _, name = parse_mode(mode)
        call = '<tool_call>{"name": "submit", "arguments": {"hypothesis": "baseline", "lesson": "x"}}</tool_call>'
        tok = self.Tok(['<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>']
                       + [call] * 50)
        return HFChatClient(self.model, tok, KVPolicy(name or "full", budget=64, sinks=4, recent=16),
                            max_tokens=3, chunk=32)


def test_kv_modes_record_cache_stats_in_results_and_report(knapsack, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    cfg = BenchConfig(modes=["truncate", "kv:streaming"], repeats=1, max_rounds=2, workers=1)
    rows = run_grid(knapsack, cfg, KVScriptFactory(), tmp_path, "tiny", log=lambda s: None)
    by_mode = {r["mode"]: r for r in rows}
    assert all(r["error"] is None and r["rounds"] == 2 for r in rows)
    kv, full = by_mode["kv:streaming"], by_mode["truncate"]
    assert kv["kv_policy"] == "streaming@64" and full["kv_policy"] == "full"
    assert kv["kv_peak_cache_tokens"] <= 64 + 32 < full["kv_peak_cache_tokens"]
    assert kv["kv_evicted_share"] > 0 == full["kv_evicted_share"]
    write_report(tmp_path)
    md = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "peak KV cache (tokens)" in md and "kv:streaming" in md


def test_report_has_no_kv_rows_for_api_runs(knapsack, tmp_path):
    cfg = BenchConfig(modes=["rlc"], repeats=1, max_rounds=2, workers=1)
    run_grid(knapsack, cfg, SubmitOnlyClient, tmp_path, "dry-run", log=lambda s: None)
    write_report(tmp_path)
    assert "KV cache" not in (tmp_path / "report.md").read_text(encoding="utf-8")
