"""Mk33 bench: small CPU-only environments for measuring self-improvement.

    mk33 bench-prepare --out bench/envs
    mk33 bench-run --envs bench/envs/envs.jsonl --model <name> --base-url <url> --out bench/results
    mk33 bench-report --out bench/results

See docs/mk33/BENCH.md.
"""

from .prepare import PreparedEnv, prepare
from .report import collect_trajectories, compare_dirs, render_comparison, summarize, write_report
from .run import BenchConfig, RetryingClient, best_so_far_curve, run_grid, run_one
from .tasks import TASKS, TASKS_BY_ID, BenchTask

__all__ = [
    "TASKS", "TASKS_BY_ID", "BenchTask", "PreparedEnv", "prepare", "BenchConfig", "RetryingClient",
    "best_so_far_curve", "run_grid", "run_one", "summarize", "write_report", "SubmitOnlyClient",
    "collect_trajectories", "compare_dirs", "render_comparison",
]


class SubmitOnlyClient:
    """A model stand-in for dry runs: reads the task once, then submits the baseline every turn.

    Exercises the whole pipeline (workspace copy, scorer, ledger, metrics, report)
    without an API key. Expect best = 0 (the baseline) everywhere.
    """

    def __init__(self):
        self.turn = 0

    def chat(self, messages, tools=None, **kw):
        self.turn += 1
        if self.turn == 1:
            call = {"id": "x", "name": "read_file", "arguments": {"path": "README.md"}}
        else:
            call = {"id": "x", "name": "submit",
                    "arguments": {"hypothesis": "resubmit the baseline", "lesson": "dry run"}}
        return {"content": "", "tool_calls": [call]}
