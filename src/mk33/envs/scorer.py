"""Isolated scorer (paper Section 3.1, "Scorer isolation").

The held-out split and scoring script live in a scorer directory that is never
mounted into the agent's sandbox. Before every run the scorer verifies the SHA-256
of its own script. In docker mode the solution is mounted read-only and the scorer
container has no network.

The scoring command must print a JSON object with a numeric ``score`` field as the
last non-empty line of stdout, e.g. ``{"score": 0.873}``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .normalize import normalized_score
from .spec import EnvSpec, sha256_file


class ScorerTampered(RuntimeError):
    pass


@dataclass
class ScoreResult:
    ok: bool
    score: float | None
    normalized: float | None
    logs: str
    error_signature: str | None = None


def _error_signature(text: str) -> str | None:
    """The most specific error line: the last 'XxxError: ...' line, else any error marker."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for s in reversed(lines):
        head = s.split(":", 1)[0]
        if ":" in s and (head.endswith("Error") or head.endswith("Exception")):
            return s[:200]
    for s in lines:
        if any(k in s for k in ("error:", "FAILED", "Killed", "Traceback")):
            return s[:200]
    return None


def _parse_score(stdout: str) -> float:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        return float(obj["score"])
    raise ValueError("scorer printed nothing")


class IsolatedScorer:
    def __init__(self, env: EnvSpec, log_limit: int = 4000):
        self.env = env
        self.spec = env.scorer
        self.log_limit = log_limit

    def verify(self) -> None:
        script = Path(self.spec.scorer_dir) / self.spec.script_path
        if not self.spec.script_sha256:
            raise ScorerTampered("scorer has no recorded sha256; refusing to run")
        actual = sha256_file(script)
        if actual != self.spec.script_sha256:
            raise ScorerTampered(f"scoring script hash mismatch: {actual} != {self.spec.script_sha256}")

    def _command(self, solution_dir: Path) -> list[str]:
        if self.spec.docker_image:
            return [
                "docker", "run", "--rm", "--network", "none",
                "-v", f"{solution_dir}:/solution:ro",
                "-v", f"{Path(self.spec.scorer_dir).resolve()}:/scorer:ro",
                "-w", "/scorer", self.spec.docker_image,
                *[c.format(solution="/solution", scorer_dir="/scorer") for c in self.spec.command],
            ]
        return [
            c.format(solution=str(solution_dir), scorer_dir=str(Path(self.spec.scorer_dir).resolve()))
            for c in self.spec.command
        ]

    def score(self, solution_dir: str | Path) -> ScoreResult:
        self.verify()
        with tempfile.TemporaryDirectory(prefix="or-score-") as tmp:
            # Score a frozen copy so the agent cannot change files mid-scoring.
            snap = Path(tmp) / "solution"
            shutil.copytree(solution_dir, snap, ignore=shutil.ignore_patterns(".git", "__pycache__"))
            cwd = None if self.spec.docker_image else self.spec.scorer_dir
            try:
                proc = subprocess.run(
                    self._command(snap), cwd=cwd, capture_output=True, text=True,
                    timeout=self.spec.timeout_s,
                )
            except subprocess.TimeoutExpired:
                return ScoreResult(False, None, None, "scorer timed out", "TimeoutExpired")
        logs = (proc.stdout + proc.stderr)[-self.log_limit :]
        if proc.returncode != 0:
            return ScoreResult(False, None, None, logs, _error_signature(logs) or f"exit {proc.returncode}")
        try:
            s = _parse_score(proc.stdout)
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            return ScoreResult(False, None, None, logs, f"bad scorer output: {e}")
        norm = None
        if self.env.baseline_score is not None and self.env.reference_score is not None:
            norm = normalized_score(
                s, self.env.baseline_score, self.env.reference_score, self.spec.higher_is_better
            )
        return ScoreResult(True, s, norm, logs)
