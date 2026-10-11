"""Trajectory schema shared by the scaffold, PALM, selection, and training.

A trajectory is a list of assistant ``Turn`` objects. Each turn holds the model's
reasoning, exactly one tool call, the observation it produced, and bookkeeping the
scaffold records automatically (workspace hash, diff, submission result).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


@dataclass
class Submission:
    """Result of one ``submit`` call (one round)."""

    round: int
    score: float | None
    normalized: float | None
    ok: bool = True
    logs: str = ""
    hypothesis: str = ""
    lesson: str = ""
    error_signature: str | None = None
    # Snapshot of the solution files that were scored: path -> content.
    solution_files: dict[str, str] = field(default_factory=dict)


@dataclass
class Turn:
    """One assistant turn: reasoning + one tool call + its observation."""

    index: int
    round: int
    reasoning: str
    tool: str
    args: dict[str, Any]
    observation: str
    workspace_hash: str = ""
    diff: str | None = None  # unified diff for edit_file turns
    submission: Submission | None = None
    timed_out: bool = False  # wait_job returned on timeout
    wall_time_s: float = 0.0
    elapsed_s: float = 0.0  # time since the run started, at the end of this turn

    @property
    def is_submit(self) -> bool:
        return self.tool == "submit" and self.submission is not None


@dataclass
class Trajectory:
    env_id: str
    domain: str  # "ml" | "judge" | "research"
    task_statement: str
    turns: list[Turn] = field(default_factory=list)
    pinned: dict[str, str] = field(default_factory=dict)  # system prompt, skills, ...
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ queries
    def submissions(self) -> list[Submission]:
        return [t.submission for t in self.turns if t.submission is not None]

    def scored_submissions(self) -> list[Submission]:
        return [s for s in self.submissions() if s.ok and s.normalized is not None]

    def best_normalized(self) -> float | None:
        scores = [s.normalized for s in self.scored_submissions()]
        return max(scores) if scores else None

    def turns_in_round(self, r: int) -> list[Turn]:
        return [t for t in self.turns if t.round == r]

    # ------------------------------------------------------------- serialization
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Trajectory":
        turns = []
        for t in d.get("turns", []):
            t = dict(t)
            sub = t.get("submission")
            t["submission"] = Submission(**sub) if sub else None
            turns.append(Turn(**t))
        return cls(
            env_id=d["env_id"],
            domain=d["domain"],
            task_statement=d.get("task_statement", ""),
            turns=turns,
            pinned=d.get("pinned", {}),
            metadata=d.get("metadata", {}),
        )

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        return cls.from_dict(json.loads(Path(path).read_text()))


def iter_jsonl(path: str | Path) -> Iterator[Trajectory]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Trajectory.from_dict(json.loads(line))


def write_jsonl(trajs: Iterable[Trajectory], path: str | Path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for t in trajs:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
            n += 1
    return n
