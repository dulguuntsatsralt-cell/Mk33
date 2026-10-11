"""Environment specification (paper Section 3.1)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ScorerSpec:
    """How a solution is scored. Runs outside the agent's sandbox."""

    command: list[str]  # e.g. ["python", "score.py", "--solution", "{solution}"]
    script_path: str  # path of the scoring script inside the scorer dir
    script_sha256: str = ""  # verified before every run
    scorer_dir: str = ""  # holds the script + held-out data; never mounted for the agent
    higher_is_better: bool = True
    timeout_s: int = 3600
    docker_image: str | None = None  # None -> local subprocess (tests, dev only)


@dataclass
class EnvSpec:
    env_id: str
    domain: str  # "ml" | "judge"
    source: str  # repo URL@commit or judge problem ID
    task_statement: str
    workspace_dir: str  # what the agent sees: code + public dev split
    scorer: ScorerSpec
    baseline_score: float | None = None
    reference_scores: list[float] = field(default_factory=list)  # k seeds
    dataset_urls: list[str] = field(default_factory=list)
    dataset_hashes: list[str] = field(default_factory=list)
    problem_ids: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def reference_score(self) -> float | None:
        if not self.reference_scores:
            return None
        return sum(self.reference_scores) / len(self.reference_scores)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EnvSpec":
        d = dict(d)
        d["scorer"] = ScorerSpec(**d["scorer"])
        return cls(**d)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EnvSpec":
        return cls.from_dict(json.loads(Path(path).read_text()))
