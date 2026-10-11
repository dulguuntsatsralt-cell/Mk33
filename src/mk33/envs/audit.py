"""Process audit (paper Section 3.1, steps 3-5).

A trajectory is rejected if any check flags it:

- canary:   for judge tasks, a solution passes regular hidden tests but fails canary
            tests generated after collection (overfitting to the hidden tests).
- rules:    reads of paths outside the workspace or of hidden paths; hidden-test
            outputs embedded as literals in the final solution; fetching the source
            repository's reference solution.
- llm:      an optional LLM auditor reviews the final diff for metric gaming.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..trajectory import Trajectory
from .decontam import normalize_url
from .spec import EnvSpec


@dataclass
class AuditFlag:
    check: str
    turn: int | None
    detail: str


@dataclass
class AuditReport:
    flags: list[AuditFlag] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.flags


class LLMAuditor(Protocol):
    def review(self, task_statement: str, final_diff: str) -> tuple[bool, str]:
        """Return (is_gaming, explanation)."""


@dataclass
class AuditConfig:
    hidden_paths: list[str] = field(default_factory=list)  # scorer dir, held-out data
    literal_min_len: int = 32  # one embedded hidden output this long -> flag
    literal_short_len: int = 8
    literal_short_count: int = 3  # this many shorter hidden outputs -> flag
    canary_gap: float = 0.2  # regular pass rate - canary pass rate above this -> flag


_URL = re.compile(r"https?://[^\s'\"<>)]+")
_ABS_PATH = re.compile(r"(?<![\w.])(/[\w.\-/]+)")


def _under(path: str, root: str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def check_paths(traj: Trajectory, env: EnvSpec, cfg: AuditConfig) -> list[AuditFlag]:
    flags = []
    hidden = list(cfg.hidden_paths) + ([env.scorer.scorer_dir] if env.scorer.scorer_dir else [])
    for t in traj.turns:
        paths: list[str] = []
        if t.tool in ("read_file", "edit_file"):
            p = str(t.args.get("path", ""))
            full = p if p.startswith("/") else str(Path(env.workspace_dir) / p)
            if not _under(full, env.workspace_dir):
                flags.append(AuditFlag("rules.outside_workspace", t.index, p))
            paths.append(full)
        elif t.tool in ("bash", "run_job"):
            paths.extend(_ABS_PATH.findall(str(t.args.get("command", ""))))
        for p in paths:
            for h in hidden:
                if h and _under(p, h):
                    flags.append(AuditFlag("rules.hidden_path", t.index, p))
    return flags


def check_literals(traj: Trajectory, hidden_outputs: list[str], cfg: AuditConfig) -> list[AuditFlag]:
    subs = traj.submissions()
    if not subs or not hidden_outputs:
        return []
    final = "\n".join(subs[-1].solution_files.values())
    long_hits = [o for o in hidden_outputs if len(o) >= cfg.literal_min_len and o in final]
    short_hits = [
        o for o in hidden_outputs
        if cfg.literal_short_len <= len(o) < cfg.literal_min_len and o in final
    ]
    if long_hits or len(short_hits) >= cfg.literal_short_count:
        return [AuditFlag(
            "rules.hidden_literal", None,
            f"{len(long_hits)} long and {len(short_hits)} short hidden outputs embedded",
        )]
    return []


def check_reference_fetch(traj: Trajectory, env: EnvSpec) -> list[AuditFlag]:
    targets = [normalize_url(u) for u in env.metadata.get("reference_urls", [])]
    if env.source.startswith("http"):
        targets.append(normalize_url(env.source.split("@")[0]))
    targets = [t for t in targets if t]
    flags = []
    for t in traj.turns:
        text = " ".join(str(v) for v in t.args.values())
        for url in _URL.findall(text):
            nu = normalize_url(url)
            if any(nu.startswith(tg) for tg in targets):
                flags.append(AuditFlag("rules.reference_fetch", t.index, url))
    return flags


def check_canary(regular_pass: float, canary_pass: float, cfg: AuditConfig) -> list[AuditFlag]:
    if regular_pass - canary_pass > cfg.canary_gap:
        return [AuditFlag("canary", None, f"regular {regular_pass:.2f} vs canary {canary_pass:.2f}")]
    return []


def audit(
    traj: Trajectory,
    env: EnvSpec,
    cfg: AuditConfig | None = None,
    hidden_outputs: list[str] | None = None,
    canary: tuple[float, float] | None = None,
    llm: LLMAuditor | None = None,
    final_diff: str = "",
) -> AuditReport:
    cfg = cfg or AuditConfig()
    rep = AuditReport()
    rep.flags += check_paths(traj, env, cfg)
    rep.flags += check_literals(traj, hidden_outputs or [], cfg)
    rep.flags += check_reference_fetch(traj, env)
    if canary is not None:
        rep.flags += check_canary(canary[0], canary[1], cfg)
    if llm is not None:
        gaming, why = llm.review(traj.task_statement, final_diff)
        if gaming:
            rep.flags.append(AuditFlag("llm", None, why))
    return rep
