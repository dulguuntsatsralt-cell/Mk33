"""Fixed tool set (paper Section 3.2, Table "Tool set").

| tool                  | notes                                                         |
|-----------------------|---------------------------------------------------------------|
| bash                  | 10 min timeout; output truncated to ~8K tokens, head + tail   |
| read_file, edit_file  | edits return a unified diff (used for PALM provenance)        |
| run_job, wait_job     | blocking wait replaces repeated polling                       |
| web_search, web_fetch | pluggable providers; disabled when None                       |
| submit                | scores the solution; requires `hypothesis` and `lesson`       |
| answer                | final answer for research tasks                               |
"""

from __future__ import annotations

import difflib
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..envs.scorer import IsolatedScorer
from ..envs.spec import EnvSpec
from ..trajectory import Submission
from .sandbox import LocalSandbox

IGNORED_DIRS = {".git", ".jobs", "__pycache__", ".ipynb_checkpoints", "node_modules", ".venv"}
TEXT_SUFFIXES = {".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".yaml", ".yml", ".json", ".toml",
                 ".cfg", ".txt", ".md", ".sh", ".ini", ".js", ".ts", ".rs", ".go", ".java"}


def _schema(name: str, desc: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required},
    }}


TOOL_SCHEMAS = {
    "bash": _schema("bash", "Run a shell command in the workspace (10 min timeout).",
                    {"command": {"type": "string"}}, ["command"]),
    "read_file": _schema("read_file", "Read a file (optionally a line range).",
                         {"path": {"type": "string"}, "start": {"type": "integer"},
                          "end": {"type": "integer"}}, ["path"]),
    "edit_file": _schema("edit_file", "Replace old_str with new_str in a file. Empty old_str with a "
                         "missing file creates it with new_str.",
                         {"path": {"type": "string"}, "old_str": {"type": "string"},
                          "new_str": {"type": "string"}}, ["path", "old_str", "new_str"]),
    "run_job": _schema("run_job", "Start a long command in the background. Returns a job id.",
                       {"command": {"type": "string"}}, ["command"]),
    "wait_job": _schema("wait_job", "Block until a job finishes or timeout_s passes. Returns status "
                        "and new log lines.",
                        {"job_id": {"type": "string"}, "timeout_s": {"type": "integer"}}, ["job_id"]),
    "web_search": _schema("web_search", "Search the web.", {"query": {"type": "string"}}, ["query"]),
    "web_fetch": _schema("web_fetch", "Fetch a web page as text.", {"url": {"type": "string"}}, ["url"]),
    "submit": _schema("submit", "Score the current solution. Ends the round.",
                      {"hypothesis": {"type": "string", "description": "What this round tried and why."},
                       "lesson": {"type": "string", "description": "One line: what you learned."},
                       "family": {"type": "string", "description": "Optional short strategy name."}},
                      ["hypothesis", "lesson"]),
    "answer": _schema("answer", "Give the final answer (research tasks).",
                      {"answer": {"type": "string"}, "confidence": {"type": "number"},
                       "evidence": {"type": "array", "items": {"type": "string"}}}, ["answer"]),
}


@dataclass
class ToolConfig:
    bash_timeout_s: int = 600
    max_output_tokens: int = 8000
    max_wait_s: int = 1800
    solution_subdir: str = ""  # scored directory, relative to the workspace
    snapshot_max_bytes: int = 200_000  # per file, for provenance snapshots
    enabled: list[str] = field(default_factory=lambda: list(TOOL_SCHEMAS))


@dataclass
class ToolResult:
    observation: str
    diff: str | None = None
    submission: Submission | None = None
    timed_out: bool = False
    done: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


def truncate_middle(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + f"\n[... {len(text) - max_chars} chars truncated ...]\n" + text[-half:]


def iter_workspace_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
        for f in sorted(filenames):
            yield Path(dirpath) / f


def workspace_hash(root: str | Path, max_bytes: int = 5_000_000) -> str:
    root = Path(root)
    h = hashlib.sha256()
    for p in iter_workspace_files(root):
        rel = p.relative_to(root).as_posix()
        st = p.stat()
        h.update(rel.encode())
        if st.st_size <= max_bytes:
            h.update(hashlib.sha256(p.read_bytes()).digest())
        else:  # large data files: size + mtime is enough to detect change
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def snapshot_text_files(root: Path, max_bytes: int) -> dict[str, str]:
    out = {}
    for p in iter_workspace_files(root):
        if p.suffix in TEXT_SUFFIXES and p.stat().st_size <= max_bytes:
            out[p.relative_to(root).as_posix()] = p.read_text(errors="replace")
    return out


def diff_stat(diffs: list[str]) -> str:
    per: dict[str, list[int]] = {}
    for d in diffs:
        cur = None
        for line in d.splitlines():
            if line.startswith("+++ "):
                cur = line[4:].removeprefix("b/")
                per.setdefault(cur, [0, 0])
            elif cur and line.startswith("+") and not line.startswith("+++"):
                per[cur][0] += 1
            elif cur and line.startswith("-") and not line.startswith("---"):
                per[cur][1] += 1
    return "; ".join(f"{f} +{a} -{r}" for f, (a, r) in per.items()) or "no changes"


class ToolExecutor:
    def __init__(
        self,
        env: EnvSpec,
        sandbox: LocalSandbox | None = None,
        scorer: IsolatedScorer | None = None,
        cfg: ToolConfig | None = None,
        search: Callable[[str], str] | None = None,
        fetch: Callable[[str], str] | None = None,
    ):
        self.env = env
        self.cfg = cfg or ToolConfig()
        self.workspace = Path(env.workspace_dir).resolve()
        self.sandbox = sandbox or LocalSandbox(self.workspace)
        self.scorer = scorer
        self.search = search
        self.fetch = fetch
        self.round = 1
        self._round_diffs: list[str] = []

    def schemas(self) -> list[dict]:
        return [TOOL_SCHEMAS[n] for n in self.cfg.enabled if n in TOOL_SCHEMAS]

    def workspace_hash(self) -> str:
        return workspace_hash(self.workspace)

    def _resolve(self, path: str) -> Path:
        p = (self.workspace / path).resolve()
        p.relative_to(self.workspace)  # raises ValueError if outside
        return p

    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        if name not in self.cfg.enabled:
            return ToolResult(f"[tool {name} is not available]")
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return ToolResult(f"[unknown tool {name}]")
        try:
            return fn(**args)
        except TypeError as e:
            return ToolResult(f"[bad arguments for {name}: {e}]")
        except ValueError as e:
            return ToolResult(f"[error: {e}]")

    # ---------------------------------------------------------------- tools
    def _t_bash(self, command: str) -> ToolResult:
        r = self.sandbox.exec(command, self.cfg.bash_timeout_s)
        out = truncate_middle(r.output, self.cfg.max_output_tokens)
        return ToolResult(f"[exit {r.returncode}]\n{out}", timed_out=r.timed_out)

    def _t_read_file(self, path: str, start: int | None = None, end: int | None = None) -> ToolResult:
        p = self._resolve(path)
        if not p.exists():
            return ToolResult(f"[no such file: {path}]")
        lines = p.read_text(errors="replace").splitlines()
        s = max((start or 1) - 1, 0)
        e = end or len(lines)
        body = "\n".join(f"{i + 1}\t{l}" for i, l in enumerate(lines[s:e], start=s))
        return ToolResult(truncate_middle(body, self.cfg.max_output_tokens))

    def _t_edit_file(self, path: str, old_str: str, new_str: str) -> ToolResult:
        p = self._resolve(path)
        if not p.exists():
            if old_str:
                return ToolResult(f"[no such file: {path}]")
            before = ""
            p.parent.mkdir(parents=True, exist_ok=True)
            after = new_str
        else:
            before = p.read_text(errors="replace")
            n = before.count(old_str) if old_str else 0
            if n != 1:
                return ToolResult(f"[old_str must match exactly once; matched {n} times]")
            after = before.replace(old_str, new_str, 1)
        p.write_text(after)
        rel = p.relative_to(self.workspace).as_posix()
        diff = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{rel}", tofile=f"b/{rel}",
        ))
        self._round_diffs.append(diff)
        return ToolResult(diff or "[no change]", diff=diff or None)

    def _t_run_job(self, command: str) -> ToolResult:
        jid = self.sandbox.start_job(command)
        return ToolResult(f"started job {jid}")

    def _t_wait_job(self, job_id: str, timeout_s: int = 600) -> ToolResult:
        timeout_s = max(1, min(int(timeout_s), self.cfg.max_wait_s))
        status, new, timed_out = self.sandbox.wait_job(job_id, timeout_s)
        body = truncate_middle(new, self.cfg.max_output_tokens) if new else "(no new output)"
        return ToolResult(f"[job {job_id}: {status}]\n{body}", timed_out=timed_out)

    def _t_web_search(self, query: str) -> ToolResult:
        if self.search is None:
            return ToolResult("[web_search is disabled for this task]")
        return ToolResult(truncate_middle(self.search(query), self.cfg.max_output_tokens))

    def _t_web_fetch(self, url: str) -> ToolResult:
        if self.fetch is None:
            return ToolResult("[web_fetch is disabled for this task]")
        return ToolResult(truncate_middle(self.fetch(url), self.cfg.max_output_tokens))

    def _t_submit(self, hypothesis: str, lesson: str, family: str = "") -> ToolResult:
        if not hypothesis.strip() or not lesson.strip():
            return ToolResult("[submit requires non-empty hypothesis and lesson]")
        sol_dir = self.workspace / self.cfg.solution_subdir
        if self.scorer is None:
            return ToolResult("[no scorer configured]")
        res = self.scorer.score(sol_dir)
        sub = Submission(
            round=self.round, score=res.score, normalized=res.normalized, ok=res.ok,
            logs=res.logs[-2000:], hypothesis=hypothesis.strip(), lesson=lesson.strip(),
            error_signature=res.error_signature,
            solution_files=snapshot_text_files(sol_dir, self.cfg.snapshot_max_bytes),
        )
        sub_meta = {"family": family, "change": diff_stat(self._round_diffs)}
        self._round_diffs = []
        self.round += 1
        if res.ok:
            norm = f" (normalized {res.normalized:.4f})" if res.normalized is not None else ""
            obs = f"[round {sub.round} scored] score {res.score:.6g}{norm}\n{res.logs[-1500:]}"
        else:
            obs = f"[round {sub.round} failed] {res.error_signature}\n{res.logs[-1500:]}"
        return ToolResult(obs, submission=sub, meta=sub_meta)

    def _t_answer(self, answer: str, confidence: float | None = None,
                  evidence: list[str] | None = None) -> ToolResult:
        sub = Submission(round=self.round, score=None, normalized=None, ok=True,
                         hypothesis=answer, lesson="",
                         solution_files={"answer.txt": answer})
        return ToolResult(f"[final answer recorded] {answer}", submission=sub, done=True)
