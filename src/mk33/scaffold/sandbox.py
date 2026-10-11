"""Execution sandboxes for the agent's tools.

``LocalSandbox`` runs commands as subprocesses in the workspace (development and
tests). ``DockerSandbox`` runs them inside a long-lived container whose only mount
is the workspace; the scorer directory is never mounted.
"""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ExecResult:
    returncode: int
    output: str
    timed_out: bool = False


class LocalSandbox:
    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.jobs_dir = self.workspace / ".jobs"
        self._procs: dict[str, subprocess.Popen] = {}
        self._read_pos: dict[str, int] = {}

    def exec(self, command: str, timeout_s: int) -> ExecResult:
        try:
            p = subprocess.run(
                command, shell=True, cwd=self.workspace, capture_output=True, text=True,
                timeout=timeout_s, start_new_session=True,
            )
            return ExecResult(p.returncode, p.stdout + p.stderr)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            return ExecResult(-1, out + f"\n[timed out after {timeout_s}s]", timed_out=True)

    def start_job(self, command: str) -> str:
        self.jobs_dir.mkdir(exist_ok=True)
        jid = uuid.uuid4().hex[:8]
        log = open(self.jobs_dir / f"{jid}.log", "w")
        self._procs[jid] = subprocess.Popen(
            command, shell=True, cwd=self.workspace, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._read_pos[jid] = 0
        return jid

    def wait_job(self, jid: str, timeout_s: int) -> tuple[str, str, bool]:
        """Block until the job ends or the timeout passes.

        Returns (status, new_log_text, timed_out).
        """
        if jid not in self._procs:
            return "unknown", f"no job {jid}", False
        p = self._procs[jid]
        timed_out = False
        try:
            p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
        log_path = self.jobs_dir / f"{jid}.log"
        text = log_path.read_text(errors="replace") if log_path.exists() else ""
        new = text[self._read_pos[jid]:]
        self._read_pos[jid] = len(text)
        status = "running" if p.poll() is None else f"exited {p.returncode}"
        return status, new, timed_out

    def kill_all(self) -> None:
        for p in self._procs.values():
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass


class DockerSandbox(LocalSandbox):
    """Runs commands in a container started with only the workspace mounted at /workspace."""

    def __init__(self, workspace: str | Path, image: str, gpus: str | None = None, network: bool = True):
        super().__init__(workspace)
        self.name = f"or-sbx-{uuid.uuid4().hex[:8]}"
        cmd = ["docker", "run", "-d", "--name", self.name, "-v", f"{self.workspace}:/workspace",
               "-w", "/workspace"]
        if gpus:
            cmd += ["--gpus", gpus]
        if not network:
            cmd += ["--network", "none"]
        subprocess.run(cmd + [image, "sleep", "infinity"], check=True, capture_output=True)

    def exec(self, command: str, timeout_s: int) -> ExecResult:
        wrapped = f"docker exec {self.name} bash -lc {shlex.quote(command)}"
        return super().exec(wrapped, timeout_s)

    def start_job(self, command: str) -> str:
        return super().start_job(f"docker exec {self.name} bash -lc {shlex.quote(command)}")

    def close(self) -> None:
        self.kill_all()
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
