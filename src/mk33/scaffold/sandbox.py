"""Execution sandboxes for the agent's tools.

``LocalSandbox`` runs commands as subprocesses in the workspace (development, tests,
and the local app). ``DockerSandbox`` runs them inside a long-lived container whose
only mount is the workspace; the scorer directory is never mounted.

``LocalSandbox`` works on Linux, macOS and Windows. Each command runs in its own
process group so that a timeout or ``kill_all`` stops the whole process tree, not just
the shell. On Windows, commands run in Git Bash when it is installed (the agent is
prompted for bash) and fall back to ``cmd.exe`` otherwise; ``shell_name`` says which.
"""

from __future__ import annotations

import os
import shlex
import shutil
import signal
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

IS_WINDOWS = os.name == "nt"

# Typical Git for Windows locations. ``C:\Windows\System32\bash.exe`` is the WSL
# launcher, which runs in a separate Linux filesystem, so it is skipped on purpose.
_GIT_BASH_CANDIDATES = [
    r"C:\Program Files\Git\bin\bash.exe",
    r"C:\Program Files (x86)\Git\bin\bash.exe",
]


@dataclass
class ExecResult:
    returncode: int
    output: str
    timed_out: bool = False


def find_windows_bash() -> str | None:
    """Git Bash on Windows, or None. ``MK33_BASH`` overrides the search."""
    override = os.environ.get("MK33_BASH")
    if override:
        return override if Path(override).exists() else None
    cands = list(_GIT_BASH_CANDIDATES)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        cands.append(str(Path(local) / "Programs" / "Git" / "bin" / "bash.exe"))
    found = shutil.which("bash")
    if found and "system32" not in found.lower():
        cands.insert(0, found)
    for c in cands:
        if Path(c).exists():
            return c
    return None


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONUTF8", "1")  # child Python scripts read and print UTF-8 on Windows too
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def kill_tree(p: subprocess.Popen) -> None:
    """Stop a process and everything it started."""
    if p.poll() is not None:
        return
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(p.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if not IS_WINDOWS:
                os.killpg(p.pid, signal.SIGKILL)
            p.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass


def shell_note(sandbox: "LocalSandbox") -> str:
    """One line for the system prompt when the shell is not plain Linux bash."""
    if not IS_WINDOWS:
        return ""
    if getattr(sandbox, "shell_name", "bash") == "cmd":
        return ("Note: this machine runs Windows. The `bash` tool runs commands in cmd.exe, not bash, "
                "so use cmd syntax. Run Python with `python`, not `python3`.")
    return "Note: this machine runs Windows with Git Bash. Run Python with `python`, not `python3`."


class LocalSandbox:
    def __init__(self, workspace: str | Path, shell: str | None = None):
        """``shell``: None picks the platform default; on Windows "cmd" forces cmd.exe and
        any other value is taken as the path to a bash executable."""
        self.workspace = Path(workspace).resolve()
        self.jobs_dir = self.workspace / ".jobs"
        self._procs: dict[str, subprocess.Popen] = {}
        self._read_pos: dict[str, int] = {}
        self._bash = None
        if IS_WINDOWS and shell != "cmd":
            self._bash = shell or find_windows_bash()
        self.shell_name = "bash" if (not IS_WINDOWS or self._bash) else "cmd"

    def _popen(self, command: str, **kw) -> subprocess.Popen:
        kw.setdefault("cwd", self.workspace)
        kw.setdefault("env", _child_env())
        if IS_WINDOWS:
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            if self._bash:
                return subprocess.Popen([self._bash, "-c", command], **kw)
            return subprocess.Popen(command, shell=True, **kw)
        return subprocess.Popen(command, shell=True, start_new_session=True, **kw)

    def exec(self, command: str, timeout_s: int) -> ExecResult:
        p = self._popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        try:
            out, _ = p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_tree(p)
            try:
                out, _ = p.communicate(timeout=5)
            except (subprocess.TimeoutExpired, ValueError):
                out = b""
            text = (out or b"").decode("utf-8", errors="replace")
            return ExecResult(-1, text + f"\n[timed out after {timeout_s}s]", timed_out=True)
        return ExecResult(p.returncode, (out or b"").decode("utf-8", errors="replace"))

    def start_job(self, command: str) -> str:
        self.jobs_dir.mkdir(exist_ok=True)
        jid = uuid.uuid4().hex[:8]
        # The child gets its own copy of the handle; closing ours right away matters on
        # Windows, where an open handle stops the run directory from being deleted.
        with open(self.jobs_dir / f"{jid}.log", "wb") as log:
            self._procs[jid] = self._popen(command, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL)
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
        data = log_path.read_bytes() if log_path.exists() else b""
        new = data[self._read_pos[jid]:].decode("utf-8", errors="replace")
        self._read_pos[jid] = len(data)
        status = "running" if p.poll() is None else f"exited {p.returncode}"
        return status, new, timed_out

    def kill_all(self) -> None:
        for p in self._procs.values():
            kill_tree(p)


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
