"""How paths the agent writes map onto the workspace.

Inside ``DockerSandbox`` the workspace is mounted at ``/workspace``, and models often
assume that layout everywhere. On a local sandbox (a laptop, Colab) the workspace is some
other folder, so ``/workspace/solution.py`` would point outside it. Both the file tools
and the process audit read ``/workspace/...`` as a path inside the workspace, so an
honest guess neither fails nor gets flagged as an attempt to leave the sandbox.
"""

from __future__ import annotations

WORKSPACE_ALIAS = "/workspace"


def agent_relative(path: str) -> str:
    """'/workspace/a/b.py' -> 'a/b.py'; '/workspace' -> '.'; other paths unchanged."""
    p = path.strip().replace("\\", "/") if path.strip().startswith(WORKSPACE_ALIAS) else path.strip()
    if p == WORKSPACE_ALIAS or p == WORKSPACE_ALIAS + "/":
        return "."
    if p.startswith(WORKSPACE_ALIAS + "/"):
        return p[len(WORKSPACE_ALIAS) + 1:]
    return path
