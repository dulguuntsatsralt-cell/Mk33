"""Canonicalization helpers for PALM (paper Section 3.3)."""

from __future__ import annotations

import json
import re
from typing import Any

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"
    r"|\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"
)
_PID = re.compile(r"\b(?:pid|PID)[ =:]+\d+\b")
_ELAPSED = re.compile(r"\b\d+(?:\.\d+)?\s?(?:s|ms|sec|secs|it/s|s/it)\b")
_PROGRESS = re.compile(r"^.*(?:\d+%\|[^|]*\||\[\s*=*>?\s*\]|\r).*$")
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[A-Za-z0-9_./\-]+")


def canonical_action(tool: str, args: dict[str, Any]) -> str:
    """Tool name + arguments with whitespace normalized, keys sorted."""
    norm = {k: _WS.sub(" ", str(v)).strip() for k, v in sorted(args.items())}
    return f"{tool} {json.dumps(norm, sort_keys=True, ensure_ascii=False)}"


def normalize_observation(text: str) -> str:
    """Strip ANSI codes, timestamps, PIDs, elapsed times, and progress-bar lines."""
    text = _ANSI.sub("", text)
    lines = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if "\r" in line:
            line = line.split("\r")[-1]  # keep the final state of a carriage-return line
        if _PROGRESS.match(line):
            continue
        line = _TIMESTAMP.sub("<T>", line)
        line = _PID.sub("pid <N>", line)
        line = _ELAPSED.sub("<D>", line)
        line = line.strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = _WORD.findall(text.lower())
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)
