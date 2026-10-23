"""Tool calls that a model wrote as text instead of using the tool-call format.

Small models sometimes stop using native tool calls partway through a run and write the
call as JSON in their reply, often in a ```json code block. A 40-round run with
Qwen2.5-7B lost its last 35 turns this way. ``calls_from_text`` recovers such calls, but
only for tools that were offered on that request, and only from a JSON object with a
``name`` and ``arguments`` (or ``parameters``) field, so ordinary JSON in a reply is
never executed by accident. Clients count recovered calls so reports can show how often
this happened.
"""

from __future__ import annotations

import json
from typing import Any

_DECODER = json.JSONDecoder()


def tool_names(tools: list[dict] | None) -> set[str]:
    return {t.get("function", {}).get("name") for t in tools or [] if t.get("function", {}).get("name")}


def calls_from_text(text: str, allowed: set[str], limit: int = 1) -> list[dict[str, Any]]:
    """Up to ``limit`` calls written as JSON objects in ``text``, in order of appearance."""
    calls: list[dict[str, Any]] = []
    if not text or not allowed:
        return calls
    i = text.find("{")
    while i != -1 and len(calls) < limit:
        try:
            obj, end = _DECODER.raw_decode(text, i)
        except ValueError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict) and obj.get("name") in allowed and ("arguments" in obj or "parameters" in obj):
            args = obj.get("arguments", obj.get("parameters"))
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            if isinstance(args, dict):
                calls.append({"id": f"text_call_{len(calls)}", "name": obj["name"], "arguments": args})
            i = text.find("{", end)
        else:
            i = text.find("{", i + 1)
    return calls
