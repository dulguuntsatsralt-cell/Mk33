"""Chat-template input shared by training and local inference.

Training (``train/sft.py``) and the local model client (``kv/client.py``) both pass
messages through ``template_messages`` before ``apply_chat_template``, so a window is
rendered into exactly the same text in both places, whatever the template does with
string versus dict tool arguments.
"""

from __future__ import annotations

import json
from typing import Any


def template_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-format messages -> what HF chat templates expect (arguments as dicts)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                fn["arguments"] = args
                calls.append({**tc, "function": fn})
            m["tool_calls"] = calls
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out
