"""Token counting. The default is a fast approximation; pass a tokenizer for exact counts."""

from __future__ import annotations

from typing import Any, Callable


def approx_tokens(text: str) -> int:
    return (len(text) + 3) // 4


def hf_counter(tokenizer_path: str) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    return lambda text: len(tok.encode(text, add_special_tokens=False))


def message_tokens(msg: dict[str, Any], count: Callable[[str], int] = approx_tokens) -> int:
    n = count(str(msg.get("content") or "")) + 4
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        n += count(fn.get("name", "")) + count(str(fn.get("arguments", "")))
    return n
