"""Minimal LLM client interface (OpenAI-compatible chat completions)."""

from __future__ import annotations

import os
from typing import Any, Protocol


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None, **kw) -> dict:
        """Return {"content": str, "tool_calls": [{"id", "name", "arguments": dict}]}."""


class OpenAIChatClient:
    """Wraps any OpenAI-compatible endpoint (vLLM, SGLang, hosted APIs)."""

    def __init__(self, model: str, base_url: str | None = None, api_key_env: str = "MODEL_API_KEY",
                 temperature: float = 0.7, max_tokens: int = 8192, timeout: float = 600.0):
        from openai import OpenAI  # lazy: optional dependency

        self.client = OpenAI(base_url=base_url, api_key=os.environ.get(api_key_env, "EMPTY"),
                             timeout=timeout, max_retries=0)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.text_tool_calls = 0  # calls recovered from the reply text (see mk33.toolcalls)

    def chat(self, messages, tools=None, **kw):
        import json

        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools or None,
            temperature=kw.get("temperature", self.temperature),
            max_tokens=kw.get("max_tokens", self.max_tokens),
        )
        msg = resp.choices[0].message
        calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})
        if not calls and tools:
            from .toolcalls import calls_from_text, tool_names

            calls = calls_from_text(msg.content or "", tool_names(tools))
            self.text_tool_calls += len(calls)
        return {"content": msg.content or "", "tool_calls": calls}
