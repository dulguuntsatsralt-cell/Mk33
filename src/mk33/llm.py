"""Minimal LLM client interface (OpenAI-compatible chat completions)."""

from __future__ import annotations

import json
import os
import time
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
        self.call_log: list[dict[str, Any]] = []  # one entry per call: time and token counts

    def chat(self, messages, tools=None, **kw):
        t0 = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, tools=tools or None,
            temperature=kw.get("temperature", self.temperature),
            max_tokens=kw.get("max_tokens", self.max_tokens),
        )
        usage = getattr(resp, "usage", None)
        self.call_log.append({
            "s": round(time.perf_counter() - t0, 3),
            "prompt_chars": prompt_chars(messages),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
        })
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


def prompt_chars(messages: list[dict[str, Any]]) -> int:
    """Characters of message text sent; a server-independent measure of prompt length."""
    n = 0
    for m in messages:
        n += len(m.get("content") or "")
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", tc)
            a = fn.get("arguments", "")
            n += len(fn.get("name", "")) + len(a if isinstance(a, str) else json.dumps(a))
    return n


def to_ollama_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI-style messages to Ollama's native format.

    Ollama takes tool-call arguments as an object rather than a JSON string, and has no
    tool-call ids. The text of every message is unchanged, so the prompt the model reads,
    and therefore what the prompt cache can reuse, is the same as through /v1.
    """
    out = []
    for m in messages:
        o: dict[str, Any] = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            calls = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        args = {"_raw": args}
                calls.append({"function": {"name": fn["name"], "arguments": args}})
            o["tool_calls"] = calls
        out.append(o)
    return out


class OllamaChatClient:
    """Ollama's native /api/chat, which reports where the time of each call went.

    The OpenAI-compatible /v1 endpoint returns token counts but no timings. /api/chat also
    returns ``prompt_eval_duration`` (time spent reading the prompt) and ``eval_duration``
    (time spent writing the reply). When Ollama reuses the cached start of the previous
    prompt, ``prompt_eval_duration`` drops while ``prompt_eval_count`` still counts the whole
    prompt, so the two together measure cache reuse directly.
    """

    def __init__(self, model: str, base_url: str | None = None, temperature: float = 0.7,
                 max_tokens: int = 8192, timeout: float = 600.0, num_ctx: int | None = None):
        import httpx  # installed with openai

        url = (base_url or "http://localhost:11434").rstrip("/")
        self.url = (url[:-3] if url.endswith("/v1") else url) + "/api/chat"
        self.http = httpx.Client(timeout=timeout)
        self.model, self.temperature, self.max_tokens, self.num_ctx = model, temperature, max_tokens, num_ctx
        self.text_tool_calls = 0
        self.call_log: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, **kw):
        options = {"temperature": kw.get("temperature", self.temperature),
                   "num_predict": kw.get("max_tokens", self.max_tokens)}
        if self.num_ctx:
            options["num_ctx"] = self.num_ctx
        body = {"model": self.model, "messages": to_ollama_messages(messages), "stream": False, "options": options}
        if tools:
            body["tools"] = tools
        t0 = time.perf_counter()
        r = self.http.post(self.url, json=body)
        r.raise_for_status()
        d = r.json()
        ns = 1e-9
        self.call_log.append({
            "s": round(time.perf_counter() - t0, 3),
            "prompt_chars": prompt_chars(messages),
            "prompt_tokens": d.get("prompt_eval_count"),
            "completion_tokens": d.get("eval_count"),
            "prompt_s": round(d.get("prompt_eval_duration", 0) * ns, 3),
            "completion_s": round(d.get("eval_duration", 0) * ns, 3),
            "load_s": round(d.get("load_duration", 0) * ns, 3),
        })
        msg = d.get("message") or {}
        calls = []
        for i, tc in enumerate(msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            calls.append({"id": f"call_{len(self.call_log)}_{i}", "name": fn.get("name", ""), "arguments": args})
        content = msg.get("content") or ""
        if not calls and tools:
            from .toolcalls import calls_from_text, tool_names

            calls = calls_from_text(content, tool_names(tools))
            self.text_tool_calls += len(calls)
        return {"content": content, "tool_calls": calls}


def summarize_calls(log: list[dict[str, Any]]) -> dict[str, Any]:
    """Medians over a run's calls (fields a backend does not report are left out)."""
    import statistics

    out: dict[str, Any] = {}
    for key in ("s", "prompt_tokens", "completion_tokens", "prompt_s", "completion_s"):
        vals = [c[key] for c in log if c.get(key) is not None]
        if vals:
            out[f"call_{key}_median"] = round(statistics.median(vals), 3)
    # Prompt tokens read per second of prompt time. Far above the model's cold reading speed
    # means most of each prompt came from the cache.
    pairs = [(c["prompt_tokens"], c["prompt_s"]) for c in log if c.get("prompt_s") and c.get("prompt_tokens")]
    if pairs:
        out["prompt_tok_per_s_median"] = round(statistics.median(t / s for t, s in pairs), 1)
    return out
