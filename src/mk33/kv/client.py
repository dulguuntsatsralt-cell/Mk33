"""A local Hugging Face chat client with a compressed KV cache.

``HFChatClient`` implements the same ``chat`` interface as ``OpenAIChatClient``, so
the agent, the bench and every context mode run unchanged on a local model. It
renders messages and tool schemas with the model's own chat template and parses tool
calls in the Hermes/Qwen format (``<tool_call>{"name": ..., "arguments": ...}</tool_call>``)
that Qwen2.5, Qwen3 and most open tool-calling models use.

Every call appends a ``KVStats`` record to ``self.stats``: prompt and generated tokens,
peak cache size, evictions and timings. The bench reads them into its results.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..chat import template_messages  # noqa: F401  (re-exported; shared with training)
from .generate import CompressedGenerator, KVStats
from .policies import KVPolicy

_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Models sometimes add a trailing brace or text after the object.
        depth, end = 0, None
        for i, ch in enumerate(text):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0 and ch == "}":
                end = i + 1
                break
        if end is None:
            raise
        return json.loads(text[:end])


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Split a model reply into visible content and OpenAI-style tool calls."""
    calls: list[dict[str, Any]] = []
    for m in _TOOL_CALL.finditer(text):
        try:
            obj = _loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"_raw": args}
        calls.append({"id": f"call_{len(calls)}", "name": str(obj["name"]), "arguments": args or {}})
    content = _TOOL_CALL.sub("", text)
    if not calls:  # a bare JSON object is also accepted as a call
        stripped = _THINK.sub("", text).strip()
        if stripped.startswith("{"):
            try:
                obj = _loads(stripped)
                if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                    args = obj["arguments"]
                    calls.append({"id": "call_0", "name": str(obj["name"]),
                                  "arguments": args if isinstance(args, dict) else {"_raw": args}})
                    content = ""
            except (json.JSONDecodeError, ValueError):
                pass
    return content.strip(), calls


class HFChatClient:
    def __init__(self, model: str | Any, tokenizer: Any = None, policy: KVPolicy | None = None,
                 temperature: float = 0.7, top_p: float = 0.95, max_tokens: int = 2048, chunk: int = 512,
                 device: str | None = None, dtype: str = "auto", seed: int = 0):
        import torch

        self.policy = policy or KVPolicy("full")
        if isinstance(model, str):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            torch_dtype = (torch.bfloat16 if device == "cuda" else torch.float32) if dtype == "auto" \
                else getattr(torch, dtype)
            self.model_name = model
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
            model = AutoModelForCausalLM.from_pretrained(
                model, dtype=torch_dtype,
                attn_implementation="eager" if self.policy.needs_attention else "sdpa",
            ).to(device).eval()
        else:
            self.model_name = getattr(getattr(model, "config", None), "_name_or_path", "local")
        self.model, self.tokenizer = model, tokenizer
        self.gen = CompressedGenerator(model, self.policy, chunk=chunk)
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens
        self.seed = seed
        self.calls = 0
        self.stats: list[KVStats] = []
        eos = getattr(tokenizer, "eos_token_id", None)
        gen_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        self.eos_ids = {e for e in ([eos] + (gen_eos if isinstance(gen_eos, list) else [gen_eos])) if e is not None}

    def encode(self, messages: list[dict[str, Any]], tools: list[dict] | None = None):
        import torch

        prompt = self.tokenizer.apply_chat_template(
            template_messages(messages), tools=tools or None, add_generation_prompt=True, tokenize=False)
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids if isinstance(ids, torch.Tensor) else torch.tensor(ids)

    def count_tokens(self, text: str) -> int:
        """Exact token count with the model's tokenizer (used for the agent's text window)."""
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def chat(self, messages, tools=None, **kw):
        ids = self.encode(messages, tools)
        new, stats = self.gen.generate(
            ids, max_new_tokens=kw.get("max_tokens", self.max_tokens), eos_ids=self.eos_ids,
            temperature=kw.get("temperature", self.temperature), top_p=self.top_p,
            seed=self.seed + self.calls)
        self.calls += 1
        self.stats.append(stats)
        text = self.tokenizer.decode(new, skip_special_tokens=True)
        content, calls = parse_tool_calls(text)
        return {"content": content, "tool_calls": calls}


def summarize_stats(stats: list[KVStats]) -> dict[str, float]:
    """Per-run numbers the bench records next to task metrics."""
    if not stats:
        return {}
    n = len(stats)
    return {
        "kv_policy": stats[0].policy,
        "kv_calls": n,
        "kv_peak_cache_tokens": max(s.peak_cache_tokens for s in stats),
        "kv_peak_cache_mb": max(s.peak_cache_bytes for s in stats) / 2**20,
        "kv_mean_prompt_tokens": sum(s.prompt_tokens for s in stats) / n,
        "kv_evicted_share": sum(s.evicted_tokens for s in stats) / max(1, sum(s.prompt_tokens + s.new_tokens
                                                                              for s in stats)),
        "kv_mean_prefill_s": sum(s.prefill_s for s in stats) / n,
        "kv_mean_decode_tok_s": sum(s.new_tokens for s in stats) / max(1e-9, sum(s.decode_s for s in stats)),
        "kv_compress_share": sum(s.compress_s for s in stats) / max(1e-9, sum(s.prefill_s + s.decode_s
                                                                              for s in stats)),
    }


class HFClientFactory:
    """Builds one ``HFChatClient`` per bench run, sharing a single loaded model.

    The bench calls it with the run's mode: ``kv:<policy>`` modes get that policy,
    every other mode gets the full cache, so all modes in a grid use the same model
    and decoding settings. H2O switches the shared model to eager attention for its
    runs. Runs must be sequential (``workers=1``), which the CLI enforces.
    """

    takes_mode = True

    def __init__(self, model: str, budget: int, sinks: int = 4, recent: int = 256, chunk: int = 512,
                 temperature: float = 0.7, max_tokens: int = 2048, device: str | None = None,
                 dtype: str = "auto", seed: int = 0):
        self.model_name, self.budget, self.sinks, self.recent = model, budget, sinks, recent
        self.chunk, self.temperature, self.max_tokens = chunk, temperature, max_tokens
        self.device, self.dtype, self.seed = device, dtype, seed
        self._model = self._tok = None

    def _load(self):
        if self._model is None:
            client = HFChatClient(self.model_name, device=self.device, dtype=self.dtype)
            self._model, self._tok = client.model, client.tokenizer
        return self._model, self._tok

    def __call__(self, mode: str) -> HFChatClient:
        from ..bench.run import parse_mode

        _, name = parse_mode(mode)
        policy = KVPolicy(name or "full", budget=self.budget, sinks=self.sinks, recent=self.recent)
        model, tok = self._load()
        set_attention(model, "eager" if policy.needs_attention else "sdpa")
        self.seed += 1  # different sampling per run, reproducible for a fixed job order
        return HFChatClient(model, tok, policy, temperature=self.temperature, max_tokens=self.max_tokens,
                            chunk=self.chunk, seed=self.seed * 100_000)


def set_attention(model, impl: str) -> None:
    if getattr(model.config, "_attn_implementation", None) == impl:
        return
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(impl)
    else:
        model.config._attn_implementation = impl
