"""Generation with a bounded KV cache.

The prompt is prefilled in chunks. After each chunk, and every ``decode_slack`` new
tokens while decoding, the cache is compressed back to the policy's budget. Peak cache
size is therefore about ``budget + chunk`` tokens per layer, whatever the prompt length.

Positions stay correct after eviction: cached keys already carry their rotary
embedding, and new tokens get their true positions through ``position_ids``, while
``cache_position`` indexes the (shorter) compressed cache.

With ``KVPolicy("full")`` and a single chunk this is ordinary generation, which the
tests use as the reference.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from .policies import KVPolicy


def _layers(cache) -> int:
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def get_kv(cache, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    if hasattr(cache, "layers"):  # transformers >= 4.56
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def set_kv(cache, i: int, k: torch.Tensor, v: torch.Tensor) -> None:
    if hasattr(cache, "layers"):
        cache.layers[i].keys, cache.layers[i].values = k, v
    else:
        cache.key_cache[i], cache.value_cache[i] = k, v


def cache_bytes(cache) -> int:
    total = 0
    for i in range(_layers(cache)):
        k, v = get_kv(cache, i)
        total += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total


@dataclass
class KVStats:
    policy: str = "full"
    prompt_tokens: int = 0
    new_tokens: int = 0
    peak_cache_tokens: int = 0
    peak_cache_bytes: int = 0
    evicted_tokens: int = 0
    compressions: int = 0
    compress_s: float = 0.0
    prefill_s: float = 0.0
    decode_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class CompressedGenerator:
    def __init__(self, model, policy: KVPolicy | None = None, chunk: int = 512, decode_slack: int = 64):
        self.model = model
        self.policy = policy or KVPolicy("full")
        self.chunk = chunk
        self.decode_slack = decode_slack
        self.device = next(model.parameters()).device
        self._gen = torch.Generator(device="cpu")
        attn = getattr(getattr(model, "config", None), "_attn_implementation", "eager")
        if self.policy.needs_attention and attn != "eager":
            raise RuntimeError(f"{self.policy.name} needs attention probabilities; load the model with "
                               f"attn_implementation='eager' (got {attn!r})")

    # ------------------------------------------------------------------ cache
    def _new_cache(self):
        from transformers import DynamicCache

        try:
            return DynamicCache(config=self.model.config)
        except TypeError:  # older transformers
            return DynamicCache()

    def _accumulate(self, acc: list[torch.Tensor] | None, attentions, cached: int) -> list[torch.Tensor] | None:
        if not self.policy.needs_attention:
            return None
        if not attentions or any(a is None for a in attentions):
            raise RuntimeError("h2o needs attention probabilities; load the model with attn_implementation='eager'")
        out = []
        for i, att in enumerate(attentions):  # [B, Hq, q, kv]
            b, hq, _, kv = att.shape
            hkv = get_kv(self._cache, i)[0].shape[1]
            a = att.float().sum(2).view(b, hkv, hq // hkv, kv).sum(2)
            if acc is not None:
                prev = acc[i]
                a[..., : prev.shape[-1]] += prev
            out.append(a)
        return out

    def _compress(self, acc, stats: KVStats, target: int):
        cache, pol = self._cache, self.policy
        length = cache.get_seq_length()
        if not pol.compresses or length <= target:
            return acc
        t0 = time.perf_counter()
        for i in range(_layers(cache)):
            k, v = get_kv(cache, i)
            b, h = k.shape[:2]
            scores = pol.scores(k, acc[i] if acc else None, self._gen)
            idx = pol.select(scores, b, h, length, k.device)
            k2, v2 = _gather(k, v, idx, pol.backend)
            set_kv(cache, i, k2, v2)
            if acc:
                acc[i] = torch.gather(acc[i], 2, idx)
        stats.evicted_tokens += length - pol.budget
        stats.compressions += 1
        stats.compress_s += time.perf_counter() - t0
        return acc

    def _forward(self, ids: torch.Tensor, pos: int, stats: KVStats):
        cached = self._cache.get_seq_length()
        n = ids.shape[1]
        out = self.model(
            input_ids=ids,
            past_key_values=self._cache,
            position_ids=torch.arange(pos, pos + n, device=self.device).unsqueeze(0),
            cache_position=torch.arange(cached, cached + n, device=self.device),
            attention_mask=torch.ones(1, cached + n, dtype=torch.long, device=self.device),
            use_cache=True,
            output_attentions=self.policy.needs_attention,
        )
        self._cache = out.past_key_values
        stats.peak_cache_tokens = max(stats.peak_cache_tokens, self._cache.get_seq_length())
        stats.peak_cache_bytes = max(stats.peak_cache_bytes, cache_bytes(self._cache))
        return out

    # ------------------------------------------------------------------ public
    @torch.no_grad()
    def prefill(self, ids: torch.Tensor, stats: KVStats | None = None) -> tuple[torch.Tensor, KVStats]:
        """Run the prompt; return the logits for the last position."""
        stats = stats or KVStats(policy=self.policy.label)
        ids = ids.to(self.device)
        stats.prompt_tokens = ids.shape[1]
        self._cache, acc, self._pos = self._new_cache(), None, 0
        t0 = time.perf_counter()
        logits = None
        for start in range(0, ids.shape[1], self.chunk):
            piece = ids[:, start:start + self.chunk]
            out = self._forward(piece, self._pos, stats)
            acc = self._accumulate(acc, getattr(out, "attentions", None), self._pos)
            self._pos += piece.shape[1]
            logits = out.logits[:, -1]
            acc = self._compress(acc, stats, self.policy.budget)
        self._acc = acc
        stats.prefill_s = time.perf_counter() - t0
        return logits, stats

    @torch.no_grad()
    def generate(self, ids: torch.Tensor, max_new_tokens: int = 512, eos_ids: set[int] | None = None,
                 temperature: float = 0.0, top_p: float = 1.0, seed: int = 0) -> tuple[list[int], KVStats]:
        self._gen.manual_seed(seed)
        logits, stats = self.prefill(ids)
        acc, new = self._acc, []
        eos_ids = eos_ids or set()
        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            tok = _sample(logits[0], temperature, top_p, self._gen)
            new.append(tok)
            if tok in eos_ids:
                break
            out = self._forward(torch.tensor([[tok]], device=self.device), self._pos, stats)
            acc = self._accumulate(acc, getattr(out, "attentions", None), self._pos)
            self._pos += 1
            logits = out.logits[:, -1]
            if self._cache.get_seq_length() >= self.policy.budget + self.decode_slack:
                acc = self._compress(acc, stats, self.policy.budget)
        stats.new_tokens = len(new)
        stats.decode_s = time.perf_counter() - t0
        return new, stats


    @torch.no_grad()
    def continuation_nll(self, ids: torch.Tensor, context: int) -> tuple[float, KVStats]:
        """Mean negative log-likelihood of ``ids[context:]`` given everything before it.

        The context is prefilled under the policy, then the continuation is scored in
        chunks with the cache compressed between chunks, exactly as during generation.
        With ``KVPolicy("full")`` this equals the model's ordinary loss on the continuation.
        """
        ids = ids.to(self.device)
        logits, stats = self.prefill(ids[:, :context])
        acc, total, count = self._acc, 0.0, 0
        rest = ids[:, context:]
        for start in range(0, rest.shape[1], self.chunk):
            piece = rest[:, start:start + self.chunk]
            out = self._forward(piece, self._pos, stats)
            acc = self._accumulate(acc, getattr(out, "attentions", None), self._pos)
            self._pos += piece.shape[1]
            preds = torch.cat([logits.unsqueeze(1), out.logits[:, :-1]], dim=1).float()
            total += torch.nn.functional.cross_entropy(preds[0], piece[0], reduction="sum").item()
            count += piece.shape[1]
            logits = out.logits[:, -1]
            acc = self._compress(acc, stats, self.policy.budget)
        return total / max(1, count), stats


def _gather(k, v, idx, backend):
    from .ops import gather_kv

    return gather_kv(k, v, idx, backend)


def _sample(logits: torch.Tensor, temperature: float, top_p: float, gen: torch.Generator) -> int:
    if temperature <= 0:
        return int(logits.argmax())
    probs = torch.softmax(logits.float() / temperature, dim=-1).cpu()
    if top_p < 1.0:
        sp, si = probs.sort(descending=True)
        keep = sp.cumsum(0) - sp < top_p
        probs = torch.zeros_like(probs).scatter(0, si[keep], sp[keep])
        probs /= probs.sum()
    return int(torch.multinomial(probs, 1, generator=gen))
