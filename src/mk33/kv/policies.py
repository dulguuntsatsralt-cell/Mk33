"""KV-cache eviction policies.

Every policy keeps ``budget`` tokens per layer and KV head. The first ``sinks`` tokens
(attention sinks: the system prompt's opening, which models attend to heavily) and the
last ``recent`` tokens are always kept; the policy decides which middle tokens fill
the rest. Different heads may keep different tokens.

    full       never evict (the baseline the others are measured against)
    random     random middle tokens; the control that any real policy has to beat
    streaming  sinks + the most recent tokens only (StreamingLLM, Xiao et al. 2023)
    keynorm    middle tokens with the smallest key L2 norm (Devoto et al. 2024 found
               low-norm keys tend to receive high attention); needs no attention scores
    h2o        "heavy hitters": middle tokens with the largest accumulated attention
               (Zhang et al. 2023); needs attention probabilities, so it forces eager
               attention, which is slower and uses more memory during prefill
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .names import POLICIES
from .ops import key_norm


@dataclass
class KVPolicy:
    name: str = "full"
    budget: int = 4096
    sinks: int = 4
    recent: int = 256
    seed: int = 0
    backend: str = "auto"  # "auto", "torch" or "triton" for the gather and norm kernels

    def __post_init__(self):
        if self.name not in POLICIES:
            raise ValueError(f"unknown KV policy {self.name!r}; choose from {', '.join(POLICIES)}")
        if self.name != "full" and self.budget < self.sinks + 1:
            raise ValueError("budget must exceed the number of sink tokens")

    @property
    def compresses(self) -> bool:
        return self.name != "full"

    @property
    def needs_attention(self) -> bool:
        return self.name == "h2o"

    @property
    def label(self) -> str:
        return "full" if self.name == "full" else f"{self.name}@{self.budget}"

    def scores(self, keys: torch.Tensor, attn_acc: torch.Tensor | None,
               gen: torch.Generator | None = None) -> torch.Tensor | None:
        """Higher score = keep. ``None`` means recency only (streaming)."""
        if self.name == "streaming":
            return None
        if self.name == "random":
            return torch.rand(keys.shape[:3], generator=gen, device="cpu").to(keys.device)
        if self.name == "keynorm":
            return -key_norm(keys, self.backend)
        if self.name == "h2o":
            if attn_acc is None:
                raise ValueError("h2o needs accumulated attention scores")
            return attn_acc
        raise ValueError(f"policy {self.name} does not score tokens")

    def select(self, scores: torch.Tensor | None, batch: int, heads: int, length: int,
               device: torch.device) -> torch.Tensor:
        """Indices to keep, ``[batch, heads, budget]``, ascending along the last axis."""
        return select_indices(scores, batch, heads, length, self.budget, self.sinks, self.recent, device)


def select_indices(scores: torch.Tensor | None, batch: int, heads: int, length: int, budget: int,
                   sinks: int, recent: int, device: torch.device) -> torch.Tensor:
    if length <= budget:
        return torch.arange(length, device=device).expand(batch, heads, length)
    s = min(sinks, budget)
    r = budget - s if scores is None else min(recent, budget - s)
    m = budget - s - r
    parts = [torch.arange(s, device=device).expand(batch, heads, s)]
    if m > 0:
        mid = scores[..., s:length - r]
        top = mid.topk(m, dim=-1).indices + s
        parts.append(top.sort(dim=-1).values)
    parts.append(torch.arange(length - r, length, device=device).expand(batch, heads, r))
    return torch.cat(parts, dim=-1).contiguous()
