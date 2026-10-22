"""YAML config loading with ``inherit:`` and deep merge.

Ablation configs inherit from ``configs/mk33/default.yaml`` and override only
what they change.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .envs.audit import AuditConfig
from .envs.filter import FilterConfig
from .palm.weights import PALMConfig
from .scaffold.agent import Budget
from .scaffold.rlc import RLCConfig
from .scaffold.tools import ToolConfig
from .selection import SelectionConfig
from .train.sft import SFTConfig


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("inherit", None)
    if parent:
        data = deep_merge(load_yaml((path.parent / parent).resolve()), data)
    return data


def _build(cls, d: dict | None):
    d = dict(d or {})
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**d)


@dataclass
class Mk33Config:
    name: str = "default"
    filter: FilterConfig = field(default_factory=FilterConfig)
    decontam: dict[str, Any] = field(default_factory=lambda: {"enabled": True, "n": 13, "threshold": 0.5})
    audit: AuditConfig = field(default_factory=AuditConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    rlc: RLCConfig = field(default_factory=RLCConfig)
    rlc_train: RLCConfig = field(default_factory=RLCConfig)  # A2: may differ from inference
    budget: dict[str, Budget] = field(default_factory=dict)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    palm: PALMConfig = field(default_factory=PALMConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Mk33Config:
    d = load_yaml(path)
    budgets = {k: _build(Budget, v) for k, v in (d.get("budget") or {}).items()}
    rlc = _build(RLCConfig, d.get("rlc"))
    rlc_train = _build(RLCConfig, d["rlc_train"]) if d.get("rlc_train") else copy.deepcopy(rlc)
    cfg = Mk33Config(
        name=d.get("name", Path(path).stem),
        filter=_build(FilterConfig, d.get("filter")),
        decontam={"enabled": True, "n": 13, "threshold": 0.5, **(d.get("decontam") or {})},
        audit=_build(AuditConfig, d.get("audit")),
        tools=_build(ToolConfig, d.get("tools")),
        rlc=rlc,
        rlc_train=rlc_train,
        budget=budgets,
        selection=_build(SelectionConfig, d.get("selection")),
        palm=_build(PALMConfig, d.get("palm")),
        sft=_build(SFTConfig, d.get("sft")),
        raw=d,
    )
    for k in cfg.raw:
        if k not in {f.name for f in fields(Mk33Config)} | {"inherit", "description", "notes"}:
            raise ValueError(f"unknown top-level config key: {k}")
    assert is_dataclass(cfg)
    return cfg
