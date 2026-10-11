"""Benchmark decontamination (paper Section 3.1).

An environment is dropped if, against any evaluation task, it shares
  1. a dataset (URL, normalized name, or file hash),
  2. a competition / problem ID, or
  3. >= ``ngram_threshold`` 13-gram overlap between task statements
     (the n-gram check of Brown et al., 2020).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .spec import EnvSpec

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def ngrams(tokens: list[str], n: int = 13) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def normalize_url(url: str) -> str:
    p = urlparse(url.strip().lower())
    host = p.netloc.removeprefix("www.")
    path = p.path.rstrip("/")
    path = re.sub(r"\.git$", "", path)
    return f"{host}{path}"


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


@dataclass
class BenchmarkTask:
    task_id: str
    benchmark: str
    statement: str = ""
    dataset_urls: list[str] = field(default_factory=list)
    dataset_names: list[str] = field(default_factory=list)
    dataset_hashes: list[str] = field(default_factory=list)
    problem_ids: list[str] = field(default_factory=list)


@dataclass
class ContaminationHit:
    rule: str  # "dataset_url" | "dataset_name" | "dataset_hash" | "problem_id" | "ngram"
    benchmark: str
    task_id: str
    detail: str


class BenchmarkIndex:
    def __init__(self, tasks: list[BenchmarkTask], n: int = 13, ngram_threshold: float = 0.5):
        self.n = n
        self.threshold = ngram_threshold
        self.tasks = {t.task_id: t for t in tasks}
        self._url: dict[str, list[str]] = defaultdict(list)
        self._name: dict[str, list[str]] = defaultdict(list)
        self._hash: dict[str, list[str]] = defaultdict(list)
        self._pid: dict[str, list[str]] = defaultdict(list)
        self._gram: dict[tuple[str, ...], set[str]] = defaultdict(set)
        for t in tasks:
            for u in t.dataset_urls:
                self._url[normalize_url(u)].append(t.task_id)
            for nm in t.dataset_names:
                self._name[normalize_name(nm)].append(t.task_id)
            for h in t.dataset_hashes:
                self._hash[h.lower()].append(t.task_id)
            for p in t.problem_ids:
                self._pid[normalize_name(p)].append(t.task_id)
            for g in ngrams(tokenize(t.statement), n):
                self._gram[g].add(t.task_id)

    @classmethod
    def from_jsonl(cls, path: str | Path, **kw) -> "BenchmarkIndex":
        tasks = []
        for line in Path(path).read_text().splitlines():
            if line.strip():
                tasks.append(BenchmarkTask(**json.loads(line)))
        return cls(tasks, **kw)

    def check(self, env: EnvSpec) -> list[ContaminationHit]:
        hits: list[ContaminationHit] = []

        def add(rule, ids, detail):
            for tid in ids:
                hits.append(ContaminationHit(rule, self.tasks[tid].benchmark, tid, detail))

        for u in env.dataset_urls:
            add("dataset_url", self._url.get(normalize_url(u), []), u)
        for nm in env.metadata.get("dataset_names", []):
            add("dataset_name", self._name.get(normalize_name(nm), []), nm)
        for h in env.dataset_hashes:
            add("dataset_hash", self._hash.get(h.lower(), []), h)
        for p in env.problem_ids:
            add("problem_id", self._pid.get(normalize_name(p), []), p)

        grams = ngrams(tokenize(env.task_statement), self.n)
        if grams:
            counts: dict[str, int] = defaultdict(int)
            for g in grams:
                for tid in self._gram.get(g, ()):
                    counts[tid] += 1
            for tid, c in counts.items():
                ratio = c / len(grams)
                if ratio >= self.threshold:
                    add("ngram", [tid], f"{self.n}-gram overlap {ratio:.2f}")
        return hits


def decontaminate(envs: list[EnvSpec], index: BenchmarkIndex):
    """Split envs into (kept, dropped) and count drops per rule."""
    kept, dropped = [], []
    per_rule: dict[str, int] = defaultdict(int)
    for e in envs:
        hits = index.check(e)
        if hits:
            dropped.append((e, hits))
            for r in {h.rule for h in hits}:
                per_rule[r] += 1
        else:
            kept.append(e)
    return kept, dropped, dict(per_rule)
