"""Mk33's round-ledger memory for LangChain / LangGraph agents.

``LedgerMiddleware`` plugs into ``langchain.agents.create_agent`` (which runs on
LangGraph). When an agent's history gets close to the context window, older steps are
replaced, in what the model sees, by a compact ledger: one line per step with the tool,
its arguments and the first line of the result, failures included. The newest steps
stay verbatim. Optionally, BM25 retrieval brings back the full text of the old steps
most relevant to the latest message (Mk33's ``rlc+retrieve`` mode).

Compared with LangChain's ``SummarizationMiddleware``:

* no extra model call: compaction is deterministic and costs nothing;
* the full history stays in the graph state (and any checkpointer); only the model's
  input is compacted, so nothing is lost and runs stay inspectable;
* failed steps are kept on the ledger, so the agent sees what already did not work.

    from langchain.agents import create_agent
    from mk33.integrations.langgraph import LedgerMiddleware, mk33_web_tools

    agent = create_agent(model, tools=mk33_web_tools(), middleware=[LedgerMiddleware(window=8000)])

Needs ``pip install 'mk33[langgraph]'``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..scaffold.memory import BM25Retriever, MemoryItem
from ..tokens import approx_tokens

LEDGER_HEADER = "[Mk33 step ledger] Earlier steps were compacted. One line per step, failures included:"
RECALL_HEADER = "[Mk33 recalled steps] Full text of earlier steps relevant to the latest message:"


@dataclass
class Step:
    """One agent step: an AI message and the tool results that answer its calls."""

    index: int
    messages: list[BaseMessage]


@dataclass
class CompactionInfo:
    total_tokens: int
    kept_tokens: int
    steps_total: int
    steps_compacted: int
    recalled: list[int] = field(default_factory=list)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # content blocks
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def message_tokens(m: BaseMessage, count: Callable[[str], int] = approx_tokens) -> int:
    n = count(_text(m.content)) + 4
    for tc in getattr(m, "tool_calls", None) or []:
        n += count(tc.get("name", "")) + count(json.dumps(tc.get("args", {}), ensure_ascii=False))
    return n


def split_steps(messages: Sequence[BaseMessage]) -> tuple[list[BaseMessage], list[Step]]:
    """(pinned head, steps). The head is everything before the first AI message: the
    task and any instructions. Each step starts at an AI message, so cutting between
    steps never separates a tool call from its result."""
    head: list[BaseMessage] = []
    steps: list[Step] = []
    for m in messages:
        if isinstance(m, AIMessage):
            steps.append(Step(len(steps), [m]))
        elif steps:
            steps[-1].messages.append(m)
        else:
            head.append(m)
    return head, steps


def _short(value: Any, limit: int) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
    s = " ".join(s.split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _first_line(text: str, limit: int) -> str:
    for line in text.splitlines():
        if line.strip():
            return _short(line.strip(), limit)
    return "(empty)"


def ledger_line(step: Step, arg_chars: int = 120, result_chars: int = 160) -> str:
    ai = step.messages[0]
    calls = getattr(ai, "tool_calls", None) or []
    results = {m.tool_call_id: m for m in step.messages[1:] if isinstance(m, ToolMessage)}
    if not calls:
        return f"step {step.index + 1}: note: {_first_line(_text(ai.content), result_chars)}"
    parts = []
    for tc in calls:
        res = results.get(tc.get("id"))
        if res is None:
            outcome = "no result"
        else:
            failed = getattr(res, "status", "success") == "error"
            outcome = ("FAILED: " if failed else "") + _first_line(_text(res.content), result_chars)
        parts.append(f"{tc.get('name')}({_short(tc.get('args', {}), arg_chars)}) -> {outcome}")
    return f"step {step.index + 1}: " + "; ".join(parts)


def step_text(step: Step) -> str:
    lines = []
    for m in step.messages:
        role = {AIMessage: "assistant", ToolMessage: "tool", HumanMessage: "user"}.get(type(m), m.type)
        body = _text(m.content)
        for tc in getattr(m, "tool_calls", None) or []:
            body += f"\n[call {tc.get('name')} {_short(tc.get('args', {}), 400)}]"
        lines.append(f"{role}: {body}")
    return "\n".join(lines)


def compact(messages: Sequence[BaseMessage], window: int = 8000, theta: float = 0.75, keep_steps: int = 6,
            ledger_cap: int = 2000, retrieve_k: int = 0, retrieve_budget: int = 1500,
            count: Callable[[str], int] = approx_tokens) -> tuple[list[BaseMessage], CompactionInfo | None]:
    """The messages to send to the model. Unchanged while they fit in ``theta * window``."""
    total = sum(message_tokens(m, count) for m in messages)
    if total <= theta * window:
        return list(messages), None
    system = [m for m in messages if isinstance(m, SystemMessage)]
    head, steps = split_steps([m for m in messages if not isinstance(m, SystemMessage)])
    fixed = sum(message_tokens(m, count) for m in system + head)

    def size(ss):
        return sum(message_tokens(m, count) for s in ss for m in s.messages)

    reserve = ledger_cap + (retrieve_budget if retrieve_k else 0)
    keep = min(keep_steps, len(steps))
    while keep > 1 and fixed + reserve + size(steps[-keep:]) > theta * window:
        keep -= 1  # always keep the newest step: it holds the call the model is answering
    kept, dropped = steps[len(steps) - keep:], steps[: len(steps) - keep]
    if not dropped:
        return list(messages), None

    lines = [ledger_line(s) for s in dropped]
    while len(lines) > 1 and count("\n".join(lines)) > ledger_cap:  # keep the newest ledger lines
        lines.pop(0)
    omitted = len(dropped) - len(lines)
    ledger = LEDGER_HEADER + (f"\n({omitted} oldest steps not shown)" if omitted else "") + "\n" + "\n".join(lines)
    out: list[BaseMessage] = [*system, *head, HumanMessage(content=ledger, name="mk33_ledger")]

    recalled: list[int] = []
    if retrieve_k and dropped:
        bm25 = BM25Retriever()
        bm25.add([MemoryItem(s.index, s.index, step_text(s), count(step_text(s))) for s in dropped])
        query = _text(messages[-1].content) + " " + " ".join(
            _short(tc.get("args", {}), 400) for tc in getattr(messages[-1], "tool_calls", None) or [])
        if isinstance(messages[-1], ToolMessage) and kept:  # answering a tool result: use the call too
            query += " " + step_text(kept[-1])
        chunks, used = [], 0
        for item, _ in bm25.search(query, retrieve_k):
            text = item.text if item.tokens <= retrieve_budget - used else item.text[: 4 * (retrieve_budget - used)]
            if not text.strip():
                break
            chunks.append(f"--- step {item.turn_index + 1} ---\n{text}")
            recalled.append(item.turn_index)
            used += count(text)
        if chunks:
            out.append(HumanMessage(content=RECALL_HEADER + "\n" + "\n".join(chunks), name="mk33_recall"))

    for s in kept:
        out.extend(s.messages)
    info = CompactionInfo(total, sum(message_tokens(m, count) for m in out), len(steps), len(dropped), recalled)
    return out, info


class LedgerMiddleware(AgentMiddleware):
    """Round-ledger compaction for ``create_agent``. See the module docstring.

    Args:
        window: the model's usable context in tokens.
        theta: compact once the history passes this share of the window (Mk33 default 0.75).
        keep_steps: newest steps kept verbatim (fewer if they alone do not fit).
        ledger_cap: tokens for the ledger; the oldest lines go first when it is full.
        retrieve_k: also recall the full text of this many relevant old steps (0 = off).
        retrieve_budget: tokens for recalled steps.
        count_tokens: token counter for strings (default: about 4 characters per token).
    """

    def __init__(self, window: int = 8000, theta: float = 0.75, keep_steps: int = 6, ledger_cap: int = 2000,
                 retrieve_k: int = 0, retrieve_budget: int = 1500,
                 count_tokens: Callable[[str], int] = approx_tokens):
        super().__init__()
        if not 0 < theta <= 1:
            raise ValueError("theta must be in (0, 1]")
        self.kw = dict(window=window, theta=theta, keep_steps=keep_steps, ledger_cap=ledger_cap,
                       retrieve_k=retrieve_k, retrieve_budget=retrieve_budget, count=count_tokens)
        self.events: list[CompactionInfo] = []

    def _prepare(self, request):
        msgs, info = compact(request.messages, **self.kw)
        if info is None:
            return request
        self.events.append(info)
        return request.override(messages=msgs)

    def wrap_model_call(self, request, handler):
        return handler(self._prepare(request))

    async def awrap_model_call(self, request, handler):
        return await handler(self._prepare(request))


def mk33_web_tools(provider: str = "auto", max_chars: int = 20_000) -> list:
    """Mk33's ``web_search`` and ``web_fetch`` as LangChain tools (see ``mk33.web``)."""
    from langchain_core.tools import StructuredTool

    from ..web import make_fetch, make_search

    search, fetch = make_search(provider), make_fetch(max_chars)

    def web_search(query: str) -> str:
        return search(query)

    def web_fetch(url: str) -> str:
        return fetch(url)

    return [
        StructuredTool.from_function(func=web_search, name="web_search",
                                     description="Search the web. Returns titles, URLs and snippets."),
        StructuredTool.from_function(func=web_fetch, name="web_fetch",
                                     description="Fetch a public web page (http or https) as readable text."),
    ]
