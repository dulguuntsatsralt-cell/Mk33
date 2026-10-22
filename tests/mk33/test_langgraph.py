"""LedgerMiddleware inside a real ``create_agent`` graph, with a scripted chat model."""

import pytest

pytest.importorskip("langchain")
pytest.importorskip("langgraph")

from langchain.agents import create_agent  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from langchain_core.tools import tool  # noqa: E402

from mk33.integrations.langgraph import (LEDGER_HEADER, RECALL_HEADER, LedgerMiddleware, compact,  # noqa: E402
                                         ledger_line, mk33_web_tools, split_steps)


class ScriptedChat(BaseChatModel):
    """Calls `lookup` n times, then answers. Records every input it receives."""

    n_calls: int = 12
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        done = sum(isinstance(m, ToolMessage) for m in messages if not getattr(m, "name", "") == "mk33_ledger")
        step = len([m for m in self.seen[-1] if isinstance(m, AIMessage)])
        if len(self.seen) <= self.n_calls:
            k = len(self.seen)
            msg = AIMessage(content=f"checking item {k}",
                            tool_calls=[{"id": f"call{k}", "name": "lookup", "args": {"item": k}}])
        else:
            msg = AIMessage(content=f"final answer after {done} visible results, {step} visible steps")
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def lookup(item: int) -> str:
    """Look up an item."""
    if item == 3:
        raise ValueError("database offline for item 3")
    return f"item {item}: value {item * 7}\n" + ("detail " * 80)


def run(middleware, n_calls=12):
    model = ScriptedChat(n_calls=n_calls, seen=[])
    from langchain.agents.middleware import ToolErrorMiddleware
    agent = create_agent(model, tools=[lookup], system_prompt="You are a test agent.",
                         middleware=[ToolErrorMiddleware(on_error=lambda e, req: f"Error: {e}"), *middleware])
    state = agent.invoke({"messages": [HumanMessage("Find the value of every item.")]})
    return model, state


def test_without_middleware_model_sees_everything():
    model, state = run([])
    assert len(model.seen[-1]) == len(state["messages"]) - 1 + 1  # all history (+ system prompt, - final answer)


def test_middleware_compacts_model_input_but_keeps_full_state():
    mw = LedgerMiddleware(window=1200, theta=0.75, keep_steps=3, ledger_cap=400)
    model, state = run([mw])
    assert mw.events, "history should have outgrown the window"
    last = model.seen[-1]
    # Full history stays in the graph state: 12 calls and their results, plus task and answer.
    assert sum(isinstance(m, ToolMessage) for m in state["messages"]) == 12
    # The model saw: system prompt, the task, the ledger, then only the newest steps verbatim.
    assert isinstance(last[0], SystemMessage) and last[1].content == "Find the value of every item."
    ledger = last[2]
    assert ledger.content.startswith(LEDGER_HEADER)
    assert sum(isinstance(m, ToolMessage) for m in last) <= 3
    # Every tool result the model sees is preceded by the AI call it answers.
    ids = set()
    for m in last:
        if isinstance(m, AIMessage):
            ids |= {tc["id"] for tc in m.tool_calls}
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in ids
    # The input stays within the window.
    assert mw.events[-1].kept_tokens <= 1200


def test_failed_steps_stay_on_the_ledger():
    mw = LedgerMiddleware(window=1200, keep_steps=2, ledger_cap=800)
    model, _ = run([mw])
    ledger = next(m for m in model.seen[-1] if getattr(m, "name", "") == "mk33_ledger").content
    assert "FAILED" in ledger and "item 3" in ledger
    assert 'lookup({"item": 1})' in ledger


def test_ledger_cap_drops_oldest_lines_first():
    mw = LedgerMiddleware(window=1200, keep_steps=2, ledger_cap=60)
    model, _ = run([mw])
    ledger = next(m for m in model.seen[-1] if getattr(m, "name", "") == "mk33_ledger").content
    assert "oldest steps not shown" in ledger
    assert "step 1:" not in ledger


def test_retrieval_recalls_relevant_old_steps():
    msgs = [HumanMessage("task")]
    for k in range(1, 9):
        msgs.append(AIMessage(content=f"step {k}", tool_calls=[{"id": f"c{k}", "name": "lookup", "args": {"item": k}}]))
        body = "zebra migration notes " if k == 2 else "routine output "
        msgs.append(ToolMessage(content=body * 60, tool_call_id=f"c{k}"))
    msgs.append(HumanMessage("what did we learn about the zebra migration?"))
    out, info = compact(msgs, window=1500, keep_steps=2, ledger_cap=300, retrieve_k=1, retrieve_budget=500)
    assert info is not None and info.recalled == [1]  # step 2 (index 1)
    recall = next(m for m in out if getattr(m, "name", "") == "mk33_recall")
    assert recall.content.startswith(RECALL_HEADER) and "zebra" in recall.content


def test_split_steps_and_ledger_line():
    msgs = [SystemMessage("s"), HumanMessage("task"),
            AIMessage(content="thinking", tool_calls=[{"id": "a", "name": "lookup", "args": {"item": 1}}]),
            ToolMessage(content="item 1: value 7", tool_call_id="a"),
            AIMessage(content="plan: try the cache\nmore")]
    head, steps = split_steps(msgs)
    assert [m.content for m in head] == ["s", "task"]
    assert ledger_line(steps[0]) == 'step 1: lookup({"item": 1}) -> item 1: value 7'
    assert ledger_line(steps[1]) == "step 2: note: plan: try the cache"


def test_short_history_passes_through_unchanged():
    msgs = [HumanMessage("hi"), AIMessage("hello")]
    out, info = compact(msgs, window=8000)
    assert info is None and out == msgs


def test_web_tools_are_langchain_tools(monkeypatch):
    from mk33 import web

    monkeypatch.setitem(web._SEARCHERS, "duckduckgo", lambda q, k: [web.SearchResult("T", "https://x.example", "s")])
    search, fetch = mk33_web_tools("duckduckgo")
    assert (search.name, fetch.name) == ("web_search", "web_fetch")
    assert "https://x.example" in search.invoke({"query": "mongolia"})
    assert "non-public address" in fetch.invoke({"url": "http://localhost:1/"})
