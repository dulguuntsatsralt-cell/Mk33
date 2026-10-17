"""Retrieval memory: retrievers, budgets, the two retrieval context modes, and
train/test consistency when retrieval refreshes the prefix."""

import pytest

from mk33.envs import IsolatedScorer
from mk33.palm import PALMConfig, compute_weights
from mk33.scaffold import Agent, AgentConfig, Budget, RLCConfig, ToolExecutor
from mk33.scaffold.ledger import LedgerEntry, RoundLedger
from mk33.scaffold.memory import (BM25Retriever, EmbeddingRetriever, MemoryItem, RetrievalMemory,
                                  render_turn, tokenize)
from mk33.scaffold.rlc import RLCContext, TurnMessages
from mk33.train import build_windows
from test_end_to_end import STEPS, JobAwareClient

TOPICS = ["learning_rate warmup schedule", "xgboost max_depth trees", "feature scaling standardize",
          "knn neighbors distance", "ridge alpha regularization", "seasonal naive drift window"]


def item(i, text, r=1):
    return MemoryItem(i, r, text, len(text) // 4)


def test_tokenize_keeps_and_splits_identifiers():
    toks = tokenize("Set max_depth=6 for the XGBoost model")
    assert "max_depth" in toks and "max" in toks and "depth" in toks and "6" in toks
    assert "the" not in toks


def test_bm25_ranks_the_matching_turn_first():
    r = BM25Retriever()
    r.add([item(i, f"tried {t}; score went down") for i, t in enumerate(TOPICS)])
    hits = r.search("what max_depth did the xgboost run use", 3)
    assert hits[0][0].turn_index == 1
    assert r.search("completely unrelated words", 3) == []


def test_bm25_breaks_ties_toward_recent_turns():
    r = BM25Retriever()
    r.add([item(0, "ridge alpha"), item(5, "ridge alpha")])
    assert [h[0].turn_index for h in r.search("ridge", 2)] == [5, 0]


def test_embedding_retriever_with_any_embed_function():
    def embed(texts):  # bag of topic words, enough to test the plumbing
        vocab = sorted({w for t in TOPICS for w in tokenize(t)})
        return [[float(w in tokenize(t)) for w in vocab] for t in texts]

    r = EmbeddingRetriever(embed)
    r.add([item(i, t) for i, t in enumerate(TOPICS)])
    assert r.search("knn distance", 1)[0][0].turn_index == 3


def test_memory_respects_budget_k_and_exclusions():
    mem = RetrievalMemory(k=2, budget_tokens=60)
    for i, t in enumerate(TOPICS):
        msgs = [{"role": "assistant", "content": f"try {t}"}, {"role": "tool", "content": "x" * 120}]
        assert mem.add_turn(i, i + 1, msgs)
    assert not mem.add_turn(0, 1, [])  # stored once
    got = mem.retrieve("ridge alpha and knn neighbors", exclude={3})
    assert [it.turn_index for it in got] == [4]  # 3 excluded; a second item would exceed 60 tokens
    assert sum(it.tokens for it in got) <= 60
    assert "[Retrieved memory]" in RetrievalMemory.render(got)


def test_render_turn_keeps_call_and_trims_long_output():
    msgs = [{"role": "assistant", "content": "check the score",
             "tool_calls": [{"function": {"name": "bash", "arguments": '{"command": "python eval.py"}'}}]},
            {"role": "tool", "content": "start " + "." * 5000 + " RMSE 2.61"}]
    text = render_turn(7, 3, msgs, max_chars=400)
    assert text.startswith("[round 3, turn 7] bash") and "python eval.py" in text
    assert "RMSE 2.61" in text and "[... trimmed ...]" in text and len(text) <= 420


def _turns(n_rounds, per_round=3, size=600):
    out, i = [], 0
    for r in range(1, n_rounds + 1):
        topic = TOPICS[(r - 1) % len(TOPICS)]
        for _ in range(per_round):
            out.append(TurnMessages(i, r, [{"role": "assistant", "content": f"round {r}: {topic} " + "x" * size},
                                          {"role": "tool", "content": f"result for {topic} " + "y" * size}]))
            i += 1
    return out


def _ledger(n):
    led = RoundLedger()
    for r in range(1, n + 1):
        led.add(LedgerEntry(round=r, hypothesis=TOPICS[r - 1], score=r / 10, normalized=r / 10, lesson="ok"))
    return led


PINNED = lambda: [{"role": "system", "content": "sys"}, {"role": "user", "content": "# Task\nfit a model"}]


def _text(msgs):
    return "\n".join(str(m["content"]) for m in msgs)


def test_retrieve_mode_puts_dropped_turns_back_by_relevance():
    cfg = RLCConfig(window=4000, theta=0.75, mode="retrieve", retrieve_k=2, retrieve_budget=700,
                    retrieve_item_chars=600)
    ctx = RLCContext(cfg)
    turns = _turns(5)
    # The current round revisits round 2's topic, so retrieval should bring round 2 back.
    turns.append(TurnMessages(len(turns), 6, [{"role": "assistant", "content": "back to xgboost max_depth"},
                                              {"role": "tool", "content": "ok"}]))
    msgs, ev = ctx.build(PINNED, _ledger(5), turns, 6)
    assert ev is not None and ev.kind == "retrieve"
    assert "Round ledger" not in _text(msgs)  # no ledger in plain retrieve mode
    assert "[Retrieved memory]" in _text(msgs)
    retrieved_rounds = {t.round for t in turns if t.turn_index in ev.retrieved}
    assert 2 in retrieved_rounds
    visible = {t.turn_index for t in ctx._visible(turns)}
    assert not visible & set(ev.retrieved)  # never duplicates what is already in context
    assert ctx.memory.stored == {t.turn_index for t in turns} - visible
    assert sum(len(m["content"]) for m in msgs) // 4 <= int(cfg.theta * cfg.window)


def test_retrieve_refreshes_each_round_and_is_append_only_within_a_round():
    cfg = RLCConfig(window=3000, theta=0.75, mode="retrieve", retrieve_budget=500)
    ctx = RLCContext(cfg)
    turns = _turns(3)
    small = [TurnMessages(9, 3, [{"role": "assistant", "content": "ok"}, {"role": "tool", "content": "ok"}])]
    _, ev = ctx.build(PINNED, None, turns, 3)
    assert ev is not None and ev.kind == "retrieve"
    a, ev_a = ctx.build(PINNED, None, turns, 3)
    assert ev_a is None
    b, ev_b = ctx.build(PINNED, None, turns + small, 3)
    assert ev_b is None and b[: len(a)] == a  # same round: context only grows
    c, ev_c = ctx.build(PINNED, None, turns + small, 4)
    assert ev_c is not None and ev_c.kind == "refresh"  # new round: retrieved block refreshed

    ctx2 = RLCContext(RLCConfig(window=3000, theta=0.75, mode="retrieve", retrieve_budget=500,
                                retrieve_refresh="compaction"))
    ctx2.build(PINNED, None, turns, 3)
    assert ctx2.build(PINNED, None, turns + small, 4)[1] is None


def test_rlc_plus_retrieve_keeps_ledger_and_memory():
    ctx = RLCContext(RLCConfig(window=4000, theta=0.75, keep_rounds=2, mode="rlc+retrieve",
                               retrieve_budget=600))
    msgs, ev = ctx.build(PINNED, _ledger(4), _turns(5), 5)
    assert ev is not None and ev.kind == "rlc+retrieve" and ev.cutoff_round == 4
    text = _text(msgs)
    assert "Round ledger" in text and "[Retrieved memory]" in text and "compacted" in text
    assert sum(len(m["content"]) for m in msgs) // 4 <= 4000


def test_unknown_mode_is_rejected_up_front():
    with pytest.raises(ValueError):
        RLCContext(RLCConfig(mode="rag"))


@pytest.mark.parametrize("mode", ["retrieve", "rlc+retrieve"])
def test_training_windows_match_agent_contexts_with_retrieval(toy_env, mode):
    client = JobAwareClient(STEPS)
    cfg = AgentConfig(budget=Budget(max_tool_calls=50),
                      rlc=RLCConfig(window=900, theta=0.75, keep_rounds=1, ledger_cap=400, mode=mode,
                                    retrieve_k=2, retrieve_budget=150, retrieve_item_chars=400))
    traj = Agent(client, ToolExecutor(toy_env, scorer=IsolatedScorer(toy_env)), toy_env, cfg).run()
    events = traj.metadata["compactions"]
    assert events and any(e["retrieved"] for e in events), "small window should retrieve something"

    weights = compute_weights(traj, PALMConfig())
    windows = build_windows(traj, weights, cfg.rlc, drop_empty=False)
    assert len(windows) == len(events) + 1
    starts = [w.first_turn for w in windows]
    for i, ctx in enumerate(client.contexts[: len(traj.turns)]):
        w = windows[max(k for k, s in enumerate(starts) if s <= i)]
        assert w.messages[: len(ctx)] == ctx, f"turn {i} context differs from its training window"
