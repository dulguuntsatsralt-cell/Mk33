"""KV-cache compression on a tiny randomly initialized model (CPU, no downloads).

The central check is positional correctness: after eviction, the next token's logits
must equal a full forward pass in which that token simply cannot see the evicted
tokens. Everything else (budgets, policies, tool-call parsing, the chat client)
builds on that.
"""

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from mk33.kv import POLICIES, KVPolicy, select_indices  # noqa: E402
from mk33.kv.client import HFChatClient, parse_tool_calls, summarize_stats, template_messages  # noqa: E402
from mk33.kv.generate import CompressedGenerator  # noqa: E402

VOCAB = 128


def tiny_model(attn="sdpa", seed=0):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def model():
    return tiny_model()


@pytest.fixture(scope="module")
def eager_model():
    return tiny_model("eager")


def ids(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB, (1, n), generator=g)


# ---------------------------------------------------------------- selection


def test_select_keeps_sinks_recent_and_exact_budget():
    scores = torch.randn(2, 3, 100)
    idx = select_indices(scores, 2, 3, 100, budget=30, sinks=4, recent=10, device=torch.device("cpu"))
    assert idx.shape == (2, 3, 30)
    for b in range(2):
        for h in range(3):
            row = idx[b, h].tolist()
            assert row == sorted(set(row))
            assert row[:4] == [0, 1, 2, 3] and row[-10:] == list(range(90, 100))
            middle = scores[b, h, 4:90]
            assert set(row[4:-10]) == set((middle.topk(16).indices + 4).tolist())


def test_streaming_keeps_only_sinks_and_most_recent():
    idx = select_indices(None, 1, 2, 50, budget=12, sinks=2, recent=3, device=torch.device("cpu"))
    assert idx[0, 0].tolist() == [0, 1] + list(range(40, 50))


def test_policy_validation():
    assert set(POLICIES) == {"full", "random", "streaming", "keynorm", "h2o"}
    with pytest.raises(ValueError):
        KVPolicy("snap")
    with pytest.raises(ValueError):
        KVPolicy("streaming", budget=4, sinks=4)
    assert KVPolicy("h2o").needs_attention and not KVPolicy("keynorm").needs_attention
    assert KVPolicy("keynorm", budget=512).label == "keynorm@512" and KVPolicy("full").label == "full"


# ---------------------------------------------------------------- generation


def test_chunked_prefill_matches_a_single_forward(model):
    x = ids(70)
    with torch.no_grad():
        ref = model(x).logits[0, -1]
    for chunk in (7, 32, 512):
        logits, st = CompressedGenerator(model, KVPolicy("full"), chunk=chunk).prefill(x)
        assert torch.allclose(logits[0], ref, atol=1e-5)
        assert st.peak_cache_tokens == 70 and st.evicted_tokens == 0


def test_eviction_keeps_true_positions(model):
    """After compression the next token sees only kept tokens, at their original positions."""
    x = ids(41)
    g = CompressedGenerator(model, KVPolicy("streaming", budget=24, sinks=4), chunk=20)
    with torch.no_grad():
        _, st = g.prefill(x[:, :40])  # 40 cached -> sinks 0..3 + recent 20..39
        ours = g._forward(x[:, 40:], 40, st).logits[0, -1]
        mask = torch.full((41, 41), float("-inf")).triu(1)
        mask[40, 4:20] = float("-inf")
        ref = model(x, attention_mask=mask[None, None]).logits[0, -1]
    assert st.evicted_tokens == 16
    assert torch.allclose(ours, ref, atol=1e-5)


@pytest.mark.parametrize("name", ["random", "streaming", "keynorm", "h2o"])
def test_every_policy_matches_full_when_the_budget_is_not_hit(model, eager_model, name):
    m = eager_model if name == "h2o" else model
    x = ids(40)
    ref, _ = CompressedGenerator(m, KVPolicy("full"), chunk=16).generate(x, max_new_tokens=12)
    got, st = CompressedGenerator(m, KVPolicy(name, budget=200), chunk=16).generate(x, max_new_tokens=12)
    assert got == ref and st.evicted_tokens == 0


@pytest.mark.parametrize("name", ["random", "streaming", "keynorm", "h2o"])
def test_compressing_policies_bound_the_cache(model, eager_model, name):
    m = eager_model if name == "h2o" else model
    budget, chunk, slack = 48, 16, 8
    g = CompressedGenerator(m, KVPolicy(name, budget=budget, sinks=4, recent=8), chunk=chunk, decode_slack=slack)
    new, st = g.generate(ids(200), max_new_tokens=40, temperature=0.8, seed=3)
    assert len(new) == 40 and st.new_tokens == 40
    assert st.peak_cache_tokens <= budget + max(chunk, slack)
    assert g._cache.get_seq_length() <= budget + slack
    assert st.evicted_tokens > 0 and st.compressions > 0
    assert st.peak_cache_bytes > 0


def test_h2o_needs_eager_attention(model):
    with pytest.raises(RuntimeError, match="eager"):
        CompressedGenerator(model, KVPolicy("h2o", budget=32), chunk=16)


def test_h2o_scores_are_accumulated_attention_and_drive_eviction(eager_model):
    from mk33.kv.generate import KVStats, get_kv

    pol = KVPolicy("h2o", budget=40, sinks=2, recent=8)
    g = CompressedGenerator(eager_model, pol, chunk=64)
    x, stats = ids(64), KVStats()
    with torch.no_grad():
        g._cache, g._pos = g._new_cache(), 0
        out = g._forward(x, 0, stats)
        acc = g._accumulate(None, out.attentions, 0)
        att = eager_model(x, output_attentions=True).attentions[0]  # [1, 4 query heads, 64, 64]
    expected = att.sum(2).view(1, 2, 2, 64).sum(2)  # summed over queries and each GQA group
    assert torch.allclose(acc[0], expected, atol=1e-5)

    k_before = get_kv(g._cache, 0)[0].clone()
    g._compress(acc, stats, pol.budget)
    idx = pol.select(expected, 1, 2, 64, torch.device("cpu"))
    want = torch.gather(k_before, 2, idx.unsqueeze(-1).expand(*idx.shape, k_before.shape[-1]))
    assert torch.equal(get_kv(g._cache, 0)[0], want)


def test_continuation_nll(model, eager_model):
    x = ids(120)
    with torch.no_grad():
        ref = model(x, labels=x).logits[0, 79:-1].float()
    want = torch.nn.functional.cross_entropy(ref, x[0, 80:]).item()
    got, st = CompressedGenerator(model, KVPolicy("full"), chunk=16).continuation_nll(x, 80)
    assert got == pytest.approx(want, rel=1e-4) and st.evicted_tokens == 0
    for name, m in (("streaming", model), ("keynorm", model), ("h2o", eager_model)):
        nll, st = CompressedGenerator(m, KVPolicy(name, budget=48, sinks=4, recent=8), chunk=16).continuation_nll(x, 80)
        assert nll > 0 and st.evicted_tokens > 0 and st.peak_cache_tokens <= 48 + 16


def test_continuation_without_compression_keeps_budget_plus_continuation(model):
    x = ids(120)
    pol = KVPolicy("streaming", budget=40, sinks=4, recent=8)
    _, st = CompressedGenerator(model, pol, chunk=16).continuation_nll(x, 80, compress_continuation=False)
    assert st.peak_cache_tokens == 40 + 40  # budget after the context, then the 40 continuation tokens
    _, st2 = CompressedGenerator(model, pol, chunk=16).continuation_nll(x, 80)
    assert st2.peak_cache_tokens <= 40 + 16


def test_random_eviction_is_reproducible_per_seed(model):
    x = ids(120)
    run = lambda seed: CompressedGenerator(model, KVPolicy("random", budget=40, sinks=4, recent=8, seed=seed),  # noqa: E731
                                           chunk=16).continuation_nll(x, 80)[0]
    assert run(0) == run(0) and run(0) != run(1)


# ---------------------------------------------------------------- tool calls and client


def test_parse_tool_calls_formats():
    text = 'Let me look.\n<tool_call>\n{"name": "read_file", "arguments": {"path": "README.md"}}\n</tool_call>'
    content, calls = parse_tool_calls(text)
    assert content == "Let me look." and calls == [{"id": "call_0", "name": "read_file",
                                                   "arguments": {"path": "README.md"}}]
    _, calls = parse_tool_calls('<tool_call>{"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}}</tool_call>')
    assert calls[0]["arguments"] == {"command": "ls"}  # string arguments, trailing brace
    _, calls = parse_tool_calls('<think>hmm</think>{"name": "submit", "arguments": {"lesson": "x"}}')
    assert calls[0]["name"] == "submit"
    _, calls = parse_tool_calls('<tool_call>{"name": "wait_job", "arguments": {}}')  # unterminated
    assert calls[0]["name"] == "wait_job"
    assert parse_tool_calls("I would edit solution.py next.")[1] == []


def test_template_messages_turns_arguments_into_dicts():
    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]}]
    out = template_messages(msgs)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"} and out[0]["content"] == ""
    assert msgs[0]["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'  # input untouched


class ScriptTokenizer:
    """Stands in for a real tokenizer: encodes characters, decodes to scripted replies."""

    eos_token_id = None

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
        self.prompts.append((messages, tools))
        return json.dumps(messages)[-300:]

    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        return {"input_ids": torch.tensor([[ord(c) % VOCAB for c in text]])}

    def decode(self, ids, skip_special_tokens=True):
        return self.replies.pop(0)


def test_hf_client_returns_agent_format_and_records_stats(model):
    reply = '<tool_call>{"name": "submit", "arguments": {"hypothesis": "h", "lesson": "l"}}</tool_call>'
    tok = ScriptTokenizer([reply, "no tool here"])
    client = HFChatClient(model, tok, KVPolicy("keynorm", budget=64, sinks=4, recent=16), max_tokens=5, chunk=32)
    tools = [{"type": "function", "function": {"name": "submit", "parameters": {}}}]
    r = client.chat([{"role": "user", "content": "x" * 400}], tools=tools)
    assert r == {"content": "", "tool_calls": [{"id": "call_0", "name": "submit",
                                                "arguments": {"hypothesis": "h", "lesson": "l"}}]}
    assert tok.prompts[0][1] == tools
    r2 = client.chat([{"role": "user", "content": "hi"}])
    assert r2 == {"content": "no tool here", "tool_calls": []}
    s = summarize_stats(client.stats)
    assert s["kv_policy"] == "keynorm@64" and s["kv_calls"] == 2
    assert s["kv_peak_cache_tokens"] <= 64 + 32 and s["kv_evicted_share"] > 0
