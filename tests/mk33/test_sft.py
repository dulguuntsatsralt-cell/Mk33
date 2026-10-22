import pytest

from mk33.train.sft import message_token_weights


class FakeTokenizer:
    """Character-level tokenizer with a simple chat template."""

    def apply_chat_template(self, messages, tools=None, tokenize=False):
        return "".join(f"<{m['role']}>{m.get('content') or ''}</>" for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_token_weights_cover_only_assistant_spans():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": "out"}, {"role": "assistant", "content": "go"}]
    ids, tw = message_token_weights(FakeTokenizer(), msgs, [None, 0.5, None, 1.0], max_len=10_000)
    text = "".join(map(chr, ids))
    assert text == "<user>hi</><assistant>ok</><tool>out</><assistant>go</>"
    weighted = "".join(c for c, w in zip(text, tw) if w)
    assert weighted == "<assistant>ok</><assistant>go</>"
    assert set(tw) == {0.0, 0.5, 1.0}


def test_weighted_loss_matches_manual():
    torch = pytest.importorskip("torch")
    from mk33.train.sft import weighted_lm_loss

    logits = torch.randn(1, 4, 7)
    labels = torch.tensor([[1, 2, 3, 4]])
    w = torch.tensor([[0.0, 1.0, 0.0, 0.5]])
    nll = torch.nn.functional.cross_entropy(logits[0, :-1], labels[0, 1:], reduction="none")
    expected = (nll * w[0, 1:]).sum() / w[0, 1:].sum()
    assert torch.allclose(weighted_lm_loss(logits, labels, w), expected)


# ------------------------------------------------- small-GPU training path
def _tiny_qwen(seed=0):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    torch.manual_seed(seed)
    return transformers.Qwen2ForCausalLM(cfg)


def _batch(torch, n=40, seed=1):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 97, (1, n), generator=g)
    w = torch.zeros(1, n)
    w[0, 10:18] = 1.0   # a fully weighted assistant turn
    w[0, 25:31] = 0.5   # a partially weighted one
    return ids, torch.ones_like(ids), ids.clone(), w


def test_palm_loss_matches_full_logits_loss():
    torch = pytest.importorskip("torch")
    from mk33.train.sft import palm_loss, weighted_lm_loss

    model = _tiny_qwen().eval()
    ids, att, labels, w = _batch(torch)
    full = weighted_lm_loss(model(input_ids=ids, attention_mask=att).logits, labels, w)
    fast = palm_loss(model, ids, att, labels, w)
    assert torch.allclose(full, fast, atol=1e-5)


def test_palm_loss_gradients_match_full_logits_loss():
    torch = pytest.importorskip("torch")
    from mk33.train.sft import palm_loss, weighted_lm_loss

    ids, att, labels, w = _batch(torch)
    m1, m2 = _tiny_qwen(), _tiny_qwen()
    weighted_lm_loss(m1(input_ids=ids, attention_mask=att).logits, labels, w).backward()
    palm_loss(m2, ids, att, labels, w).backward()
    for (n, p1), (_, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-5), n


def test_palm_loss_without_weighted_tokens_is_zero_and_differentiable():
    torch = pytest.importorskip("torch")
    from mk33.train.sft import palm_loss

    model = _tiny_qwen()
    ids, att, labels, _ = _batch(torch)
    loss = palm_loss(model, ids, att, labels, torch.zeros(1, ids.size(1)))
    assert loss.item() == 0.0
    loss.backward()  # Trainer calls backward on every batch


def test_lora_wraps_model_and_trains_only_adapters():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from mk33.train.sft import add_lora, palm_loss

    model = add_lora(_tiny_qwen().half(), {"r": 4, "alpha": 8, "dropout": 0.0}, gradient_checkpointing=False)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert all(p.dtype == torch.float32 for p in trainable.values())  # fp16 base, fp32 adapters
    model.float()  # CPU has no fp16 matmul for every op; the dtype check above is what matters
    ids, att, labels, w = _batch(torch)
    palm_loss(model, ids, att, labels, w).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable.values())


def test_colab_configs_match_the_bench_agent():
    """Training windows must be built exactly the way bench-run builds contexts."""
    from mk33.bench.run import BENCH_TOOLS, BenchConfig
    from mk33.config import load_config

    palm = load_config("configs/mk33/colab/palm.yaml")
    uniform = load_config("configs/mk33/colab/uniform.yaml")
    bench = BenchConfig(window=8000).rlc_for("rlc")
    for cfg in (palm, uniform):
        assert cfg.sft.tools == BENCH_TOOLS
        assert cfg.rlc_train == bench
        assert cfg.sft.lora and cfg.sft.dtype == "float16" and cfg.sft.attn_implementation == "sdpa"
        assert not cfg.sft.deepspeed
    assert (palm.palm.mode, uniform.palm.mode) == ("palm", "all")
    # The two arms differ only in the loss mask (and which files they read and write).
    assert palm.selection == uniform.selection
    own = {"output_dir", "train_files"}
    assert {k: v for k, v in vars(palm.sft).items() if k not in own} == \
           {k: v for k, v in vars(uniform.sft).items() if k not in own}


def test_training_and_inference_render_tool_calls_identically():
    """A window's text at training time equals the prompt the local client builds."""
    from mk33.chat import template_messages
    from mk33.scaffold.context import turn_messages
    from mk33.trajectory import Turn

    class DictOnlyTokenizer(FakeTokenizer):
        # Renders arguments with repr(): a string and a dict would give different text.
        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
            parts = []
            for m in messages:
                calls = "".join(repr(c["function"]["arguments"]) for c in m.get("tool_calls") or [])
                parts.append(f"<{m['role']}>{m.get('content') or ''}{calls}</>")
            return "".join(parts)

    turn = Turn(index=0, round=1, reasoning="look", tool="read_file", args={"path": "a.py", "end": 5},
                observation="1\tx = 1")
    msgs = [{"role": "user", "content": "task"}, *turn_messages(turn)]
    tok = DictOnlyTokenizer()
    ids, _ = message_token_weights(tok, msgs, [None, 1.0, None], max_len=10_000)
    assert "".join(map(chr, ids)) == tok.apply_chat_template(template_messages(msgs))
