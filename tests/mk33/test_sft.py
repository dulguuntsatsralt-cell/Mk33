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
