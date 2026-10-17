"""Text-level memory at a budget: exact length, order, and that retrieval finds the
relevant old chunk. No PyTorch needed."""

import pytest

from mk33.kv.textmem import text_context

WORDS = ["alpha", "beta", "gamma", "delta", "kappa", "omega", "sigma", "theta"]


def decode(ids):
    return " ".join(f"tok{t % 50}" for t in ids)


def test_short_input_is_untouched():
    assert text_context(list(range(40)), 100, "retrieve", decode)[0] == list(range(40))


@pytest.mark.parametrize("mode", ["recent", "retrieve"])
def test_exact_budget_order_sinks_and_recent(mode):
    ids = list(range(2000))
    out, info = text_context(ids, 500, mode, decode, sinks=4, recent=100, chunk=64)
    assert len(out) == 500 and out == sorted(out) and len(set(out)) == 500
    assert out[:4] == [0, 1, 2, 3] and out[-100:] == ids[-100:]


def test_recent_mode_is_truncation():
    out, _ = text_context(list(range(1000)), 300, "recent", decode, sinks=4)
    assert out == [0, 1, 2, 3] + list(range(704, 1000))


def test_retrieve_brings_back_the_matching_old_chunk():
    # Distinct vocabulary per 64-token chunk; the recent window repeats chunk 3's words.
    words = {t: f"filler{t}" for t in range(3000)}
    for t in range(4 + 3 * 64, 4 + 4 * 64):
        words[t] = "xgboost max_depth"
    ids = list(range(2000)) + list(range(4 + 3 * 64, 4 + 3 * 64 + 50))
    dec = lambda xs: " ".join(words[t] for t in xs)  # noqa: E731
    out, info = text_context(ids, 4 + 64 + 50, "retrieve", dec, sinks=4, recent=50, chunk=64)
    assert out[4:68] == list(range(4 + 3 * 64, 4 + 4 * 64))
    assert info["keyword_matches"] == 1 and info["retrieved_chunks"] == 1


def test_unmatched_budget_falls_back_to_the_most_recent_chunks():
    ids = list(range(1000))
    dec = lambda xs: " ".join(f"u{t}" for t in xs)  # no word repeats: no keyword matches  # noqa: E731
    out, info = text_context(ids, 4 + 100 + 50, "retrieve", dec, sinks=4, recent=50, chunk=64)
    assert info["keyword_matches"] == 0
    assert len(out) == 154 and out[4:104] == sorted(out[4:104])
    assert max(out[4:104]) < 950 and min(out[4:104]) > 800  # nearest the recent window


def test_unknown_mode():
    with pytest.raises(ValueError):
        text_context(list(range(10)), 5, "summarize", decode)
