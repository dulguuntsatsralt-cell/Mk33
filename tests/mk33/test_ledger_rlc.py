from mk33.scaffold.ledger import LedgerEntry, RoundLedger, strategy_family
from mk33.scaffold.rlc import RLCConfig, RLCContext, TurnMessages


def _ledger(n=6):
    led = RoundLedger()
    for r in range(1, n + 1):
        ok = r % 2 == 0
        led.add(LedgerEntry(round=r, hypothesis=f"tune learning rate variant {r}" if r < 4 else f"bigger batch {r}",
                            score=r / 10 if ok else None, normalized=r / 10 if ok else None,
                            error_signature=None if ok else "CUDA OOM", lesson=f"lesson {r}" * 3))
    return led


def test_strategy_family():
    assert strategy_family("Try a larger learning rate with warmup") == "larger learning rate"


def test_ledger_delta_and_best():
    led = _ledger()
    assert led.best().round == 6
    assert led.entries[3].delta_vs_best == 0.4 - 0.2
    assert "error: CUDA OOM" in led.render()


def test_ledger_compact_keeps_best_and_failures_as_merged():
    led = _ledger()
    before = led.tokens()
    merges = led.compact(cap=before // 2)
    assert merges >= 1 and led.tokens() <= before
    assert led.best().round == 6  # best never merged
    rounds = sorted(r for e in led.entries for r in (e.merged_rounds or [e.round]))
    assert rounds == [1, 2, 3, 4, 5, 6]  # no round disappears
    assert any("errored" in e.lesson for e in led.entries if e.merged_rounds)


def _turns(n_rounds, per_round=3, size=800):
    out, i = [], 0
    for r in range(1, n_rounds + 1):
        for _ in range(per_round):
            out.append(TurnMessages(i, r, [{"role": "assistant", "content": "x" * size},
                                          {"role": "tool", "content": "y" * size}]))
            i += 1
    return out


def test_rlc_compacts_old_rounds_and_freezes_prefix():
    cfg = RLCConfig(window=4000, theta=0.75, keep_rounds=2, ledger_cap=2000)
    ctx = RLCContext(cfg)
    led = _ledger(3)
    pinned = lambda: [{"role": "system", "content": "sys"}]
    turns = _turns(4)
    msgs, ev = ctx.build(pinned, led, turns[:2], 1)
    assert ev is None
    msgs2, ev2 = ctx.build(pinned, led, turns[:3], 1)
    assert msgs2[: len(msgs)] == msgs  # append-only while no event
    msgs3, ev3 = ctx.build(pinned, led, turns, 4)
    assert ev3 is not None and ev3.cutoff_round == 3
    assert any("Round ledger" in m["content"] for m in msgs3)
    assert any("compacted" in m["content"] for m in msgs3)
    kept = [t for t in turns if t.round >= 3]
    assert msgs3[-1] == kept[-1].messages[-1]
    assert sum(len(m["content"]) for m in msgs3) // 4 <= cfg.window


def test_rlc_truncate_mode_has_no_ledger():
    ctx = RLCContext(RLCConfig(window=2000, theta=0.75, mode="truncate"))
    msgs, ev = ctx.build(lambda: [{"role": "system", "content": "s"}], _ledger(2), _turns(4), 4)
    assert ev is not None and ev.kind == "truncate"
    assert not any("Round ledger" in m["content"] for m in msgs)
