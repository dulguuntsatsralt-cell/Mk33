import pytest

from mk33.envs.audit import AuditFlag, AuditReport
from mk33.metrics import best_so_far_auc, mean_gain, productive_rounds, recovery_rate
from mk33.selection import SelectionConfig, accept, select
from conftest import make_sub, make_traj, make_turn


def _traj(scores, env="e", domain="ml"):
    turns = [make_turn(i, "submit", {}, sub=make_sub(i + 1, s, ok=s is not None), rnd=i + 1,
                       elapsed=(i + 1) * 100.0) for i, s in enumerate(scores)]
    t = make_traj(turns, domain=domain)
    t.env_id = env
    return t


def test_accept_rules():
    assert accept(_traj([0.5, 0.8, 0.95])).accepted
    d = accept(_traj([0.95, 0.9, 0.9]))
    assert not d.accepted and "no improving round" in d.reasons[0]
    assert not accept(_traj([0.5, 0.95])).accepted  # too few rounds
    assert not accept(_traj([0.5, 0.8, 0.85])).accepted  # below tau
    assert not accept(_traj([0.5, 0.8, 0.94], domain="judge")).accepted
    bad = AuditReport([AuditFlag("canary", None, "x")])
    assert not accept(_traj([0.5, 0.8, 0.95]), audit=bad).accepted


def test_select_per_env_cap():
    ts = [_traj([0.5, 0.8, x]) for x in (0.91, 0.99, 0.95)]
    acc, dec = select(ts, SelectionConfig(max_per_env=2))
    assert sorted(t.best_normalized() for t in acc) == [0.95, 0.99]
    assert "per-env cap" in dec[0].reasons[0]


def test_productive_rounds_and_gain():
    s = [0.2, 0.5, 0.4, 0.6, 0.6, 0.5, 0.55, 0.58, 0.59]
    assert productive_rounds(s, m=5) == 4
    assert productive_rounds([0.1, 0.2], m=5) == 2
    assert mean_gain([0.2, 0.5, 0.4, 0.6]) == pytest.approx(0.2)


def test_recovery_and_auc():
    assert recovery_rate([0.5, 0.3, 0.6, None, 0.4, 0.4, 0.4]) == pytest.approx(1 / 5)
    assert recovery_rate([0.1, 0.2]) is None
    # best 0 until t=50, 0.5 until 100 -> area 25 over budget 100
    assert best_so_far_auc([(0.5, 50.0)], 100.0) == pytest.approx(0.25)
