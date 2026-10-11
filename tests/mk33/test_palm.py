import pytest

from mk33.palm import PALMConfig, compute_weights, is_null, is_polling, is_redundant, provenance_set
from mk33.palm.canonical import normalize_observation
from mk33.palm.provenance import hunk_survives, parse_hunks
from mk33.palm.validate import cohen_kappa, evaluate
from conftest import make_sub, make_traj, make_turn

DIFF_A = """--- a/model.py
+++ b/model.py
@@ -1,2 +1,3 @@
 import torch
-lr = 0.1
+lr = 0.01
+warmup_steps = 500
"""
DIFF_B = """--- a/model.py
+++ b/model.py
@@ -1,3 +1,3 @@
 import torch
-lr = 0.01
+lr = 0.5
 warmup_steps = 500
"""


def test_normalize_observation_strips_noise():
    raw = "2026-10-05 12:00:01 loss 0.3\x1b[0m\n 45%|####      | 9/20 [00:01<00:02]\npid=1234 done in 3.2s"
    assert normalize_observation(raw) == "<T> loss 0.3\npid <N> done in <D>"


def test_redundant_requires_unchanged_workspace():
    turns = [
        make_turn(0, "bash", {"command": "python train.py --epochs 3"}, "acc 0.5", ws="h1"),
        make_turn(1, "bash", {"command": "python  train.py --epochs 3"}, "acc 0.5", ws="h1"),
        make_turn(2, "edit_file", {"path": "m.py", "old_str": "a", "new_str": "b"}, "diff", ws="h2"),
        make_turn(3, "bash", {"command": "python train.py --epochs 3"}, "acc 0.6", ws="h2"),
    ]
    assert is_redundant(turns, 1)
    assert not is_redundant(turns, 3)  # re-run after an edit is a new experiment


def test_null_observation():
    turns = [
        make_turn(0, "read_file", {"path": "a.py"}, "1\tx = 1", ws="h1"),
        make_turn(1, "read_file", {"path": "a.py"}, "1\tx = 1", ws="h1"),
        make_turn(2, "bash", {"command": "true"}, "[exit 0]\n", ws="h1"),
        make_turn(3, "bash", {"command": "pip install -q foo"}, "[exit 0]\n", ws="h2"),
    ]
    assert is_null(turns, 1)
    assert is_null(turns, 2)
    assert not is_null(turns, 3)  # empty output but the workspace changed


def test_polling():
    turns = [
        make_turn(0, "run_job", {"command": "python train.py"}, "started job j1"),
        make_turn(1, "wait_job", {"job_id": "j1"}, "[job j1: running]\nepoch 1 loss 0.9", timed_out=True),
        make_turn(2, "wait_job", {"job_id": "j1"}, "[job j1: running]\n(no new output)", timed_out=True),
        make_turn(3, "bash", {"command": "tail -n 5 train.log"}, "[exit 0]\nepoch 1"),
        make_turn(4, "bash", {"command": "tail -n 5 train.log"}, "[exit 0]\nepoch 1"),
        make_turn(5, "bash", {"command": "tail -n 5 train.log"}, "[exit 0]\nepoch 1\nepoch 2"),
    ]
    assert not is_polling(turns, 1)
    assert is_polling(turns, 2)
    assert not is_polling(turns, 3)
    assert is_polling(turns, 4)
    assert not is_polling(turns, 5)


def test_hunk_survival():
    h = parse_hunks(DIFF_A)[0]
    assert h.path == "model.py" and h.added == ["lr = 0.01", "warmup_steps = 500"]
    assert hunk_survives(h, {"model.py": "import torch\nlr = 0.01\n\nwarmup_steps = 500\n"})
    assert not hunk_survives(h, {"model.py": "import torch\nlr = 0.5\nwarmup_steps = 500\n"})


def _improvement_traj():
    good = {"model.py": "import torch\nlr = 0.01\nwarmup_steps = 500\n"}
    bad = {"model.py": "import torch\nlr = 0.5\nwarmup_steps = 500\n"}
    turns = [
        make_turn(0, "read_file", {"path": "model.py"}, "1\timport torch\n2\tlr = 0.1", ws="h0"),
        make_turn(1, "web_search", {"query": "cats"}, "unrelated result about cats", ws="h0"),
        make_turn(2, "edit_file", {"path": "model.py"}, DIFF_A, ws="h1", diff=DIFF_A,
                  reasoning="lr = 0.1 in model.py is too high"),
        make_turn(3, "submit", {}, "score", ws="h1", sub=make_sub(1, 0.6, files=good)),
        make_turn(4, "edit_file", {"path": "model.py"}, DIFF_B, ws="h2", diff=DIFF_B, rnd=2),
        make_turn(5, "submit", {}, "score", ws="h2", rnd=2, sub=make_sub(2, 0.3, files=bad)),
        make_turn(6, "edit_file", {"path": "model.py"}, "revert", ws="h3", rnd=3,
                  diff=DIFF_B.replace("-lr = 0.01\n+lr = 0.5", "-lr = 0.5\n+lr = 0.01")),
        make_turn(7, "submit", {}, "score", ws="h3", rnd=3, sub=make_sub(3, 0.7, files=good)),
        make_turn(8, "read_file", {"path": "model.py"}, "same", ws="h3", rnd=4),
        make_turn(9, "read_file", {"path": "model.py"}, "same", ws="h3", rnd=4),
    ]
    return make_traj(turns, eps=0.01)


def test_provenance_set():
    s = provenance_set(_improvement_traj())
    assert s[3] == "improving_submission" and s[7] == "improving_submission"
    assert s[2] == "surviving_edit"
    assert s[0] == "supporting_evidence"  # read of model.py used by the edit
    assert 1 not in s  # unrelated search
    assert 4 not in s and 5 not in s  # regression edit and submit
    assert s[6] in ("surviving_edit", "recovery")


def test_weights_modes():
    t = _improvement_traj()
    w = {x.index: x for x in compute_weights(t, PALMConfig(lam=0.5))}
    assert w[2].weight == 1 and w[1].weight == 0.5 and w[4].weight == 0.5
    assert w[9].weight == 0 and "R" in w[9].reasons
    arex = {x.index: x.weight for x in compute_weights(t, PALMConfig(mode="arex"))}
    assert arex[1] == 1 and arex[9] == 0
    assert all(x.weight == 1 for x in compute_weights(t, PALMConfig(mode="all")))
    with pytest.raises(ValueError):
        compute_weights(t, PALMConfig(mode="nope"))


def test_validate():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == 1.0
    t = _improvement_traj()
    ws = compute_weights(t)
    gold = ["progress" if x.weight == 1 else "no_progress" if x.weight == 0 else "neutral" for x in ws]
    out = evaluate(ws, gold, gold)
    assert out["zero_rules"]["precision"] == 1.0 and out["kappa_rules_vs_gold"] == 1.0
