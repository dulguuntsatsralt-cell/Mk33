import dataclasses

import pytest

from mk33.envs import (AuditConfig, BenchmarkIndex, BenchmarkTask, FilterConfig, IsolatedScorer,
                              ScorerTampered, audit, decontaminate, headroom_check, normalized_noise,
                              normalized_score)
from conftest import make_sub, make_traj, make_turn


# ---------------------------------------------------------------- normalize
def test_normalized_score_endpoints():
    assert normalized_score(0.5, 0.5, 0.9) == 0
    assert normalized_score(0.9, 0.5, 0.9) == pytest.approx(1)
    assert normalized_score(1.0, 0.5, 0.9) > 1


def test_normalized_score_lower_is_better():
    # RMSE: baseline 10, reference 2; score 4 is 75% of the way.
    assert normalized_score(4, 10, 2, higher_is_better=False) == pytest.approx(0.75)


def test_normalized_noise_and_degenerate():
    assert normalized_noise(0.01, 0.5, 0.9) == pytest.approx(0.025)
    with pytest.raises(ValueError):
        normalized_score(1, 1, 1)


# ------------------------------------------------------------------- filter
def test_headroom_filter_sigma_and_margin(toy_env):
    assert headroom_check(toy_env).kept  # gap 42 vs tiny noise
    noisy = dataclasses.replace(toy_env, baseline_score=-0.2, reference_scores=[0.0, -0.3, 0.3])
    r = headroom_check(noisy)
    assert not r.kept and r.required_gap == pytest.approx(3 * 0.3)
    assert headroom_check(noisy, FilterConfig(use_sigma=False)).kept  # rel term is 5% of |0.0| = 0
    assert headroom_check(noisy, FilterConfig(enabled=False)).kept


def test_headroom_filter_relative_margin(toy_env):
    env = dataclasses.replace(toy_env, baseline_score=0.97, reference_scores=[1.0, 1.0, 1.0])
    assert not headroom_check(env).kept  # gap 0.03 < 5% of 1.0
    assert headroom_check(env, FilterConfig(use_rel_margin=False)).kept  # sigma 0


def test_headroom_filter_needs_reference_runs(toy_env):
    env = dataclasses.replace(toy_env, reference_scores=[0.0])
    assert "reference runs" in headroom_check(env).reason


def test_headroom_filter_judge(toy_env):
    env = dataclasses.replace(toy_env, domain="judge", baseline_score=0.5, reference_scores=[1, 1, 1],
                              metadata={"score_kind": "pass_fraction"})
    assert headroom_check(env).kept
    env.baseline_score = 0.7
    assert not headroom_check(env).kept


# ----------------------------------------------------------------- decontam
def test_decontam_rules(toy_env):
    stmt = " ".join(f"w{i}" for i in range(40))
    idx = BenchmarkIndex([
        BenchmarkTask("mle/leaf", "mle", statement=stmt, dataset_urls=["https://www.kaggle.com/c/leaf/"],
                      problem_ids=["leaf-classification"]),
    ])
    clean = dataclasses.replace(toy_env)
    assert idx.check(clean) == []
    by_url = dataclasses.replace(toy_env, dataset_urls=["http://kaggle.com/c/leaf"])
    assert [h.rule for h in idx.check(by_url)] == ["dataset_url"]
    by_pid = dataclasses.replace(toy_env, problem_ids=["Leaf Classification"])
    assert [h.rule for h in idx.check(by_pid)] == ["problem_id"]
    by_text = dataclasses.replace(toy_env, task_statement=stmt + " extra words here")
    assert [h.rule for h in idx.check(by_text)] == ["ngram"]
    kept, dropped, per_rule = decontaminate([clean, by_url, by_text], idx)
    assert len(kept) == 1 and per_rule == {"dataset_url": 1, "ngram": 1}


# ------------------------------------------------------------------- scorer
def test_isolated_scorer_scores_and_normalizes(toy_env):
    r = IsolatedScorer(toy_env).score(toy_env.workspace_dir)
    assert r.ok and r.score == -42.0 and r.normalized == pytest.approx(0.0)


def test_isolated_scorer_detects_tampering(toy_env, tmp_path):
    with open(f"{toy_env.scorer.scorer_dir}/score.py", "a") as fh:
        fh.write("\n# tampered\n")
    with pytest.raises(ScorerTampered):
        IsolatedScorer(toy_env).score(toy_env.workspace_dir)


def test_isolated_scorer_failure_signature(toy_env):
    with open(f"{toy_env.workspace_dir}/solution.py", "w") as fh:
        fh.write("VALUE = undefined_name\n")
    r = IsolatedScorer(toy_env).score(toy_env.workspace_dir)
    assert not r.ok and "NameError" in r.error_signature


# -------------------------------------------------------------------- audit
def test_audit_flags(toy_env):
    sc = toy_env.scorer.scorer_dir
    turns = [
        make_turn(0, "read_file", {"path": "solution.py"}),
        make_turn(1, "bash", {"command": f"cat {sc}/heldout/target.txt"}),
        make_turn(2, "read_file", {"path": "../scorer/score.py"}),
        make_turn(3, "web_fetch", {"url": "https://github.com/example/toy/blob/main/best.py"}),
        make_turn(4, "submit", {}, sub=make_sub(1, 0.5, files={"solution.py": "OUT = 'secret-hidden-output-0123456789-abcdefgh'"})),
    ]
    rep = audit(make_traj(turns), toy_env, AuditConfig(),
                hidden_outputs=["secret-hidden-output-0123456789-abcdefgh"], canary=(1.0, 0.5))
    checks = sorted({f.check for f in rep.flags})
    assert checks == ["canary", "rules.hidden_literal", "rules.hidden_path",
                      "rules.outside_workspace", "rules.reference_fetch"]
    clean = audit(make_traj(turns[:1]), toy_env)
    assert clean.passed
