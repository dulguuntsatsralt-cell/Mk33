"""Re-try rate: rounds that repeat a solution already submitted or an idea that already failed."""

from conftest import make_traj, make_turn
from mk33.metrics import process_metrics, retry_rate
from mk33.trajectory import Submission


def run(*rounds):
    """rounds: (normalized score or None for a failed run, hypothesis, solution text)."""
    turns = []
    for i, (score, idea, code) in enumerate(rounds):
        sub = Submission(round=i + 1, score=score, normalized=score, ok=score is not None, hypothesis=idea,
                         lesson="", solution_files={"solution.py": code})
        turns.append(make_turn(i, "submit", sub=sub, rnd=i + 1))
    return make_traj(turns)


def test_exact_resubmission_is_a_retry_even_with_whitespace_changes():
    t = run((0.2, "greedy by value density", "x = 1"),
            (0.5, "dynamic programming over capacity", "x = 2"),
            (0.5, "try something", "x  =  2\n"))
    assert retry_rate(t) == 0.5  # round 3 repeats round 2's solution


def test_repeating_a_failed_idea_is_a_retry():
    t = run((0.3, "sort items by value to weight ratio then fill greedily", "a"),
            (0.1, "add random restarts to the local search over item swaps", "b"),   # did not improve
            (0.6, "dynamic programming table over capacity", "c"),
            (0.6, "random restarts for the local search over item swaps again", "d"))  # same dead idea
    assert retry_rate(t) == 1 / 3


def test_repeating_an_idea_that_worked_is_not_a_retry():
    t = run((0.3, "dynamic programming table over capacity", "a"),
            (0.6, "dynamic programming table over capacity with pruning", "b"))
    assert retry_rate(t) == 0.0


def test_failed_rounds_count_as_dead_ends():
    t = run((None, "vectorize the inner loop with numpy arrays", "a"),
            (0.4, "vectorize the inner loop with numpy arrays", "b"))
    assert retry_rate(t) == 1.0


def test_vague_ideas_are_not_compared():
    t = run((0.0, "fix", "a"), (0.0, "fix", "b"), (0.0, "fix", "c"))
    assert retry_rate(t) == 0.0  # different code, too few words to call the ideas the same


def test_needs_two_rounds_and_appears_in_process_metrics():
    assert retry_rate(run((0.5, "one idea only here", "a"))) is None
    t = run((0.2, "a b c d", "x"), (0.2, "e f g h", "x"))
    assert process_metrics(t).retry_rate == 1.0
