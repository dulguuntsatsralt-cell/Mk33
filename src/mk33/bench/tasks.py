"""Small, CPU-only environments for measuring self-improvement without a GPU.

Each task is generated deterministically from a seed and needs only the Python
standard library to score. A run of the agent on one task takes minutes, not hours,
so a full grid (tasks x context modes x repeats) fits in an afternoon on a laptop
with an API model.

Every task has

* a public workspace: ``README.md`` (the task statement), ``solution.py`` (a weak
  baseline), and a public training split;
* a scorer directory (never shown to the agent): ``score.py``, the held-out split,
  and a stronger stochastic reference solution run with three seeds.

Normalized scores follow the paper: baseline -> 0, reference mean -> 1. Values above
1 beat the reference. ``knapsack`` has a known optimum, so its normalized score has a
ceiling; the other three do not.
"""

from __future__ import annotations

import json
import math
import random
import textwrap
from dataclasses import dataclass
from typing import Callable

Files = dict[str, str]


@dataclass
class BenchTask:
    task_id: str
    title: str
    statement: str
    higher_is_better: bool
    generate: Callable[[random.Random], tuple[Files, Files]]  # -> (public, heldout)
    score_py: str
    baseline_py: str
    reference_py: str
    timeout_s: int = 300


def _csv(header: list[str], rows: list[list]) -> str:
    out = [",".join(header)]
    for r in rows:
        out.append(",".join(f"{v:.6f}" if isinstance(v, float) else str(v) for v in r))
    return "\n".join(out) + "\n"


# Shared by every score.py: load the solution module from the submitted directory.
_LOADER = '''
import importlib.util, json, sys, time
from pathlib import Path

def load_solution(solution_dir):
    sys.path.insert(0, solution_dir)
    spec = importlib.util.spec_from_file_location("solution", Path(solution_dir) / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def read_csv(path):
    lines = Path(path).read_text().strip().splitlines()
    return lines[0].split(","), [[float(x) for x in l.split(",")] for l in lines[1:]]

solution_dir = sys.argv[sys.argv.index("--solution") + 1]
sol = load_solution(solution_dir)
t0 = time.time()
'''


def _statement(title: str, body: str) -> str:
    common = textwrap.dedent("""
        ## How you are scored

        Call the `submit` tool to score the current `solution.py` on a hidden held-out split.
        Each submit ends a round; give a short hypothesis (what you tried) and a one-line lesson.
        You have many rounds: iterate, keep what helps, revert what hurts. The held-out data
        and the scoring script are not in your workspace; do not look for them.

        Rules: keep the function signature, keep the solution deterministic for a fixed input,
        and keep one scoring call under 60 seconds on one CPU core. Only the Python standard
        library is guaranteed to be installed.
        """).strip()
    return f"# {title}\n\n{textwrap.dedent(body).strip()}\n\n{common}\n"


# --------------------------------------------------------------------------- regression

def _gen_regression(rng: random.Random) -> tuple[Files, Files]:
    def row():
        x = [rng.random() for _ in range(10)]
        y = (10 * math.sin(math.pi * x[0] * x[1]) + 20 * (x[2] - 0.5) ** 2 + 10 * x[3] + 5 * x[4]
             + rng.gauss(0, 1.0))
        return x, y

    train = [row() for _ in range(800)]
    test = [row() for _ in range(2000)]
    head = [f"x{i}" for i in range(10)]
    train_csv = _csv(head + ["y"], [x + [y] for x, y in train])
    public = {"train.csv": train_csv}
    heldout = {
        "heldout/train.csv": train_csv,
        "heldout/test_X.csv": _csv(head, [x for x, _ in test]),
        "heldout/test_y.json": json.dumps([y for _, y in test]),
    }
    return public, heldout


REG_SCORE = _LOADER + '''
_, train = read_csv("heldout/train.csv")
_, test_X = read_csv("heldout/test_X.csv")
test_y = json.loads(Path("heldout/test_y.json").read_text())
pred = list(sol.fit_predict([r[:-1] for r in train], [r[-1] for r in train], test_X))
if len(pred) != len(test_y):
    raise ValueError(f"expected {len(test_y)} predictions, got {len(pred)}")
rmse = (sum((float(p) - y) ** 2 for p, y in zip(pred, test_y)) / len(test_y)) ** 0.5
print(f"rmse {rmse:.4f} ({time.time() - t0:.1f}s)")
print(json.dumps({"score": rmse}))
'''

REG_BASELINE = '''
def fit_predict(train_X, train_y, test_X):
    """Predict the training mean for every test row."""
    mean = sum(train_y) / len(train_y)
    return [mean for _ in test_X]
'''

REG_REFERENCE = '''
import random
from pathlib import Path

SEED = int(Path(__file__).with_name("seed.txt").read_text())


def _solve(a, b):
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[c][c]:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def fit_predict(train_X, train_y, test_X):
    """Ridge regression on a seeded 90% bootstrap of the training rows."""
    rng = random.Random(SEED)
    idx = [rng.randrange(len(train_X)) for _ in range(int(0.9 * len(train_X)))]
    X = [[1.0] + train_X[i] for i in idx]
    y = [train_y[i] for i in idx]
    d = len(X[0])
    xtx = [[sum(r[i] * r[j] for r in X) + (1e-3 if i == j else 0) for j in range(d)] for i in range(d)]
    xty = [sum(r[i] * t for r, t in zip(X, y)) for i in range(d)]
    w = _solve(xtx, xty)
    return [sum(wi * xi for wi, xi in zip(w, [1.0] + x)) for x in test_X]
'''

REG_STATEMENT = _statement("Tabular regression", """
    Predict a continuous target `y` from ten features `x0`..`x9`, each in [0, 1].
    `train.csv` holds 800 labeled rows. The hidden test split has 2,000 rows from the same
    distribution. Some features may be irrelevant; the target may be nonlinear.

    Implement `fit_predict(train_X, train_y, test_X)` in `solution.py`. It receives lists of
    rows (lists of floats) and must return one float prediction per test row.

    Metric: RMSE on the hidden test split (lower is better).
""")


# ---------------------------------------------------------------- text classification

_POS = ["good", "great", "excellent", "lovely", "fantastic", "enjoyable", "solid", "fun", "brilliant",
        "charming", "moving", "clever"]
_NEG = ["bad", "awful", "boring", "poor", "terrible", "dull", "weak", "messy", "annoying", "bland",
        "clumsy", "forgettable"]
_TOPIC = ["plot", "acting", "camera work", "story", "ending", "soundtrack", "pacing", "cast", "script",
          "dialogue", "lighting", "editing", "set design", "lead actor", "villain", "finale"]
_FILL = ["honestly", "overall", "to be fair", "i think", "frankly", "in my view", "at times", "mostly"]


def _gen_text(rng: random.Random) -> tuple[Files, Files]:
    def review():
        label = rng.randint(0, 1)
        clauses = []
        for _ in range(rng.randint(2, 4)):
            supportive = rng.random() < 0.8
            negate = rng.random() < 0.25
            # The word's own polarity: supports the label unless (not supportive) xor negated.
            word_pos = (label == 1) == (supportive != negate)
            w = rng.choice(_POS if word_pos else _NEG)
            adj = f"not {w}" if negate else w
            clause = f"the {rng.choice(_TOPIC)} was {adj}"
            if rng.random() < 0.4:
                clause = f"{rng.choice(_FILL)} {clause}"
            clauses.append(clause)
        text = ", ".join(clauses)
        if rng.random() < 0.03:  # label noise
            label = 1 - label
        return text, label

    train = [review() for _ in range(600)]
    test = [review() for _ in range(1500)]
    train_json = json.dumps([{"text": t, "label": y} for t, y in train], indent=0)
    public = {"train.json": train_json}
    heldout = {
        "heldout/train.json": train_json,
        "heldout/test_texts.json": json.dumps([t for t, _ in test]),
        "heldout/test_labels.json": json.dumps([y for _, y in test]),
    }
    return public, heldout


TEXT_SCORE = _LOADER + '''
train = json.loads(Path("heldout/train.json").read_text())
texts = json.loads(Path("heldout/test_texts.json").read_text())
labels = json.loads(Path("heldout/test_labels.json").read_text())
pred = list(sol.fit_predict([r["text"] for r in train], [r["label"] for r in train], texts))
if len(pred) != len(labels):
    raise ValueError(f"expected {len(labels)} predictions, got {len(pred)}")
acc = sum(int(p) == y for p, y in zip(pred, labels)) / len(labels)
print(f"accuracy {acc:.4f} ({time.time() - t0:.1f}s)")
print(json.dumps({"score": acc}))
'''

TEXT_BASELINE = '''
def fit_predict(train_texts, train_labels, test_texts):
    """Predict the majority class."""
    majority = int(sum(train_labels) * 2 >= len(train_labels))
    return [majority for _ in test_texts]
'''

TEXT_REFERENCE = '''
import math
import random
from collections import Counter
from pathlib import Path

SEED = int(Path(__file__).with_name("seed.txt").read_text())


def fit_predict(train_texts, train_labels, test_texts):
    """Unigram multinomial naive Bayes on a seeded 90% subsample."""
    rng = random.Random(SEED)
    idx = rng.sample(range(len(train_texts)), int(0.9 * len(train_texts)))
    counts = {0: Counter(), 1: Counter()}
    prior = Counter()
    for i in idx:
        counts[train_labels[i]].update(train_texts[i].replace(",", " ").split())
        prior[train_labels[i]] += 1
    vocab = set(counts[0]) | set(counts[1])
    tot = {c: sum(counts[c].values()) for c in (0, 1)}

    def score(text, c):
        s = math.log(prior[c] / len(idx))
        for w in text.replace(",", " ").split():
            s += math.log((counts[c][w] + 1) / (tot[c] + len(vocab)))
        return s

    return [int(score(t, 1) > score(t, 0)) for t in test_texts]
'''

TEXT_STATEMENT = _statement("Sentiment classification", """
    Classify short movie reviews as positive (1) or negative (0). `train.json` holds 600
    labeled reviews (`{"text", "label"}`). The hidden test split has 1,500 reviews from the
    same distribution. About 3% of labels are noisy.

    Implement `fit_predict(train_texts, train_labels, test_texts)` in `solution.py`. It must
    return one integer label per test text.

    Metric: accuracy on the hidden test split (higher is better).
""")


# ------------------------------------------------------------------------- forecasting

def _gen_forecast(rng: random.Random) -> tuple[Files, Files]:
    hist_len, horizon = 168, 28
    history, future = [], []
    for _ in range(30):
        level = rng.uniform(20, 200)
        slope = rng.uniform(-0.3, 0.6) * level / 100
        shape = [rng.gauss(0, 1) for _ in range(7)]
        amp = rng.uniform(0.05, 0.3) * level
        noise = rng.uniform(0.02, 0.08) * level
        shift_at = rng.randrange(60, hist_len - 20) if rng.random() < 0.3 else None
        shift = rng.uniform(-0.25, 0.25) * level
        ys = []
        for t in range(hist_len + horizon):
            y = level + slope * t + amp * shape[t % 7] + rng.gauss(0, noise)
            if shift_at is not None and t >= shift_at:
                y += shift
            ys.append(round(y, 4))
        history.append(ys[:hist_len])
        future.append(ys[hist_len:])
    hist_json = json.dumps({"horizon": horizon, "series": history})
    public = {"history.json": hist_json}
    heldout = {"heldout/history.json": hist_json, "heldout/future.json": json.dumps(future)}
    return public, heldout


FC_SCORE = _LOADER + '''
h = json.loads(Path("heldout/history.json").read_text())
future = json.loads(Path("heldout/future.json").read_text())
pred = list(sol.forecast([list(s) for s in h["series"]], h["horizon"]))
if len(pred) != len(future) or any(len(p) != len(f) for p, f in zip(pred, future)):
    raise ValueError("forecast must return one list of `horizon` floats per series")
errs = []
for hist, p, f in zip(h["series"], pred, future):
    scale = sum(abs(hist[t] - hist[t - 7]) for t in range(7, len(hist))) / (len(hist) - 7)
    errs.append(sum(abs(float(a) - b) for a, b in zip(p, f)) / len(f) / scale)
mase = sum(errs) / len(errs)
print(f"MASE {mase:.4f} ({time.time() - t0:.1f}s)")
print(json.dumps({"score": mase}))
'''

FC_BASELINE = '''
def forecast(series, horizon):
    """Repeat the last observed value."""
    return [[s[-1]] * horizon for s in series]
'''

FC_REFERENCE = '''
import random
from pathlib import Path

SEED = int(Path(__file__).with_name("seed.txt").read_text())


def forecast(series, horizon):
    """Seasonal naive (period 7) plus a drift estimated on a seeded window of recent weeks."""
    rng = random.Random(SEED)
    out = []
    for s in series:
        weeks = rng.randint(3, 6)
        a, b = s[-7 * (weeks + 1):-7 * weeks], s[-7:]
        drift = (sum(b) - sum(a)) / 7 / (7 * weeks)
        out.append([s[-7 + (h % 7)] + drift * (h + 1) for h in range(horizon)])
    return out
'''

FC_STATEMENT = _statement("Weekly-seasonal forecasting", """
    Forecast 30 daily time series 28 steps ahead. `history.json` holds
    `{"horizon": 28, "series": [[...168 floats...], ...]}`. Series have a level, a trend, a
    weekly pattern and noise; some have a level shift partway through.

    Implement `forecast(series, horizon)` in `solution.py`. It must return one list of
    `horizon` floats per series.

    Metric: MASE, the mean absolute error divided by each series' in-sample seasonal-naive
    error, averaged over series (lower is better).
""")


# ---------------------------------------------------------------------------- knapsack

def _knapsack_opt(w: list[int], v: list[int], cap: int) -> int:
    best = [0] * (cap + 1)
    for wi, vi in zip(w, v):
        for c in range(cap, wi - 1, -1):
            if best[c - wi] + vi > best[c]:
                best[c] = best[c - wi] + vi
    return best[cap]


def _gen_knapsack(rng: random.Random) -> tuple[Files, Files]:
    def instance(kind: str):
        n = rng.randint(60, 90)
        w = [rng.randint(10, 100) for _ in range(n)]
        if kind == "strong":
            v = [x + 10 for x in w]
        elif kind == "inverse":
            v = [max(1, 110 - x + rng.randint(-5, 5)) for x in w]
        else:
            v = [rng.randint(10, 100) for _ in range(n)]
        cap = sum(w) // rng.choice([2, 3, 4])
        return {"weights": w, "values": v, "capacity": cap}

    kinds = ["strong", "inverse", "uncorrelated"]
    examples = [instance(kinds[i % 3]) for i in range(6)]
    for ex in examples:
        ex["optimum"] = _knapsack_opt(ex["weights"], ex["values"], ex["capacity"])
    tests = [instance(kinds[i % 3]) for i in range(24)]
    opts = [_knapsack_opt(t["weights"], t["values"], t["capacity"]) for t in tests]
    public = {"examples.json": json.dumps(examples)}
    heldout = {"heldout/tests.json": json.dumps(tests), "heldout/optima.json": json.dumps(opts)}
    return public, heldout


KS_SCORE = _LOADER + '''
tests = json.loads(Path("heldout/tests.json").read_text())
opts = json.loads(Path("heldout/optima.json").read_text())
ratios = []
for inst, opt in zip(tests, opts):
    chosen = sorted(set(int(i) for i in sol.solve(list(inst["weights"]), list(inst["values"]),
                                                    inst["capacity"])))
    if any(i < 0 or i >= len(inst["weights"]) for i in chosen):
        raise ValueError("solve returned an index out of range")
    weight = sum(inst["weights"][i] for i in chosen)
    value = sum(inst["values"][i] for i in chosen)
    ratios.append(value / opt if weight <= inst["capacity"] else 0.0)
gap = 1 - sum(ratios) / len(ratios)
infeasible = sum(r == 0.0 for r in ratios)
print(f"mean optimality gap {gap:.5f}, infeasible {infeasible} ({time.time() - t0:.1f}s)")
print(json.dumps({"score": gap}))
'''

KS_BASELINE = '''
def solve(weights, values, capacity):
    """Take items in the given order while they fit."""
    chosen, used = [], 0
    for i, w in enumerate(weights):
        if used + w <= capacity:
            chosen.append(i)
            used += w
    return chosen
'''

KS_REFERENCE = '''
import random
from pathlib import Path

SEED = int(Path(__file__).with_name("seed.txt").read_text())


def solve(weights, values, capacity):
    """Value/weight greedy with seeded +-3% noise on the ratios (a randomized greedy)."""
    rng = random.Random(SEED)
    key = {i: values[i] / weights[i] * rng.uniform(0.97, 1.03) for i in range(len(weights))}
    chosen, used = [], 0
    for i in sorted(key, key=lambda i: -key[i]):
        if used + weights[i] <= capacity:
            chosen.append(i)
            used += weights[i]
    return sorted(chosen)
'''

KS_STATEMENT = _statement("0/1 knapsack heuristics", """
    Solve 0/1 knapsack instances: choose items to maximize total value without exceeding the
    capacity. `examples.json` holds six public instances with their optimal values
    (`{"weights", "values", "capacity", "optimum"}`). The hidden test set has 24 instances
    of the same three kinds (strongly correlated, inversely correlated, uncorrelated), with
    60 to 90 items each.

    Implement `solve(weights, values, capacity)` in `solution.py`. It must return a list of
    chosen item indices. An over-capacity answer scores 0 on that instance.

    Metric: mean optimality gap, 1 - mean(value / optimum) (lower is better; 0 is optimal).
""")


TASKS: list[BenchTask] = [
    BenchTask("regression", "Tabular regression", REG_STATEMENT, False, _gen_regression,
              REG_SCORE, REG_BASELINE, REG_REFERENCE),
    BenchTask("sentiment", "Sentiment classification", TEXT_STATEMENT, True, _gen_text,
              TEXT_SCORE, TEXT_BASELINE, TEXT_REFERENCE),
    BenchTask("forecast", "Weekly-seasonal forecasting", FC_STATEMENT, False, _gen_forecast,
              FC_SCORE, FC_BASELINE, FC_REFERENCE),
    BenchTask("knapsack", "0/1 knapsack heuristics", KS_STATEMENT, False, _gen_knapsack,
              KS_SCORE, KS_BASELINE, KS_REFERENCE),
]

TASKS_BY_ID = {t.task_id: t for t in TASKS}
