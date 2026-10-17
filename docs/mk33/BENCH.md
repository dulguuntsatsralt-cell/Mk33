# Mk33 bench

A small benchmark for measuring whether an agent keeps improving its own solution
over many rounds. It runs on one CPU with any OpenAI-compatible model, so it can produce
real numbers before any model has been trained with the full recipe.

It reuses the repository's own components unchanged: the `Agent` loop with Round Ledger
Compaction, the fixed tool set, the isolated scorer with SHA-256 checks, the headroom
filter, the process audit, and the process metrics from Section 4.

## What it measures

The main question is an inference-time version of ablation A2 (`A2_rlc_test_only`):

> With the model held fixed, does the round ledger (RLC) help an agent keep improving
> once its early turns no longer fit in the context window, compared with dropping the
> oldest turns (`truncate`)?

The context window is set to 24K tokens by default, small enough that a 15-round run
must be compacted. Both modes run on the same environments with the same seeds and
budgets, and the report compares them pair by pair.

## Environments

All four are generated from a fixed seed and use only the Python standard library.

| ID | Task | Metric | Baseline | Reference (3 seeds) |
| --- | --- | --- | --- | --- |
| `bench-regression` | 10 features, nonlinear target (Friedman #1), 800 train / 2,000 test | RMSE, lower is better | training mean | ridge regression on a 90% bootstrap |
| `bench-sentiment` | Synthetic movie reviews with negation and 3% label noise, 600 / 1,500 | accuracy | majority class | unigram naive Bayes on a 90% subsample |
| `bench-forecast` | 30 daily series with trend, weekly season, some level shifts; 28-step horizon | MASE, lower is better | last value | seasonal naive with a randomly sized drift window |
| `bench-knapsack` | 24 hidden 0/1 knapsack instances, three hardness kinds, 60 to 90 items | mean optimality gap | take items in order | value/weight greedy with seeded noise |

Measured raw scores (seed 0):

| ID | Baseline | Reference runs |
| --- | --- | --- |
| `bench-regression` | 4.970 | 2.616, 2.614, 2.642 |
| `bench-sentiment` | 0.511 | 0.653, 0.657, 0.659 |
| `bench-forecast` | 4.239 | 1.041, 1.019, 0.984 |
| `bench-knapsack` | 0.321 | 0.0052, 0.0063, 0.0048 |

All four pass the repository's headroom filter (gap at least 3 sigma and at least 5% of
the reference). Scores are normalized as in the paper: baseline 0, reference mean 1.
Above 1 beats the reference. Knapsack has a known optimum, so its normalized score
tops out near 1.02; the exact DP reaches it. The other three have open headroom: a
hand-written k-NN reaches 1.33 on regression and a negation-aware naive Bayes reaches
2.11 on sentiment.

The agent sees `README.md`, a weak `solution.py` and the public training split. The
held-out split, the scoring script and the reference solutions live in a separate scorer
directory. The audit flags any run that touches it.

## Running it

```bash
pip install -e '.[mk33]'
mk33 bench-prepare --out bench/envs

# Check the plumbing without an API key (every score should be 0.000):
mk33 bench-run --dry-run --out bench/dry --repeats 1 --max-rounds 3

# Real run. Any OpenAI-compatible endpoint works: a hosted API, or vLLM/SGLang locally.
export MODEL_API_KEY=...
mk33 bench-run --model <model-name> --base-url <https://.../v1> --out bench/results
mk33 bench-report --out bench/results
```

Defaults: modes `rlc,truncate`, 3 repeats, 15 rounds, 80 tool calls and 45 minutes per
run, 24K window, 4 runs in parallel. That is 24 runs. Runs are resumable: rerunning the
same command skips finished runs, so an interrupted grid picks up where it stopped.

Useful flags: `--tasks regression,sentiment` for a subset, `--repeats 5` for tighter
intervals, `--window 128000` to check that the modes agree when nothing needs
compaction, `--modes rlc,truncate,none` (only with a large window).

Memory study modes: `retrieve` and `rlc+retrieve` add retrieval memory over compacted turns
(any model); `kv:full`, `kv:streaming`, `kv:keynorm`, `kv:h2o` and `kv:random` compress a
local model's KV cache (`--backend hf`). See [MEMORY.md](MEMORY.md).

### Rate-limited and local endpoints

| Flag | Default | Use |
| --- | --- | --- |
| `--rpm N` | 0 (off) | Space model calls to at most N per minute across all runs, for free API tiers. |
| `--timeout S` | 300 | Abandon a single model call after S seconds. |
| `--max-tokens N` | 4096 | Cap the length of each model reply. |
| `--no-think` | off | Append Qwen3's `/no_think` switch to the system prompt. |
| `--workers N` | 4 | Runs in parallel; use 1 for local models and free tiers. |

Rate-limit responses (HTTP 429) are retried with backoff, honoring `Retry-After`; client
errors such as 401 or 402 stop the run at once. A daily request cap ends the remaining
runs with an error; rerun the same command later and the finished runs are skipped.

Each agent step is one model call, and a run that submits several rounds typically needs
40 to 80 steps. Plan budgets accordingly: a 50-request daily cap covers less than one
full run, and very small local models (around 3B parameters) often describe an edit
instead of calling the tool for it, so they rarely get past the baseline.

**Safety.** Bench runs use the local sandbox: the agent runs shell commands on the
machine that runs the bench, inside a copy of the workspace but not in a container. Run
it in a throwaway VM or container, not on a laptop with personal files. `mk33
rollout --sandbox-image` shows how the repository's Docker sandbox is wired in if you
want to extend the bench to it.

## Output

```text
bench/results/
  bench_config.json            budgets and model
  results.jsonl                one row per run
  report.md                    tables below
  summary.json, curves.csv     the same numbers, machine-readable
  runs/<env>/<mode>/rep<k>/
    trajectory.json            full trajectory (feeds mk33 weights / build-sft)
    result.json                metrics for this run
    workspace/                 the agent's final solution
```

`report.md` contains:

1. Per mode: best normalized score, best-so-far AUC over rounds, productive rounds T*,
   mean gain, recovery rate, wasted-turn share, compactions, share of runs that beat the
   reference, and audit pass rate.
2. Best score per environment and mode.
3. Paired comparisons: for each (environment, repeat) pair, the difference between
   modes, with a seeded bootstrap 95% interval. The report only calls a difference real
   when the interval excludes zero.
4. The mean best-so-far curve by round.

## Pilot run

One pair, recorded October 2026, to check the pipeline with a real model:

```bash
mk33 bench-run --model "apodex/apodex-1.1-mini:free" --base-url https://openrouter.ai/api/v1 \
  --tasks knapsack --repeats 7 --modes rlc,truncate \
  --max-tool-calls 25 --max-rounds 8 --workers 1 --rpm 15 --out bench/free
```

| Run | Mode | Rounds submitted | Best normalized score |
| --- | --- | --- | --- |
| rep0 | `truncate` | 1 | 1.017 (the optimum) |
| rep0 | `rlc` | 0 | 0.000 |

The `rlc` run wrote and tested an exact DP solver but used its 25 tool calls before
submitting. The free tier's 50 requests per day then stopped the grid at the third run.
Before the first compaction both modes send nearly the same context, and the ledger is
empty until the first submission, so this pair says nothing about RLC. What it shows:
the pipeline runs end to end against a hosted model, run-to-run variance is large, and
25 tool calls is too tight; use the default 80.

Earlier attempts with local models on a laptop CPU did not finish: Qwen3 with thinking
exceeded the per-call timeout, and a 3B model described edits instead of calling tools.

The full grid (four environments, both modes, three repeats) has not been run yet.

## Reading the results honestly

- These are small, synthetic tasks. A result here is evidence about context management
  in long improvement loops, not a claim about MLE-bench or the paper's full benchmarks.
- With 4 environments and 3 repeats there are 12 pairs per comparison. Expect wide
  intervals; increase `--repeats` before drawing conclusions.
- The trajectories are real teacher rollouts. Concatenated into one `.jsonl` file
  (`jq -c . bench/results/runs/*/*/*/trajectory.json > trajs.jsonl`), they feed
  `mk33 audit`, `mk33 weights` and `mk33 build-sft`, which makes the
  bench a small end-to-end test of the data pipeline as well.
