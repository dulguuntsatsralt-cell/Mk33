# Mk33 bench results

Model: `Qwen/Qwen2.5-7B-Instruct (4bit)`. Runs: 30 (6 failed and excluded).
Scores are normalized per environment: baseline = 0, reference = 1, above 1 beats the reference.

Budget per run: 8 rounds, 40 tool calls, 25 min. Context window 8000 tokens, compaction at 0.75, keep last 2 rounds.

## By context mode (mean ± sd over runs)

| metric | kv:full | retrieve | rlc | truncate |
| --- | --- | --- | --- | --- |
| best score | 0.498 ± 0.545 | 0.725 ± 0.579 | 0.666 ± 0.516 | 0.666 ± 0.516 |
| best-so-far AUC | 0.373 ± 0.409 | 0.402 ± 0.383 | 0.457 ± 0.366 | 0.457 ± 0.366 |
| productive rounds T* | 2.000 ± 2.280 | 2.000 ± 2.449 | 3.333 ± 3.077 | 3.000 ± 2.966 |
| mean gain | 0.498 ± 0.545 | 0.725 ± 0.579 | 0.665 ± 0.515 | 0.665 ± 0.515 |
| recovery rate | 0.000 ± 0.000 | 0.150 ± 0.300 | 0.167 ± 0.333 | 0.333 ± 0.577 |
| wasted-turn share | 0.310 ± 0.306 | 0.304 ± 0.285 | 0.310 ± 0.302 | 0.327 ± 0.293 |
| re-try rate (lower is better) | 0.724 ± 0.185 | 0.776 ± 0.172 | 0.659 ± 0.188 | 0.718 ± 0.185 |
| compactions | 0.000 ± 0.000 | 3.333 ± 2.944 | 0.833 ± 0.983 | 3.167 ± 4.355 |
| peak KV cache (tokens) | 9224.667 ± 4513.765 | 6591.667 ± 1151.652 | 7508.167 ± 2085.827 | 7166.667 ± 2048.745 |
| peak KV cache (MB) | 504.474 ± 246.847 | 360.482 ± 62.981 | 410.603 ± 114.069 | 391.927 ± 112.041 |
| evicted token share | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| prompt tokens per call | 4329.408 ± 1331.117 | 3832.197 ± 839.337 | 3826.006 ± 684.634 | 3764.997 ± 797.594 |
| prefill s per call | 5.065 ± 1.755 | 4.319 ± 1.005 | 4.297 ± 0.890 | 4.226 ± 0.938 |
| decode tokens/s | 7.530 ± 1.851 | 8.146 ± 0.908 | 7.918 ± 1.110 | 8.055 ± 1.011 |
| time in compression | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| beat reference | 0% | 17% | 17% | 17% |
| audit passed | 100% | 100% | 100% | 100% |

## Best score by environment

| environment | kv:full | retrieve | rlc | truncate |
| --- | --- | --- | --- | --- |
| bench-knapsack | 0.995 ± 0.000 | 0.995 ± 0.000 | 0.995 ± 0.000 | 0.995 ± 0.000 |
| bench-regression | 0.000 ± 0.000 | 0.454 ± 0.787 | 0.336 ± 0.582 | 0.336 ± 0.582 |

## Paired comparisons

| comparison | metric | pairs | mean diff | 95% CI | wins (a / b) | real difference? |
| --- | --- | --- | --- | --- | --- | --- |
| kv:full − retrieve | best | 6 | -0.227 | [-0.682, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − retrieve | auc_rounds | 6 | -0.028 | [-0.085, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − retrieve | retry_rate | 6 | -0.052 | [-0.201, +0.067] | 1 / 2 | no, CI includes 0 |
| kv:full − rlc | best | 6 | -0.168 | [-0.504, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − rlc | auc_rounds | 6 | -0.084 | [-0.252, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − rlc | retry_rate | 6 | +0.065 | [+0.000, +0.149] | 2 / 0 | no, CI includes 0 |
| kv:full − truncate | best | 6 | -0.168 | [-0.504, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − truncate | auc_rounds | 6 | -0.084 | [-0.252, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − truncate | retry_rate | 6 | +0.006 | [-0.054, +0.071] | 1 / 1 | no, CI includes 0 |
| retrieve − rlc | best | 6 | +0.059 | [-0.504, +0.682] | 1 / 1 | no, CI includes 0 |
| retrieve − rlc | auc_rounds | 6 | -0.056 | [-0.252, +0.085] | 1 / 1 | no, CI includes 0 |
| retrieve − rlc | retry_rate | 6 | +0.117 | [+0.034, +0.225] | 4 / 0 | yes |
| retrieve − truncate | best | 6 | +0.059 | [-0.504, +0.682] | 1 / 1 | no, CI includes 0 |
| retrieve − truncate | auc_rounds | 6 | -0.056 | [-0.252, +0.085] | 1 / 1 | no, CI includes 0 |
| retrieve − truncate | retry_rate | 6 | +0.058 | [-0.107, +0.210] | 3 / 1 | no, CI includes 0 |
| rlc − truncate | best | 6 | +0.000 | [+0.000, +0.000] | 1 / 0 | no, CI includes 0 |
| rlc − truncate | auc_rounds | 6 | +0.000 | [+0.000, +0.000] | 1 / 0 | no, CI includes 0 |
| rlc − truncate | retry_rate | 6 | -0.060 | [-0.179, +0.000] | 0 / 1 | no, CI includes 0 |

## Best-so-far by round (mean over runs)

| round | kv:full | retrieve | rlc | truncate |
| --- | --- | --- | --- | --- |
| 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.498 | 0.498 | 0.498 | 0.498 |
| 4 | 0.498 | 0.498 | 0.498 | 0.498 |
| 5 | 0.498 | 0.498 | 0.665 | 0.665 |
| 6 | 0.498 | 0.498 | 0.666 | 0.666 |
| 7 | 0.498 | 0.498 | 0.666 | 0.666 |
| 8 | 0.498 | 0.725 | 0.666 | 0.666 |
