# Mk33 bench results

Model: `Qwen/Qwen2.5-3B-Instruct`. Runs: 30 (0 failed and excluded).
Scores are normalized per environment: baseline = 0, reference = 1, above 1 beats the reference.

Budget per run: 8 rounds, 40 tool calls, 25 min. Context window 8000 tokens, compaction at 0.75, keep last 2 rounds.

## By context mode (mean ± sd over runs)

| metric | kv:full | kv:h2o | retrieve | rlc | truncate |
| --- | --- | --- | --- | --- | --- |
| best score | 0.168 ± 0.411 | 0.209 ± 0.512 | 0.168 ± 0.411 | 0.247 ± 0.604 | 0.168 ± 0.411 |
| best-so-far AUC | 0.168 ± 0.411 | 0.199 ± 0.486 | 0.168 ± 0.411 | 0.227 ± 0.556 | 0.168 ± 0.411 |
| productive rounds T* | 1.000 ± 1.265 | 1.333 ± 2.422 | 0.500 ± 0.837 | 0.833 ± 1.329 | 0.500 ± 0.837 |
| mean gain | 0.168 ± 0.411 | 0.104 ± 0.256 | 0.168 ± 0.411 | 0.123 ± 0.302 | 0.168 ± 0.411 |
| recovery rate | 0.000 ± 0.000 | 0.062 ± 0.125 | 0.000 ± 0.000 | 0.200 ± 0.447 | 0.000 ± 0.000 |
| wasted-turn share | 0.324 ± 0.322 | 0.466 ± 0.267 | 0.254 ± 0.255 | 0.299 ± 0.232 | 0.366 ± 0.308 |
| re-try rate (lower is better) | 0.771 ± 0.436 | 0.663 ± 0.410 | 0.771 ± 0.436 | 0.714 ± 0.416 | 0.800 ± 0.400 |
| compactions | 0.000 ± 0.000 | 0.000 ± 0.000 | 3.167 ± 2.994 | 2.500 ± 2.739 | 9.500 ± 6.656 |
| peak KV cache (tokens) | 12836.667 ± 6777.099 | 7712.333 ± 800.168 | 7410.833 ± 1952.867 | 7589.167 ± 1548.782 | 7094.000 ± 209.104 |
| peak KV cache (MB) | 451.289 ± 238.257 | 271.137 ± 28.131 | 260.537 ± 68.655 | 266.807 ± 54.449 | 249.398 ± 7.351 |
| evicted token share | 0.000 ± 0.000 | 0.123 ± 0.181 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| prompt tokens per call | 6006.398 ± 2538.310 | 6810.660 ± 3969.899 | 4233.130 ± 903.817 | 4505.574 ± 1493.323 | 4873.553 ± 777.759 |
| prefill s per call | 3.173 ± 1.798 | 7.293 ± 6.090 | 1.908 ± 0.464 | 2.077 ± 0.795 | 2.272 ± 0.355 |
| decode tokens/s | 8.332 ± 2.689 | 13.149 ± 1.098 | 10.158 ± 1.507 | 10.150 ± 1.498 | 9.674 ± 0.820 |
| time in compression | 0.000 ± 0.000 | 0.043 ± 0.072 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| beat reference | 17% | 17% | 17% | 17% | 17% |
| audit passed | 100% | 100% | 100% | 100% | 100% |

## Best score by environment

| environment | kv:full | kv:h2o | retrieve | rlc | truncate |
| --- | --- | --- | --- | --- | --- |
| bench-knapsack | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| bench-regression | 0.335 ± 0.581 | 0.418 ± 0.724 | 0.335 ± 0.581 | 0.493 ± 0.854 | 0.335 ± 0.581 |

## Paired comparisons

| comparison | metric | pairs | mean diff | 95% CI | wins (a / b) | real difference? |
| --- | --- | --- | --- | --- | --- | --- |
| kv:full − kv:h2o | best | 6 | -0.041 | [-0.124, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − kv:h2o | auc_rounds | 6 | -0.031 | [-0.093, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − kv:h2o | retry_rate | 5 | +0.109 | [+0.000, +0.223] | 2 / 0 | no, CI includes 0 |
| kv:full − retrieve | best | 6 | +0.000 | [+0.000, +0.000] | 0 / 0 | no, CI includes 0 |
| kv:full − retrieve | auc_rounds | 6 | +0.000 | [+0.000, +0.000] | 0 / 0 | no, CI includes 0 |
| kv:full − retrieve | retry_rate | 5 | +0.000 | [+0.000, +0.000] | 0 / 0 | no, CI includes 0 |
| kv:full − rlc | best | 6 | -0.079 | [-0.237, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − rlc | auc_rounds | 6 | -0.059 | [-0.178, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − rlc | retry_rate | 5 | +0.057 | [+0.000, +0.114] | 2 / 0 | no, CI includes 0 |
| kv:full − truncate | best | 6 | +0.000 | [+0.000, +0.000] | 0 / 0 | no, CI includes 0 |
| kv:full − truncate | auc_rounds | 6 | +0.000 | [+0.000, +0.000] | 0 / 0 | no, CI includes 0 |
| kv:full − truncate | retry_rate | 5 | +0.011 | [+0.000, +0.034] | 1 / 0 | no, CI includes 0 |
| kv:h2o − retrieve | best | 6 | +0.041 | [+0.000, +0.124] | 1 / 0 | no, CI includes 0 |
| kv:h2o − retrieve | auc_rounds | 6 | +0.031 | [+0.000, +0.093] | 1 / 0 | no, CI includes 0 |
| kv:h2o − retrieve | retry_rate | 5 | -0.109 | [-0.223, +0.000] | 0 / 2 | no, CI includes 0 |
| kv:h2o − rlc | best | 6 | -0.038 | [-0.113, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:h2o − rlc | auc_rounds | 6 | -0.028 | [-0.085, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:h2o − rlc | retry_rate | 5 | -0.051 | [-0.183, +0.063] | 1 / 2 | no, CI includes 0 |
| kv:h2o − truncate | best | 6 | +0.041 | [+0.000, +0.124] | 1 / 0 | no, CI includes 0 |
| kv:h2o − truncate | auc_rounds | 6 | +0.031 | [+0.000, +0.093] | 1 / 0 | no, CI includes 0 |
| kv:h2o − truncate | retry_rate | 5 | -0.097 | [-0.211, +0.000] | 0 / 2 | no, CI includes 0 |
| retrieve − rlc | best | 6 | -0.079 | [-0.237, +0.000] | 0 / 1 | no, CI includes 0 |
| retrieve − rlc | auc_rounds | 6 | -0.059 | [-0.178, +0.000] | 0 / 1 | no, CI includes 0 |
| retrieve − rlc | retry_rate | 5 | +0.057 | [+0.000, +0.114] | 2 / 0 | no, CI includes 0 |
| retrieve − truncate | best | 6 | +0.000 | [+0.000, +0.000] | 0 / 0 | no, CI includes 0 |
| retrieve − truncate | auc_rounds | 6 | +0.000 | [+0.000, +0.000] | 0 / 0 | no, CI includes 0 |
| retrieve − truncate | retry_rate | 5 | +0.011 | [+0.000, +0.034] | 1 / 0 | no, CI includes 0 |
| rlc − truncate | best | 6 | +0.079 | [+0.000, +0.237] | 1 / 0 | no, CI includes 0 |
| rlc − truncate | auc_rounds | 6 | +0.059 | [+0.000, +0.178] | 1 / 0 | no, CI includes 0 |
| rlc − truncate | retry_rate | 5 | -0.046 | [-0.103, +0.000] | 0 / 2 | no, CI includes 0 |

## Best-so-far by round (mean over runs)

| round | kv:full | kv:h2o | retrieve | rlc | truncate |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.168 | 0.168 | 0.168 | 0.168 | 0.168 |
| 2 | 0.168 | 0.168 | 0.168 | 0.168 | 0.168 |
| 3 | 0.168 | 0.209 | 0.168 | 0.247 | 0.168 |
| 4 | 0.168 | 0.209 | 0.168 | 0.247 | 0.168 |
| 5 | 0.168 | 0.209 | 0.168 | 0.247 | 0.168 |
| 6 | 0.168 | 0.209 | 0.168 | 0.247 | 0.168 |
| 7 | 0.168 | 0.209 | 0.168 | 0.247 | 0.168 |
| 8 | 0.168 | 0.209 | 0.168 | 0.247 | 0.168 |
