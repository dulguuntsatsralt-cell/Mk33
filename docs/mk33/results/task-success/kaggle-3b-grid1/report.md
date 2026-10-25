# Mk33 bench results

Model: `Qwen/Qwen2.5-3B-Instruct`. Runs: 30 (0 failed and excluded).
Scores are normalized per environment: baseline = 0, reference = 1, above 1 beats the reference.

Budget per run: 8 rounds, 40 tool calls, 25 min. Context window 8000 tokens, compaction at 0.75, keep last 2 rounds.

## By context mode (mean ± sd over runs)

| metric | kv:full | kv:h2o | retrieve | rlc | truncate |
| --- | --- | --- | --- | --- | --- |
| best score | -0.098 ± 1.183 | -0.150 ± 1.099 | -0.112 ± 1.160 | -0.112 ± 1.160 | -0.191 ± 1.040 |
| best-so-far AUC | 0.225 ± 0.552 | 0.197 ± 0.484 | 0.217 ± 0.531 | 0.227 ± 0.556 | 0.168 ± 0.411 |
| productive rounds T* | 2.167 ± 3.125 | 0.833 ± 1.329 | 1.500 ± 2.811 | 0.833 ± 1.329 | 0.500 ± 0.837 |
| mean gain | 0.130 ± 0.319 | 0.104 ± 0.254 | 0.123 ± 0.302 | 0.123 ± 0.302 | 0.168 ± 0.411 |
| recovery rate | 0.250 ± 0.500 | 0.125 ± 0.250 | 0.200 ± 0.447 | 0.250 ± 0.500 | 0.000 ± 0.000 |
| wasted-turn share | 0.297 ± 0.298 | 0.508 ± 0.219 | 0.340 ± 0.278 | 0.249 ± 0.238 | 0.259 ± 0.308 |
| re-try rate (lower is better) | 0.714 ± 0.440 | 0.676 ± 0.409 | 0.705 ± 0.417 | 0.743 ± 0.433 | 0.771 ± 0.436 |
| compactions | 0.000 ± 0.000 | 0.000 ± 0.000 | 2.833 ± 2.787 | 3.333 ± 4.590 | 7.000 ± 9.077 |
| peak KV cache (tokens) | 14759.500 ± 11716.599 | 7183.167 ± 1318.067 | 11746.333 ± 12500.731 | 11987.667 ± 12438.405 | 11710.500 ± 12520.352 |
| peak KV cache (MB) | 518.889 ± 411.912 | 252.533 ± 46.338 | 412.957 ± 439.479 | 421.441 ± 437.288 | 411.697 ± 440.169 |
| evicted token share | 0.000 ± 0.000 | 0.210 ± 0.337 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| prompt tokens per call | 7042.386 ± 5171.472 | 14049.176 ± 21532.957 | 6365.575 ± 5292.414 | 6233.935 ± 5429.655 | 6239.397 ± 5377.852 |
| prefill s per call | 5.109 ± 6.270 | 20.244 ± 36.758 | 4.610 ± 6.440 | 4.565 ± 6.490 | 4.554 ± 6.481 |
| decode tokens/s | 8.051 ± 3.334 | 12.979 ± 1.094 | 9.088 ± 3.360 | 9.180 ± 3.579 | 9.004 ± 3.390 |
| time in compression | 0.000 ± 0.000 | 0.096 ± 0.181 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| beat reference | 17% | 17% | 17% | 17% | 17% |
| audit passed | 100% | 100% | 100% | 100% | 100% |

## Best score by environment

| environment | kv:full | kv:h2o | retrieve | rlc | truncate |
| --- | --- | --- | --- | --- | --- |
| bench-knapsack | -0.716 ± 1.241 | -0.716 ± 1.241 | -0.716 ± 1.241 | -0.716 ± 1.241 | -0.716 ± 1.241 |
| bench-regression | 0.520 ± 0.901 | 0.416 ± 0.721 | 0.493 ± 0.854 | 0.493 ± 0.854 | 0.335 ± 0.581 |

## Paired comparisons

| comparison | metric | pairs | mean diff | 95% CI | wins (a / b) | real difference? |
| --- | --- | --- | --- | --- | --- | --- |
| kv:full − kv:h2o | best | 6 | +0.052 | [+0.000, +0.156] | 1 / 0 | no, CI includes 0 |
| kv:full − kv:h2o | auc_rounds | 6 | +0.028 | [+0.000, +0.084] | 1 / 0 | no, CI includes 0 |
| kv:full − kv:h2o | retry_rate | 5 | +0.038 | [-0.086, +0.200] | 1 / 1 | no, CI includes 0 |
| kv:full − retrieve | best | 6 | +0.014 | [+0.000, +0.041] | 1 / 0 | no, CI includes 0 |
| kv:full − retrieve | auc_rounds | 6 | +0.008 | [+0.000, +0.025] | 1 / 0 | no, CI includes 0 |
| kv:full − retrieve | retry_rate | 5 | +0.010 | [-0.057, +0.086] | 1 / 1 | no, CI includes 0 |
| kv:full − rlc | best | 6 | +0.014 | [+0.000, +0.041] | 1 / 0 | no, CI includes 0 |
| kv:full − rlc | auc_rounds | 6 | -0.001 | [-0.004, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − rlc | retry_rate | 5 | -0.029 | [-0.086, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:full − truncate | best | 6 | +0.093 | [+0.000, +0.278] | 1 / 0 | no, CI includes 0 |
| kv:full − truncate | auc_rounds | 6 | +0.058 | [+0.000, +0.173] | 1 / 0 | no, CI includes 0 |
| kv:full − truncate | retry_rate | 5 | -0.057 | [-0.171, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:h2o − retrieve | best | 6 | -0.039 | [-0.116, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:h2o − retrieve | auc_rounds | 6 | -0.020 | [-0.059, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:h2o − retrieve | retry_rate | 5 | -0.029 | [-0.190, +0.086] | 2 / 1 | no, CI includes 0 |
| kv:h2o − rlc | best | 6 | -0.039 | [-0.116, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:h2o − rlc | auc_rounds | 6 | -0.029 | [-0.088, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:h2o − rlc | retry_rate | 5 | -0.067 | [-0.200, +0.000] | 0 / 1 | no, CI includes 0 |
| kv:h2o − truncate | best | 6 | +0.040 | [+0.000, +0.121] | 1 / 0 | no, CI includes 0 |
| kv:h2o − truncate | auc_rounds | 6 | +0.030 | [+0.000, +0.089] | 1 / 0 | no, CI includes 0 |
| kv:h2o − truncate | retry_rate | 5 | -0.095 | [-0.229, +0.000] | 0 / 2 | no, CI includes 0 |
| retrieve − rlc | best | 6 | -0.000 | [-0.000, +0.000] | 0 / 1 | no, CI includes 0 |
| retrieve − rlc | auc_rounds | 6 | -0.010 | [-0.030, +0.000] | 0 / 1 | no, CI includes 0 |
| retrieve − rlc | retry_rate | 5 | -0.038 | [-0.095, +0.000] | 0 / 2 | no, CI includes 0 |
| retrieve − truncate | best | 6 | +0.079 | [+0.000, +0.237] | 1 / 0 | no, CI includes 0 |
| retrieve − truncate | auc_rounds | 6 | +0.049 | [+0.000, +0.148] | 1 / 0 | no, CI includes 0 |
| retrieve − truncate | retry_rate | 5 | -0.067 | [-0.143, +0.000] | 0 / 2 | no, CI includes 0 |
| rlc − truncate | best | 6 | +0.079 | [+0.000, +0.237] | 1 / 0 | no, CI includes 0 |
| rlc − truncate | auc_rounds | 6 | +0.059 | [+0.000, +0.178] | 1 / 0 | no, CI includes 0 |
| rlc − truncate | retry_rate | 5 | -0.029 | [-0.086, +0.000] | 0 / 1 | no, CI includes 0 |

## Best-so-far by round (mean over runs)

| round | kv:full | kv:h2o | retrieve | rlc | truncate |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.168 | 0.168 | 0.168 | 0.168 | 0.168 |
| 2 | 0.168 | 0.168 | 0.168 | 0.168 | 0.168 |
| 3 | 0.168 | 0.207 | 0.168 | 0.247 | 0.168 |
| 4 | 0.260 | 0.207 | 0.247 | 0.247 | 0.168 |
| 5 | 0.260 | 0.207 | 0.247 | 0.247 | 0.168 |
| 6 | 0.260 | 0.207 | 0.247 | 0.247 | 0.168 |
| 7 | 0.260 | 0.208 | 0.247 | 0.247 | 0.168 |
| 8 | 0.260 | 0.208 | 0.247 | 0.247 | 0.168 |
