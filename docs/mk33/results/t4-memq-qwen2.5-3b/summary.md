# RAG vs KV-cache eviction: Qwen/Qwen2.5-3B-Instruct

8 windows. Loss increase over the full cache (nats; lower is better), mean and bootstrap 95% interval over windows. 'vs random' is the paired difference against kv:random (negative = better than random); 'yes' when its interval excludes zero.

| method | budget | vs full | 95% CI | vs random | 95% CI | better than random? |
| --- | --- | --- | --- | --- | --- | --- |
| kv:h2o | 1024 | +0.203 | [+0.127, +0.291] | -0.056 | [-0.120, +0.004] | unclear |
| kv:keynorm | 1024 | +0.304 | [+0.178, +0.445] | +0.044 | [+0.009, +0.080] | no, worse |
| kv:random | 1024 | +0.259 | [+0.139, +0.398] | (control) | |  |
| kv:streaming | 1024 | +0.284 | [+0.157, +0.424] | +0.025 | [+0.010, +0.039] | no, worse |
| text:recent | 1024 | +0.288 | [+0.158, +0.432] | +0.028 | [+0.012, +0.044] | no, worse |
| text:retrieve | 1024 | +0.288 | [+0.175, +0.406] | +0.028 | [-0.006, +0.068] | unclear |
| kv:h2o | 2048 | +0.120 | [+0.078, +0.168] | -0.072 | [-0.143, -0.006] | yes |
| kv:keynorm | 2048 | +0.230 | [+0.124, +0.349] | +0.039 | [+0.005, +0.076] | no, worse |
| kv:random | 2048 | +0.191 | [+0.096, +0.303] | (control) | |  |
| kv:streaming | 2048 | +0.210 | [+0.100, +0.336] | +0.019 | [-0.004, +0.042] | unclear |
| text:recent | 2048 | +0.206 | [+0.096, +0.334] | +0.015 | [-0.007, +0.038] | unclear |
| text:retrieve | 2048 | +0.222 | [+0.131, +0.329] | +0.031 | [+0.014, +0.049] | no, worse |
| kv:h2o | 4096 | +0.049 | [+0.031, +0.072] | -0.030 | [-0.055, -0.007] | yes |
| kv:keynorm | 4096 | +0.104 | [+0.052, +0.162] | +0.025 | [+0.007, +0.045] | no, worse |
| kv:random | 4096 | +0.079 | [+0.041, +0.124] | (control) | |  |
| kv:streaming | 4096 | +0.141 | [+0.049, +0.246] | +0.062 | [-0.001, +0.127] | unclear |
| text:recent | 4096 | +0.133 | [+0.043, +0.230] | +0.054 | [-0.005, +0.113] | unclear |
| text:retrieve | 4096 | +0.125 | [+0.053, +0.220] | +0.046 | [+0.001, +0.103] | no, worse |
