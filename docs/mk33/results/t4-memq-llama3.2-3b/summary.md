# RAG vs KV-cache eviction: unsloth/Llama-3.2-3B-Instruct

8 windows. Loss increase over the full cache (nats; lower is better), mean and bootstrap 95% interval over windows. 'vs random' is the paired difference against kv:random (negative = better than random); 'yes' when its interval excludes zero.

| method | budget | vs full | 95% CI | vs random | 95% CI | better than random? |
| --- | --- | --- | --- | --- | --- | --- |
| kv:h2o | 1024 | +0.265 | [+0.195, +0.323] | -0.191 | [-0.278, -0.096] | yes |
| kv:keynorm | 1024 | +0.376 | [+0.270, +0.477] | -0.079 | [-0.135, -0.012] | yes |
| kv:random | 1024 | +0.456 | [+0.329, +0.578] | (control) | |  |
| kv:streaming | 1024 | +0.470 | [+0.325, +0.608] | +0.014 | [-0.019, +0.047] | unclear |
| text:recent | 1024 | +0.291 | [+0.157, +0.428] | -0.164 | [-0.197, -0.133] | yes |
| text:retrieve | 1024 | +0.327 | [+0.213, +0.447] | -0.128 | [-0.180, -0.076] | yes |
| kv:h2o | 2048 | +0.156 | [+0.116, +0.189] | -0.193 | [-0.287, -0.098] | yes |
| kv:keynorm | 2048 | +0.262 | [+0.184, +0.340] | -0.087 | [-0.148, -0.013] | yes |
| kv:random | 2048 | +0.349 | [+0.247, +0.457] | (control) | |  |
| kv:streaming | 2048 | +0.363 | [+0.252, +0.483] | +0.015 | [-0.007, +0.037] | unclear |
| text:recent | 2048 | +0.196 | [+0.088, +0.325] | -0.152 | [-0.190, -0.120] | yes |
| text:retrieve | 2048 | +0.242 | [+0.150, +0.330] | -0.106 | [-0.164, -0.052] | yes |
| kv:h2o | 4096 | +0.068 | [+0.047, +0.086] | -0.076 | [-0.144, -0.014] | yes |
| kv:keynorm | 4096 | +0.103 | [+0.064, +0.142] | -0.041 | [-0.084, +0.002] | unclear |
| kv:random | 4096 | +0.144 | [+0.091, +0.209] | (control) | |  |
| kv:streaming | 4096 | +0.262 | [+0.181, +0.365] | +0.118 | [+0.076, +0.164] | no, worse |
| text:recent | 4096 | +0.131 | [+0.043, +0.243] | -0.013 | [-0.060, +0.043] | unclear |
| text:retrieve | 4096 | +0.122 | [+0.076, +0.166] | -0.022 | [-0.092, +0.039] | unclear |
