# Memory for long-horizon agents: text-level vs KV-level forgetting

A long improvement run outgrows any context window. Something has to be forgotten,
and there are two places to do it:

- **In the text**, before the model sees it: drop old turns (`truncate`), summarize them
  into a ledger (`rlc`), or move them into a store and retrieve the relevant ones back
  (`retrieve`, `rlc+retrieve`).
- **In the model**, after it has read them: keep the full text history and compress the
  model's KV cache to a fixed budget, choosing which tokens' keys and values to keep
  (`kv:<policy>`).

This study runs both on the same bench, the same model and the same budgets, and asks:
at a fixed memory budget, which kind of forgetting lets an agent keep improving?

## Modes

| Mode | What the model sees | What is forgotten | Needs |
| --- | --- | --- | --- |
| `truncate` | recent turns | oldest turns, silently | any model |
| `rlc` | ledger (one line per round) + recent rounds | details of old rounds | any model |
| `retrieve` | recent turns + top-k old turns by relevance | old turns not retrieved this round | any model |
| `rlc+retrieve` | ledger + recent rounds + top-k old turns | details not retrieved | any model |
| `kv:full` | everything (up to the model's limit) | nothing | local model |
| `kv:streaming` | everything, but the cache keeps sinks + most recent tokens | the middle | local model |
| `kv:keynorm` | everything; cache keeps low-norm keys + sinks + recent | high-norm keys | local model |
| `kv:h2o` | everything; cache keeps most-attended tokens + sinks + recent | rarely attended tokens | local model, eager attention |
| `kv:random` | everything; cache keeps random tokens + sinks + recent | random tokens | local model |

`kv:random` is the control: a policy that does not beat it has not shown it knows which
tokens matter. `kv:full` is the ceiling at unlimited memory.

**Equal-memory comparison.** A text-level mode with `--window W` holds about W tokens in
the model's cache. A kv mode with `--kv-budget B` holds about B. Running both with
`--window 8000 --kv-budget 8000` (the default sets the budget to the window) compares
them at the same cache size, which is the point of the study.

## Retrieval memory (`src/mk33/scaffold/memory.py`)

- **What is stored.** Each turn that leaves the context becomes one item: the agent's
  reasoning, its tool call, and the head and tail of the tool output (where commands
  echo and where scores and tracebacks end up), capped at `--retrieve-item-chars`.
- **Retriever.** BM25 over identifiers and numbers, using only the standard library, so
  CPU runs need nothing extra. `--retriever embed --embed-model <name>` switches to
  dense retrieval through any OpenAI-compatible `/embeddings` endpoint.
- **Query.** The agent's last two turns, so retrieval follows what it is working on now.
- **Budget.** At most `--retrieve-k` items and `--retrieve-budget` tokens. That budget is
  reserved when compacting, so the context still fits the window.
- **Refresh.** The retrieved block sits in the prefix, which Mk33 freezes between events
  so the context only grows by appending. Changing it is therefore an event like a
  compaction: it starts a new training window and invalidates the KV cache for
  everything after the system prompt. With `--retrieve-refresh round` (default) this
  happens at every new round; with `compaction` only when compacting. The trade-off
  (fresher retrieval vs fewer cache invalidations) is measurable with the compaction
  count in the report.
- **Train/test consistency.** Retrieval runs inside `ContextBuilder`, which the training
  replay also uses, so `mk33 build-sft` reproduces the retrieved contexts exactly. The
  tests check this turn by turn, as they already did for `rlc`. Ablation config
  `A8_rlc_retrieve` trains with it.

## KV-cache compression (`src/mk33/kv/`)

- **Where it runs.** `HFChatClient` is a drop-in `ChatClient` for a local Hugging Face
  model. It formats messages and tool schemas with the model's own chat template and
  parses Hermes/Qwen-style `<tool_call>` blocks. With `--backend hf`, every mode in a
  grid (text-level ones too) runs on the same local model.
- **Bounded cache.** The prompt is prefilled in chunks (`--kv-chunk`). After each chunk,
  and every 64 tokens while decoding, the cache is compressed back to the budget. Peak
  cache size is about `budget + chunk` tokens per layer, whatever the prompt length.
- **Positions.** Cached keys already carry their rotary embedding, so evicting rows does
  not disturb them; new tokens get their true positions via `position_ids` while
  `cache_position` indexes the shorter cache. A test checks this exactly: after
  eviction, the next token's logits equal a full forward pass in which that token is
  masked from the evicted ones.
- **Per head.** Each KV head keeps its own set of tokens (same count, different indices).
- **Policies** (`policies.py`): sinks and recent tokens are always kept; the policy fills
  the middle. `keynorm` needs no attention scores; `h2o` accumulates attention
  probabilities over all queries so far, which needs eager attention (slower prefill,
  more memory). `kv_quality.py` and the report show the cost.
- **Triton kernels** (`triton_ops.py`): one fused kernel gathers the kept rows of K and V
  in a single launch with arbitrary input strides; another computes key norms with
  float32 accumulation without a float32 copy of the cache. Both are checked against
  the PyTorch reference (bit-exact for the gather) on GPU or in Triton's CPU
  interpreter. `scripts/mk33/bench_kv_kernels.py` measures time and bandwidth.
- **Simplification.** Each model call prefills its prompt from scratch; there is no
  cache reuse across calls. Prefill time is therefore reported per call, and compression
  is applied as the prompt is read. Cross-call prefix caching under compression is a
  natural next step.

## Running it

On a machine with an NVIDIA GPU (Colab: Runtime > Change runtime type > GPU).

```bash
git clone https://github.com/dulguuntsatsralt-cell/mk33 && cd mk33
pip install -e '.[mk33-kv,dev]'
python -m pytest tests/mk33 -q          # on a GPU this includes the Triton kernel tests

# 1. Kernels: PyTorch vs Triton, time and bandwidth
python scripts/mk33/bench_kv_kernels.py --out bench/kv_kernels.json

# 2. Does the model's tool calling work with HFChatClient?
python scripts/mk33/kv_quality.py --model Qwen/Qwen2.5-7B-Instruct --smoke

# 3. Perplexity vs budget for every policy (minutes, no agent)
python scripts/mk33/kv_quality.py --model Qwen/Qwen2.5-7B-Instruct --context 8192 \
  --budgets 512,1024,2048,4096 --out bench/kv_quality.json

# 4. The agent grid at an equal memory budget (start small to time it)
mk33 bench-prepare --out bench/envs
mk33 bench-run --backend hf --model Qwen/Qwen2.5-7B-Instruct --window 8000 \
  --modes truncate,rlc,retrieve,rlc+retrieve,kv:full,kv:streaming,kv:keynorm,kv:random \
  --tasks knapsack --repeats 1 --out bench/memory
mk33 bench-report --out bench/memory
```

Text-level modes also run against any API model without a GPU:

```bash
mk33 bench-run --model <name> --base-url <url> --modes rlc,truncate,retrieve,rlc+retrieve --out bench/retrieval
```

**Choosing a model and GPU.** Qwen2.5-7B-Instruct in bfloat16 needs about 16 GB plus
cache: an L4 (24 GB) or A100 works. On a T4 (16 GB, no fast bfloat16) use
Qwen2.5-3B-Instruct with `--dtype float16`, keeping in mind that models around 3B often
describe edits instead of calling tools (see BENCH.md). `kv:h2o` uses eager attention;
lower `--kv-chunk` if prefill runs out of memory.

## What to report

1. `bench/memory/report.md`: per mode, best normalized score, best-so-far AUC, peak KV
   cache, evicted share, prefill time; and paired bootstrap intervals for every pair of
   modes. Only call a difference real when its interval excludes zero.
2. `kv_quality.json`: perplexity gap to the full cache as a function of budget, per policy.
3. `kv_kernels.json`: Triton vs PyTorch time and achieved bandwidth on your GPU.

## First results (Tesla T4, Qwen2.5-3B-Instruct, October 2026)

One free Colab T4 session. Raw numbers: [results/t4-qwen2.5-3b/](results/t4-qwen2.5-3b/).
All tests passed on the GPU, including the Triton kernel tests on real hardware.

### Kernels

float16, batch 1, 8 KV heads, head dim 128, keeping 25% of rows. The T4's peak memory
bandwidth is 320 GB/s.

| tokens | gather: PyTorch / Triton (ms) | speedup | Triton GB/s | key norm: PyTorch / Triton (ms) | speedup | Triton GB/s |
| --- | --- | --- | --- | --- | --- | --- |
| 2,048 | 0.054 / 0.128 | 0.43x | 33 | 0.145 / 0.129 | 1.1x | 33 |
| 8,192 | 0.180 / 0.122 | 1.5x | 139 | 0.512 / 0.075 | 6.9x | 228 |
| 32,768 | 0.582 / 0.304 | 1.9x | 223 | 1.459 / 0.267 | 5.5x | 256 |
| 131,072 | 1.886 / 1.195 | 1.6x | 226 | 5.646 / 1.050 | 5.4x | 260 |

- The key-norm kernel reaches about 80% of peak bandwidth and is 5 to 7 times faster
  than `k.float().norm(-1)`, which first writes a float32 copy of the keys.
- The fused gather is 1.5 to 1.9 times faster than two `torch.gather` calls from 8K
  tokens up, at about 70% of peak.
- At 2K tokens both kernels are dominated by a fixed launch cost of about 0.12 ms, and
  the gather is slower than PyTorch. For short caches the PyTorch path is the better
  default; `backend="auto"` does not yet switch on size.

### Perplexity vs cache budget

Next-token loss (NLL, nats) on 1,024 tokens of the paper and scaffold code, given the
8,192 tokens before them. Sinks 4, recent 256. Full cache: NLL 1.641, 324 MB peak.
Lower "vs full" is better.

| budget | random | streaming | keynorm | h2o |
| --- | --- | --- | --- | --- |
| 512 | +0.596 | +0.615 | +0.651 | +0.626 |
| 1,024 | +0.475 | +0.525 | +0.595 | +0.420 |
| 2,048 | +0.309 | +0.334 | +0.472 | +0.228 |
| 4,096 | +0.127 | +0.268 | +0.224 | +0.106 |

- **Random eviction is a strong control.** It beats streaming and key-norm at every
  budget. Keeping a spread of tokens from the whole document matters more here than
  keeping only the recent ones.
- **Only H2O beats random**, and only from 1,024 tokens up: by 0.055, 0.081 and 0.021
  nats. At 4,096 tokens it keeps the loss within 0.11 nats of the full cache with
  about half the memory.
- **Key-norm is worse than random at every budget on this model.** The low-norm
  heuristic was reported on other model families; Qwen2.5's attention uses biased
  query/key projections, which may change how key norm relates to attention. This
  needs checking on a second model family before drawing a conclusion.
- H2O ran with 128-token chunks (eager attention stores every layer's scores) and the
  others with 512, so its lower peak memory in the raw files comes from the chunk size,
  not the policy. Its time includes eager attention.

**Limits.** One model, one document, one continuation, one random seed, no confidence
intervals. Differences of a few hundredths of a nat (for example H2O vs random at
4,096) may not survive repeats. Next: several documents and seeds with bootstrap
intervals, and a second model family (Llama 3.2 3B).

## Status

Implemented and tested on CPU and on a T4 GPU: retrieval modes, train/test consistency
with retrieval, every KV policy, positional correctness after eviction, the bench and
report wiring, and both Triton kernels. Kernel timings and perplexity results above.
The agent bench has not yet produced a usable comparison. A first pilot (Qwen2.5-3B,
knapsack, `rlc`, 8 rounds, about 6 minutes on the T4) shows the pipeline works end to
end: 10 of 11 replies parsed as tool calls and every submission was scored. But the model
never called `edit_file`. It described a greedy algorithm, then dynamic programming,
then a genetic algorithm in its submit hypotheses, and resubmitted the unchanged baseline
each time (the ledger records "no changes" for all 8 rounds). This is the failure noted
in BENCH.md for models around 3B. The context also peaked at 5.1K tokens, below the 8K
window, so no compaction or eviction happened and the run could not have separated the
memory strategies anyway. The agent comparison needs a model that edits code reliably
(7B or larger) and runs long enough to outgrow the window.
