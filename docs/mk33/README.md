# Mk33 design notes

Mk33 is a fully specified version of the AREX-2 training recipe
([Qian et al., 2026](https://arxiv.org/abs/2609.38288)). Wherever the AREX-2 paper leaves a
component unspecified, this code makes it explicit, gives it a config key, and gives it an
ablation. This page maps each part of the paper to the code.

## Paper to code

| Paper section | What it specifies | Code | Config key |
| --- | --- | --- | --- |
| 3.1 Sources | Repo candidate rules, teacher prompt, scorer layout | `mk33/envs/synthesis.py` | none |
| 3.1 Normalized score | `s~ = (s - b) / (s_ref - b)`, sign flipped if lower is better | `mk33/envs/normalize.py` | none |
| 3.1 Headroom filter | `s_ref - b >= max(3 sigma, 0.05 abs(s_ref))`; judge: `b <= 0.6`, `s_ref >= 0.95` | `mk33/envs/filter.py` | `filter` |
| 3.1 Decontamination | Dataset URL, name, or hash; problem ID; 13-gram overlap >= 0.5 | `mk33/envs/decontam.py` | `decontam` |
| 3.1 Scorer isolation | Separate scorer dir, SHA-256 check, read-only and no-network docker mode | `mk33/envs/scorer.py` | none |
| 3.1 Process audit | Canary tests, rule checks, optional LLM auditor | `mk33/envs/audit.py` | `audit` |
| 3.2 Tool set | bash, read/edit, run_job/wait_job, web, submit, answer | `mk33/scaffold/tools.py` | `tools` |
| 3.2 Round ledger | One entry per submit; merge per strategy family when over cap | `mk33/scaffold/ledger.py` | `rlc.ledger_cap` |
| 3.2 RLC | Compact rounds older than K once over `theta * window` | `mk33/scaffold/rlc.py` | `rlc` |
| 3.2 Same contexts at train and test | One `ContextBuilder` for the agent and the replay | `mk33/scaffold/context.py`, `mk33/train/windows.py` | `rlc_train` |
| 3.3 PALM rules R, N, P | Redundant, null, and polling turns | `mk33/palm/rules.py` | `palm.window`, `palm.jaccard` |
| 3.3 PALM S+ | Surviving edits, improving submits, one-hop evidence, recovery | `mk33/palm/provenance.py` | `palm.m_rounds`, `palm.evidence_window`, `palm.ngram` |
| 3.3 Weights and loss | `w in {0, 1, lambda}`; weighted CE normalized by `sum w abs(a)` | `mk33/palm/weights.py`, `mk33/train/sft.py` | `palm.lam`, `palm.mode` |
| 3.3 Checking the rules | Precision, recall, and Cohen's kappa against human labels | `mk33/palm/validate.py` | none |
| 3.4 Acceptance | tau_ml 0.9, tau_judge 0.95, at least 3 rounds, iteration, audit, at most 2 per env | `mk33/selection/__init__.py` | `selection` |
| 3.4 Training | Full hyperparameter set | `mk33/train/sft.py`, `configs/mk33/default.yaml` | `sft` |
| 4 Process metrics | T*, mean gain, AUC, recovery rate, wasted-turn share | `mk33/metrics/__init__.py` | none |
| 4 Ablations A1 to A7 | One config per variant | `configs/mk33/ablations/` | `inherit:` |

## Design decisions not stated in the paper

The paper draft does not cover these choices. They are recorded here so they can be
reviewed.

1. **Frozen prefix between compactions.** The pinned block (task, skills, current best) and
   the ledger are rendered only at the start of a run and at each compaction. Between
   compactions the context only grows. This keeps the KV cache valid, and it means each
   segment between two compactions is exactly one training window. New ledger lines still
   reach the model right away, because they are appended to the `submit` observation.
2. **Deterministic tool-call IDs** (`call_<turn index>`). The agent and the training replay
   produce identical messages. `tests/mk33/test_end_to_end.py` checks that every
   context the agent sent is a prefix of its training window.
3. **R is not applied to status checks or to turns that changed the workspace.** Repeated
   status checks are handled by P, because a repeated `tail` can return new lines. A turn
   that changed the workspace is never a no-op.
4. **Hunk survival is block containment.** An added block survives if it appears line for
   line (ignoring blank lines and trailing spaces) in a later improving submission's
   snapshot. A deletion-only hunk survives if the removed block is still absent. This is an
   approximation of `git blame` that needs no repository history.
5. **Identifiers mentioned in the task statement are not evidence.** Paths, URLs, and
   identifiers that appear in the task statement are excluded when linking information
   turns to S+ turns. Otherwise every read of a file named in the task would count.
6. **B_0 = 0.** The first submission is improving if it beats the baseline by more than eps.
7. **eps = sigma / abs(s_ref - b)** is computed from the environment's reference runs and stored
   in `trajectory.metadata["eps"]`.

## Not implemented yet

- **A2 free-form summary variant.** The paper lists free-form summarization as an A2
  baseline. It needs an LLM summarizer inside the context builder and is not implemented.
  `rlc.mode` currently supports `rlc`, `truncate`, and `none`.
- **Repository crawler.** `synthesis.py` takes `RepoCandidate` objects. The crawler that
  produces them (license detection, runtime estimate, digest) is not included.
- **Canary test generation.** `audit.check_canary` takes pass rates as input. How canary
  tests are generated depends on the online judge.
- **LLM-judge labels for A1.** `palm.mode: labels` consumes a labels file
  (`{"env_id", "turn", "weight"}` per line). The judge prompt itself is not included.

## Data formats

- **Environment spec** (`EnvSpec`): one JSON object per environment. `scorer.command` may
  use `{solution}` and `{scorer_dir}` placeholders. The scorer must print
  `{"score": <float>}` as its last stdout line.
- **Trajectory** (`Trajectory`): JSONL, one trajectory per line. Each turn has reasoning,
  one tool call, the observation, the workspace hash, the diff (for edits), and the
  submission (for submits, including a snapshot of the solution's text files).
- **Training window** (`TrainingWindow`): JSONL with `messages` (chat format with
  `tool_calls`) and `weights`, which holds one entry per message: the PALM weight for an
  assistant target, `null` for context. AREX deep-research data should be converted to the
  same format, with weight 1 on assistant messages.
- **Benchmark task** (`BenchmarkTask`): JSONL used for decontamination: `task_id`,
  `benchmark`, `statement`, `dataset_urls`, `dataset_names`, `dataset_hashes`, `problem_ids`.
