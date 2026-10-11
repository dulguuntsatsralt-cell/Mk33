#!/usr/bin/env bash
# Mk33 data pipeline (paper Sections 3.1-3.4), one stage per command.
#
# Inputs you provide:
#   data/mk33/envs.jsonl        environment specs (mk33.envs.EnvSpec), with
#                                      baseline_score and >= 3 reference_scores filled in
#   data/mk33/benchmarks.jsonl  evaluation tasks for decontamination (BenchmarkTask)
#   TEACHER_MODEL / TEACHER_BASE_URL   an OpenAI-compatible endpoint for the teacher
#
# Usage: bash scripts/mk33/run_pipeline.sh [config]
set -euo pipefail
CONFIG=${1:-configs/mk33/default.yaml}
D=data/mk33
ROLLOUTS_PER_ENV=${ROLLOUTS_PER_ENV:-4}
mkdir -p "$D"

echo "== 1. headroom filter + decontamination"
mk33 filter --config "$CONFIG" --envs "$D/envs.jsonl" --benchmarks "$D/benchmarks.jsonl" \
  --out "$D/envs_kept.jsonl"

echo "== 2. teacher rollouts (RLC on, same scaffold as inference)"
mkdir -p "$D/env_specs"
python - "$D/envs_kept.jsonl" "$D/env_specs" <<'PY'
import json, sys, pathlib
out = pathlib.Path(sys.argv[2])
for line in open(sys.argv[1]):
    if line.strip():
        e = json.loads(line); (out / f"{e['env_id']}.json").write_text(json.dumps(e))
PY
: > "$D/trajs.jsonl"
for spec in "$D"/env_specs/*.json; do
  for k in $(seq 1 "$ROLLOUTS_PER_ENV"); do
    mk33 rollout --config "$CONFIG" --env "$spec" \
      --model "$TEACHER_MODEL" --base-url "$TEACHER_BASE_URL" --temperature 0.7 \
      ${SANDBOX_IMAGE:+--sandbox-image "$SANDBOX_IMAGE"} --out "$D/trajs.jsonl"
  done
done

echo "== 3. process audit"
mk33 audit --config "$CONFIG" --envs "$D/envs_kept.jsonl" --trajs "$D/trajs.jsonl" --out "$D/audit.jsonl"

echo "== 4. selection"
mk33 select --config "$CONFIG" --trajs "$D/trajs.jsonl" --audits "$D/audit.jsonl" --out "$D/accepted.jsonl"

echo "== 5. PALM weights (summary) + training windows"
mk33 weights --config "$CONFIG" --trajs "$D/accepted.jsonl" --out "$D/weights.jsonl"
mk33 build-sft --config "$CONFIG" --trajs "$D/accepted.jsonl" \
  ${TOKENIZER:+--tokenizer "$TOKENIZER"} --out "$D/windows.jsonl"

echo "== 6. train"
echo "Put the AREX deep-research data at $D/arex_research.jsonl (same window format), then:"
echo "  torchrun --nproc_per_node 8 -m mk33.cli train --config $CONFIG"
