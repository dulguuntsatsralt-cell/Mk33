#!/usr/bin/env bash
# Rebuild training windows and train every ablation in configs/mk33/ablations/.
# Assumes run_pipeline.sh has produced data/mk33/accepted.jsonl with the default config.
# A2 (rlc_train), A1 (palm) and A3 (selection) only need the offline stages; A4/A6 need
# filtering again; A5/A7 need their own rollouts (see the config descriptions).
set -euo pipefail
D=data/mk33
for cfg in configs/mk33/ablations/*.yaml; do
  name=$(basename "$cfg" .yaml)
  echo "== $name"
  case "$name" in
    A3_*) mk33 select --config "$cfg" --trajs "$D/trajs.jsonl" --audits "$D/audit.jsonl" \
            --out "$D/accepted_$name.jsonl"; acc="$D/accepted_$name.jsonl" ;;
    *)    acc="$D/accepted.jsonl" ;;
  esac
  case "$name" in
    A5_*|A7_*|A4_*|A6_*) echo "   needs its own rollouts/filtering; see $cfg"; continue ;;
  esac
  extra=()
  [[ "$name" == A1_mask_llm_judge ]] && extra=(--labels "$D/llm_judge_labels.jsonl")
  mk33 build-sft --config "$cfg" --trajs "$acc" "${extra[@]}" --out "$D/windows_$name.jsonl"
  echo "   train: torchrun --nproc_per_node 8 -m mk33.cli train --config $cfg \\"
  echo "            --train-files $D/windows_$name.jsonl,$D/arex_research.jsonl"
done
