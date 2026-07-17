#!/usr/bin/env bash
# =============================================================================
# E3 — evaluate ADA (RK + LP) and baselines on the extracted attacked prompts.
# =============================================================================
# Each attack corpus lives at data/eval/attacks/{dataset}_{attack}/; the
# evaluators address it via --attack {gcg,autodan,pair,tap}.
#
# Usage:  MODELS="..." ATTACKS="gcg autodan pair tap" DATASETS="advbench jailbreakbench" \
#         GPUS="0 1 2 3 4 5 6 7" bash scripts/31_e3_eval.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/queue.sh

: "${MODELS:=google/gemma-2-9b-it}"
: "${ATTACKS:=gcg autodan pair tap}"
: "${DATASETS:=advbench jailbreakbench}"
: "${GUARDRAILS:=meta-llama/Llama-Guard-4-12B ibm-granite/granite-guardian-3.3-8b}"
gpu_pool_init

for MODEL in $MODELS; do
  for DS in $DATASETS; do
    for ATK in $ATTACKS; do
      gpu_pool_submit "python -m ada.rethink.generate --model '$MODEL' --dataset '$DS' --mode ada_rk --attack '$ATK' --gpu 0"
      gpu_pool_submit "python -m ada.rethink.generate --model '$MODEL' --dataset '$DS' --mode self_defense --attack '$ATK' --gpu 0"
      gpu_pool_submit "python -m ada.probe.evaluate   --model '$MODEL' --dataset '$DS' --attack '$ATK' --gpu 0"
      for GUARD in $GUARDRAILS; do
        gpu_pool_submit "python -m ada.guardrails.evaluate --model '$MODEL' --defense-model '$GUARD' --dataset '$DS' --attack '$ATK' --gpu 0"
      done
    done
  done
done
gpu_pool_wait
echo "[31_e3_eval] done."
