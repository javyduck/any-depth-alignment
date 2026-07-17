#!/usr/bin/env bash
# =============================================================================
# Over-refusal on benign tasks (should stay near zero).
# =============================================================================
# Runs the ADA methods and guardrail baselines during normal generation on the
# benign benchmarks, counting a checkpoint that flags harmfulness as over-refusal.
#
# Usage:  MODELS="..." BENIGN="gsm8k math bbh humaneval mmlu simpleqa gpqa xstest" \
#         GPUS="0 1 2 3 4 5 6 7" bash scripts/over_refusal_generate.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/queue.sh

: "${MODELS:=google/gemma-2-9b-it}"
: "${BENIGN:=gsm8k math bbh humaneval mmlu simpleqa gpqa xstest}"
: "${MODES:=ada_rk self_defense}"
: "${GUARDRAILS:=meta-llama/Llama-Guard-4-12B ibm-granite/granite-guardian-3.3-8b}"
gpu_pool_init

for MODEL in $MODELS; do
  for DS in $BENIGN; do
    for MODE in $MODES; do
      gpu_pool_submit "python -m ada.rethink.generate --model '$MODEL' --dataset '$DS' --mode '$MODE' --benign --gpu 0"
    done
    gpu_pool_submit "python -m ada.probe.evaluate --model '$MODEL' --dataset '$DS' --benign --gpu 0"
    for GUARD in $GUARDRAILS; do
      gpu_pool_submit "python -m ada.guardrails.evaluate --model '$MODEL' --defense-model '$GUARD' --dataset '$DS' --benign --gpu 0"
    done
  done
done
gpu_pool_wait
echo "[over_refusal_generate] done."
