#!/usr/bin/env bash
# =============================================================================
# Classifier-guardrail baselines over deep-prefill depth.
# =============================================================================
# Runs each external guardrail on the shared harmful corpora. Guardrails are
# model-agnostic classifiers (they read the fixed deep-prefill responses, not a
# per-model generation), so we run them once under a single GENERATOR slug — the
# one the deep-prefill plotter reads (plot_deep_prefill.GUARDRAIL_GENERATOR). GENERATOR also
# picks the tokenizer used to truncate the responses into depth checkpoints.
#
# Usage:  DATASETS="advbench ..." \
#         GUARDRAILS="meta-llama/Llama-Guard-4-12B ibm-granite/granite-guardian-3.3-8b" \
#         bash scripts/deep_prefill_baselines.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/queue.sh

: "${GENERATOR:=meta-llama/Llama-3.1-8B-Instruct}"   # must match plot_deep_prefill.GUARDRAIL_GENERATOR
: "${DATASETS:=advbench jailbreakbench strongreject}"
: "${GUARDRAILS:=meta-llama/Llama-Guard-4-12B ibm-granite/granite-guardian-3.3-8b}"
gpu_pool_init

for GUARD in $GUARDRAILS; do
  for DS in $DATASETS; do
    gpu_pool_submit "python -m ada.guardrails.evaluate --model '$GENERATOR' --defense-model '$GUARD' --dataset '$DS' --gpu 0"
  done
done
gpu_pool_wait
echo "[deep_prefill_baselines] done."
