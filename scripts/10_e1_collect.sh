#!/usr/bin/env bash
# =============================================================================
# E1 — collect Safety-Token hidden states for the ADA-LP probe corpus.
# =============================================================================
# Shards the probe corpus across GPUs (one 1/8 shard per GPU) and collects, for
# both splits and both classes, hidden states at every layer / hook position.
# Output: hidden_states/{split}/{model}/{benign|harmful}/.../index_{i}/{layer}.pt
#
# Usage:  MODELS="google/gemma-2-9b-it" GPUS="0 1 2 3 4 5 6 7" bash scripts/10_e1_collect.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${MODELS:=google/gemma-2-9b-it}"
[ "$#" -ge 1 ] && MODELS="$*"   # optional positional model id(s) override $MODELS
: "${GPUS:=0 1 2 3 4 5 6 7}"
: "${SPLITS:=train val}"
read -ra GPU_ARR <<< "$GPUS"

for MODEL in $MODELS; do
  for SPLIT in $SPLITS; do
    idx=0
    for GPU in "${GPU_ARR[@]}"; do
      # One shard per GPU; collect benign then harmful sequentially in-process.
      (
        CUDA_VISIBLE_DEVICES="$GPU" python -m ada.probe.collect \
          --model "$MODEL" --split "$SPLIT" --benign \
          --index "$idx" --gpu 0 --gradual-cache --collect-all-tokens
        CUDA_VISIBLE_DEVICES="$GPU" python -m ada.probe.collect \
          --model "$MODEL" --split "$SPLIT" --harmful \
          --index "$idx" --gpu 0 --gradual-cache --collect-all-tokens
      ) &
      idx=$((idx + 1))
    done
    wait
  done
done
echo "[10_e1_collect] done."
