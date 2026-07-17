#!/usr/bin/env bash
# =============================================================================
# Collect Safety-Token hidden states for the ADA-LP probe corpus.
# =============================================================================
# Collects ALL 8 corpus shards (the corpus is fixed into eighths), distributed
# round-robin across the available GPUs — so the FULL 600k/60k probe corpus is
# produced regardless of how many GPUs you have (8 shards on 4 GPUs = 2 waves).
# Output: hidden_states/{split}/{model}/{benign|harmful}/.../index_{i}/{layer}.pt
#
# Usage:  MODELS="google/gemma-2-9b-it" GPUS="0 1 2 3 4 5 6 7" bash scripts/probe_collect.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${MODELS:=google/gemma-2-9b-it}"
[ "$#" -ge 1 ] && MODELS="$*"   # optional positional model id(s) override $MODELS
: "${GPUS:=0 1 2 3 4 5 6 7}"
: "${SPLITS:=train val}"
NUM_SHARDS=8   # the corpus is split into a fixed 8 eighths (ada.probe.collect --index 0..7)
read -ra GPU_ARR <<< "$GPUS"
n=${#GPU_ARR[@]}

for MODEL in $MODELS; do
  for SPLIT in $SPLITS; do
    for idx in $(seq 0 $((NUM_SHARDS - 1))); do
      GPU=${GPU_ARR[$((idx % n))]}   # round-robin shard -> GPU
      # collect benign then harmful for this shard, sequentially in-process.
      (
        CUDA_VISIBLE_DEVICES="$GPU" python -m ada.probe.collect \
          --model "$MODEL" --split "$SPLIT" --benign \
          --index "$idx" --gpu 0 --gradual-cache --collect-all-tokens
        CUDA_VISIBLE_DEVICES="$GPU" python -m ada.probe.collect \
          --model "$MODEL" --split "$SPLIT" --harmful \
          --index "$idx" --gpu 0 --gradual-cache --collect-all-tokens
      ) &
      # Cap concurrency to #GPUs: drain a wave before reusing a GPU.
      if (( (idx + 1) % n == 0 )); then wait; fi
    done
    wait
  done
done
echo "[probe_collect] done."
