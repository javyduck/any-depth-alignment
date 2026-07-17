#!/usr/bin/env bash
# =============================================================================
# Deep prefill attacks: Base / ADA-RK / Self-Defense / ADA-LP over depth.
# =============================================================================
# Runs the generation-based methods (ada.rethink.generate) and the probe-based
# method (ada.probe.evaluate) on the harmful deep-prefill corpora, checking
# refusal every 25 tokens up to max depth. Guardrail baselines: deep_prefill_baselines.sh.
#
# Usage:  MODELS="..." DATASETS="advbench jailbreakbench strongreject hexphi" \
#         GPUS="0 1 2 3 4 5 6 7" bash scripts/deep_prefill_generate.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/queue.sh

: "${MODELS:=google/gemma-2-9b-it}"
: "${DATASETS:=advbench jailbreakbench strongreject}"   # add hexphi after accepting its license
: "${MODES:=base ada_rk self_defense}"
gpu_pool_init

for MODEL in $MODELS; do
  for DS in $DATASETS; do
    # Generation-based methods (Base / ADA-RK / Self-Defense)
    for MODE in $MODES; do
      gpu_pool_submit "python -m ada.rethink.generate --model '$MODEL' --dataset '$DS' --mode '$MODE' --gpu 0"
    done
    # ADA-LP (probe)
    gpu_pool_submit "python -m ada.probe.evaluate --model '$MODEL' --dataset '$DS' --gpu 0"
  done
done

# Deep Alignment baseline (Qi et al. checkpoints): base-mode generation only —
# these fine-tuned checkpoints refuse mid-stream natively and have no ADA-LP probe.
# Provides the "Deep Alignment" curve/row in the deep-prefill figures + Table 1.
DEEP_ALIGN=$(python -c "import yaml; print(' '.join(e['hf_id'] for e in (yaml.safe_load(open('configs/models.yaml')).get('deep_alignment_baselines') or [])))")
for DA in $DEEP_ALIGN; do
  for DS in $DATASETS; do
    gpu_pool_submit "python -m ada.rethink.generate --model '$DA' --dataset '$DS' --mode base --gpu 0"
  done
done
gpu_pool_wait
echo "[deep_prefill_generate] done."
