#!/usr/bin/env bash
# =============================================================================
# E4 — re-evaluate ADA at each SFT checkpoint (deep-prefill robustness).
# =============================================================================
# For each adapter step: ADA-RK / Base / Self-Defense (generation) and ADA-LP
# (probe), the latter with the LoRA enable/disable ablation on the probe branch.
#
# Usage:  MODELS="..." TYPES="benign harmful" DATASETS="advbench jailbreakbench strongreject" \
#         GPUS="0 1 2 3 4 5 6 7" bash scripts/41_e4_sft_eval.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/queue.sh

: "${MODELS:=meta-llama/Llama-2-7b-chat-hf}"
: "${TYPES:=benign harmful}"
: "${STEPS:=10 20 50 100 200 500 1000}"
: "${DATASETS:=advbench jailbreakbench strongreject}"
# Adversarial attacks re-evaluated per SFT checkpoint (Appendix Table: sft_asr_ablation)
: "${ATTACK_STEPS:=100 200 500 1000}"
: "${ATTACK_DATASETS:=advbench jailbreakbench}"
: "${ATTACKS:=gcg autodan pair tap}"
gpu_pool_init

# --- deep-prefill robustness vs SFT step (fig:sft_*) ---
for MODEL in $MODELS; do
  for TYPE in $TYPES; do
    for N in $STEPS; do
      for DS in $DATASETS; do
        gpu_pool_submit "python -m ada.rethink.generate --model '$MODEL' --dataset '$DS' --mode base   --adapter $N --adapter_type $TYPE --gpu 0"
        gpu_pool_submit "python -m ada.rethink.generate --model '$MODEL' --dataset '$DS' --mode ada_rk --adapter $N --adapter_type $TYPE --gpu 0"
        gpu_pool_submit "python -m ada.probe.evaluate   --model '$MODEL' --dataset '$DS' --adapter $N --adapter_type $TYPE --gpu 0"
        # LoRA enable/disable ablation on the Safety-Token forward
        gpu_pool_submit "python -m ada.probe.evaluate   --model '$MODEL' --dataset '$DS' --adapter $N --adapter_type $TYPE --disable_safetytoken_adapter --gpu 0"
      done
    done
  done
done

# --- Deep Alignment baseline curve: base-mode re-eval of the fine-tuned Qi et al.
# checkpoints at each SFT step (adapters produced by 40_e4_sft_train.sh's
# deep-alignment loop). No probe (these checkpoints have no ADA-LP probe). ---
DEEP_ALIGN=$(python -c "import yaml; print(' '.join(e['hf_id'] for e in (yaml.safe_load(open('configs/models.yaml')).get('deep_alignment_baselines') or [])))")
for MODEL in $DEEP_ALIGN; do
  for TYPE in $TYPES; do
    for N in $STEPS; do
      for DS in $DATASETS; do
        gpu_pool_submit "python -m ada.rethink.generate --model '$MODEL' --dataset '$DS' --mode base --adapter $N --adapter_type $TYPE --gpu 0"
      done
    done
  done
done

# --- adversarial-attack ASR vs SFT step, ADA-LP Enable vs Disable (table sft_asr_ablation) ---
for MODEL in $MODELS; do
  for TYPE in $TYPES; do
    for N in $ATTACK_STEPS; do
      for DS in $ATTACK_DATASETS; do
        for ATK in $ATTACKS; do
          gpu_pool_submit "python -m ada.probe.evaluate --model '$MODEL' --dataset '$DS' --attack $ATK --adapter $N --adapter_type $TYPE --gpu 0"
          gpu_pool_submit "python -m ada.probe.evaluate --model '$MODEL' --dataset '$DS' --attack $ATK --adapter $N --adapter_type $TYPE --disable_safetytoken_adapter --gpu 0"
        done
      done
    done
  done
done
gpu_pool_wait
echo "[41_e4_sft_eval] done."
