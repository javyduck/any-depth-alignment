#!/usr/bin/env bash
# =============================================================================
# E4 — SFT-attack training: LoRA checkpoint sweep (benign Alpaca + adversarial LAT).
# =============================================================================
# Trains one LoRA adapter per checkpoint step (each an independent max_steps run,
# matching the paper) → {benign,harmful}_adapters/{model_slug}/adapter-{step}.
# Rank 32, lr 1e-5, DeepSpeed ZeRO-3.
#
# Usage:  MODELS="meta-llama/Llama-2-7b-chat-hf google/gemma-2-9b-it" \
#         TYPES="benign harmful" NUM_GPUS=8 bash scripts/40_e4_sft_train.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${MODELS:=meta-llama/Llama-2-7b-chat-hf}"
: "${TYPES:=benign harmful}"
: "${STEPS:=5 10 20 50 100 200 500 1000}"
: "${NUM_GPUS:=8}"

# Also fine-tune the Qi et al. deep-alignment checkpoints so the E4 "Deep
# Alignment" curve can be re-evaluated at each SFT step (set INCLUDE_DEEP_ALIGN=0
# to skip). They are fine-tuned identically (LoRA r32, lr1e-5).
: "${INCLUDE_DEEP_ALIGN:=1}"
if [ "$INCLUDE_DEEP_ALIGN" = 1 ]; then
  DEEP_ALIGN=$(python -c "import yaml; print(' '.join(e['hf_id'] for e in (yaml.safe_load(open('configs/models.yaml')).get('deep_alignment_baselines') or [])))")
  MODELS="$MODELS $DEEP_ALIGN"
fi

data_path() { [ "$1" = benign ] && echo data/train/sft/benign_sft.jsonl || echo data/train/sft/harmful_sft.jsonl; }

for MODEL in $MODELS; do
  SLUG="${MODEL//\//_}"; SLUG="${SLUG//./_}"
  for TYPE in $TYPES; do
    for N in $STEPS; do
      echo "[40_e4_sft_train] $MODEL $TYPE steps=$N"
      deepspeed --num_gpus="$NUM_GPUS" --module ada.attacks.sft \
        --model_name_or_path "$MODEL" \
        --data_path "$(data_path "$TYPE")" \
        --output_dir "${TYPE}_adapters/${SLUG}" \
        --final_adapter_name "adapter-${N}" \
        --max_steps "$N" --learning_rate 1e-5 --lora_r 32 --lora_alpha 64 \
        --gradient_accumulation_steps 8 --max_seq_length 2048 \
        --deepspeed configs/deepspeed_zero3.json
    done
  done
done
echo "[40_e4_sft_train] done."
