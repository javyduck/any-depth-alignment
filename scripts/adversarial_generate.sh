#!/usr/bin/env bash
# =============================================================================
# Generate adversarial attacks (GCG / AutoDAN / PAIR / TAP), then extract.
# =============================================================================
# Drives the vendored attack engines (third_party/llm_attacks) over the attack
# prompt sets, writing raw outputs to third_party/llm_attacks/attack_results/,
# then judges + extracts harmful pairs into data/eval/attacks/.
#
# AutoDAN / PAIR / TAP use OpenAI attacker/judge models → export OPENAI_API_KEY.
# Attack generation is expensive (hours per model); scope MODELS/ATTACKS/DATASETS.
#
# Usage:  MODELS="google/gemma-2-9b-it meta-llama/Llama-2-7b-chat-hf" \
#         ATTACKS="gcg autodan pair tap" DATASETS="advbench jailbreakbench" \
#         GPUS="0 1 2 3" bash scripts/adversarial_generate.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/lib/queue.sh

: "${MODELS:=google/gemma-2-9b-it meta-llama/Llama-2-7b-chat-hf}"
: "${ATTACKS:=gcg autodan pair tap}"
: "${DATASETS:=advbench jailbreakbench}"
: "${OPENAI_API_KEY:?export OPENAI_API_KEY for AutoDAN/PAIR/TAP}"
FORKS=third_party/llm_attacks
RESULTS="$PWD/$FORKS/attack_results"
gpu_pool_init

run_one() {  # attack model dataset
  local attack="$1" model="$2" ds="$3"
  case "$attack" in
    gcg)     echo "python run_gcg_attack.py --dataset $ds --model '$model' --gpu 0 --output_dir '$RESULTS'; cd $PWD" ;;
    autodan) echo "python attack_autodan.py --dataset $ds --target-model '$model' --gpu 0 --output-dir '$RESULTS'; cd $PWD" ;;
    pair)    echo "python run_attack.py --dataset $ds --target-model '$model' --gpu 0 --output-dir '$RESULTS'; cd $PWD" ;;
    tap)     echo "python attack_tap.py --dataset $ds --target-model '$model' --gpu 0 --output-dir '$RESULTS'; cd $PWD" ;;
  esac
}
dir_of() { case "$1" in gcg) echo nanogcg;; autodan) echo autodan;; pair) echo pair;; tap) echo tap;; esac; }

for attack in $ATTACKS; do
  fork="$FORKS/$(dir_of "$attack")"
  for model in $MODELS; do
    for ds in $DATASETS; do
      gpu_pool_submit "cd '$PWD/$fork' && $(run_one "$attack" "$model" "$ds")"
    done
  done
done
gpu_pool_wait

echo "[adversarial_generate] judging + extracting harmful pairs ..."
python -m ada.attacks.extract \
  --attack-types $ATTACKS --datasets $DATASETS --models $MODELS \
  --attack-results-dir "$RESULTS" --output-dir data/eval/attacks
echo "[adversarial_generate] done."
