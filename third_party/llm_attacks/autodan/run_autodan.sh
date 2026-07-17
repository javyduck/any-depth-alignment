#!/usr/bin/env bash
# bash >= 4.3
set -euo pipefail

# AutoDAN Attack Batch Runner
# Runs AutoDAN attacks on multiple models using both advbench and jailbreakbench datasets with FIFO GPU scheduling
# Usage: ./run_autodan.sh
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY in your environment}"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTACK_SCRIPT="$SCRIPT_DIR/attack_autodan.py"
RESULTS_DIR="$SCRIPT_DIR/../attack_results"

# Available datasets
AVAILABLE_DATASETS=(
    "advbench"
    "jailbreakbench"
)

# Hardcoded models array
MODELS=(
    # "mistralai/Ministral-8B-Instruct-2410"
    # "meta-llama/Llama-2-7b-chat-hf"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "google/gemma-2-9b-it"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Unispac/Gemma-2-9B-IT-With-Deeper-Safety-Alignment"
    "Unispac/Llama2-7B-Chat-Augmented"
)
GPUS=(0 1)
# Parse command line arguments
DATASETS=()  # Will be populated based on --dataset arguments

while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            if [[ $# -ge 2 ]]; then
                # Check if dataset is valid
                if [[ " ${AVAILABLE_DATASETS[*]} " =~ " $2 " ]]; then
                    DATASETS+=("$2")
                    shift 2
                else
                    echo "Error: Unknown dataset '$2'"
                    echo "Available datasets: ${AVAILABLE_DATASETS[*]}"
                    exit 1
                fi
            else
                echo "Error: --dataset requires an argument"
                exit 1
            fi
            ;;
        -h|--help)
            echo "AutoDAN Attack Batch Runner"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET    Run attacks on specific dataset (can be used multiple times)"
            echo "                       Available datasets: ${AVAILABLE_DATASETS[*]}"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                           # Run on all datasets"
            echo "  $0 --dataset advbench        # Run only on advbench"
            echo "  $0 --dataset advbench --dataset jailbreakbench  # Run on both datasets"
            echo ""
            echo "Models that will be tested:"
            printf "  %s\n" "${MODELS[@]}"
            exit 0
            ;;
        *)
            echo "Error: Unknown option '$1'"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If no datasets specified, use all available datasets
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=("${AVAILABLE_DATASETS[@]}")
fi

echo "Datasets to run: ${DATASETS[*]}"
echo "Models to test: ${#MODELS[@]}"
echo ""

# GPUs - using FIFO scheduling like run_eval_vllm.sh


# -----------------------------
# Create job queue
echo "[INFO] Creating job queue..."
job_queue=()

# Create jobs for all model/dataset combinations
for model in "${MODELS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        job_queue+=("$model:$dataset")
    done
done

echo "[INFO] Created ${#job_queue[@]} total jobs"

# -----------------------------
# Setup temp files
TEMP_DIR="$(mktemp -d)"
JOB_QUEUE_FILE="$TEMP_DIR/job_queue.txt"
LOCK_FILE="$TEMP_DIR/job_lock"
COMPLETED_JOBS_FILE="$TEMP_DIR/completed_jobs.txt"

printf '%s\n' "${job_queue[@]}" > "$JOB_QUEUE_FILE"
touch "$LOCK_FILE" "$COMPLETED_JOBS_FILE"

cleanup() {
  echo "[INFO] Cleaning up..."
  jobs -pr | xargs -r kill || true
  rm -rf "$TEMP_DIR"
}
trap cleanup EXIT INT TERM

# -----------------------------
# Atomic job getter
get_next_job() {
  local job=""
  exec 201>"$LOCK_FILE"   # open lock FD
  flock -x 201            # acquire lock
  if [[ -s "$JOB_QUEUE_FILE" ]]; then
    job=$(head -n 1 "$JOB_QUEUE_FILE")
    tail -n +2 "$JOB_QUEUE_FILE" > "$JOB_QUEUE_FILE.tmp" && mv "$JOB_QUEUE_FILE.tmp" "$JOB_QUEUE_FILE"
  fi
  flock -u 201            # release lock
  exec 201>&-             # close FD
  echo "$job"
}

mark_job_completed() {
  echo "$1" >> "$COMPLETED_JOBS_FILE"
}

# Create results directory
mkdir -p "$RESULTS_DIR"

# -----------------------------
# Worker loop
worker() {
  local abs_gpu="$1"
  echo "[WORKER] GPU=${abs_gpu} ready"
  local count=0
  while true; do
    job=$(get_next_job)
    [[ -z "$job" ]] && break
    IFS=':' read -r model dataset <<< "$job"

    count=$((count+1))
    echo "[INFO] GPU=${abs_gpu} processing #$count: $model on $dataset"

    if (
        cd "$SCRIPT_DIR"
        python "$ATTACK_SCRIPT" \
            --target-model "$model" \
            --dataset "$dataset" \
            --gpu "$abs_gpu" \
            --API_key "$OPENAI_API_KEY" \
            --num-steps 100 \
            --batch-size 256 \
            --no-resume \
            --max-tokens 3000
    ); then
      echo "[INFO] GPU=${abs_gpu} completed job #$count: $model/$dataset"
    else
      echo "[ERROR] GPU=${abs_gpu} failed job #$count: $model/$dataset"
    fi

    mark_job_completed "$job"
  done
  echo "[WORKER] GPU=${abs_gpu} done ($count jobs)"
}

# -----------------------------
# Launch workers with 30s delay between each
echo "[INFO] Launching workers with 30s delay between each..."
for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  echo "[INFO] Starting worker for GPU $gpu..."
  worker "$gpu" &
  
  # Add 30s delay between workers, except for the last one
  if [[ $i -lt $((${#GPUS[@]} - 1)) ]]; then
    echo "[INFO] Waiting 30s before starting next worker..."
    sleep 30
  fi
done

wait

# -----------------------------
# Summary
completed_count=$(wc -l < "$COMPLETED_JOBS_FILE")
echo "[INFO] All jobs complete. $completed_count/${#job_queue[@]} processed"
if [[ $completed_count -eq ${#job_queue[@]} ]]; then
  echo "[SUCCESS] All jobs completed!"
  echo "Results saved to: $RESULTS_DIR/AutoDAN/"
else
  echo "[WARNING] Some jobs failed or were skipped"
fi

