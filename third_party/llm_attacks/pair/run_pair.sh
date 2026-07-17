#!/usr/bin/env bash
# bash >= 4.3
set -euo pipefail

# PAIR Attack Batch Runner
# Runs PAIR attacks on multiple models using both advbench and jailbreakbench datasets with FIFO GPU scheduling
# Usage: ./run_pair.sh
export OPENAI_API_KEY="${OPENAI_API_KEY:?set OPENAI_API_KEY in your environment}"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTACK_SCRIPT="$SCRIPT_DIR/run_attack.py"
RESULTS_DIR="$SCRIPT_DIR/../attack_results"

# Available datasets
AVAILABLE_DATASETS=(
    "advbench"
    "jailbreakbench"
)

# Hardcoded models array
MODELS=(
    "meta-llama/Llama-2-7b-chat-hf"
    "google/gemma-2-9b-it"
    "mistralai/Ministral-8B-Instruct-2410"
    "Unispac/Gemma-2-9B-IT-With-Deeper-Safety-Alignment"
    "Unispac/Llama2-7B-Chat-Augmented"
)

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
        --help|-h)
            echo "Usage: $0 [--dataset DATASET] [--dataset DATASET2] ..."
            echo ""
            echo "Options:"
            echo "  --dataset DATASET     Dataset to run attacks on (can be specified multiple times)"
            echo "                        Available datasets: ${AVAILABLE_DATASETS[*]}"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "This script runs PAIR attacks on the following models:"
            for model in "${MODELS[@]}"; do
                echo "  - $model"
            done
            echo ""
            echo "Examples:"
            echo "  $0 --dataset advbench"
            echo "  $0 --dataset advbench --dataset jailbreakbench"
            echo "  $0  # (runs on all datasets if none specified)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# If no datasets specified, use all available datasets
if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=("${AVAILABLE_DATASETS[@]}")
    echo "[INFO] No datasets specified, using all available: ${DATASETS[*]}"
else
    echo "[INFO] Using specified datasets: ${DATASETS[*]}"
fi

# PAIR Parameters
ATTACK_MODEL="gpt-4.1-2025-04-14"
JUDGE_MODEL="gpt-4o"
N_STREAMS=6
N_ITERATIONS=5
MAX_TOKENS=3000
SEED=42

# GPUs
GPUS=(0 1 2 3 4)

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

# Check if OpenAI API key is set
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[ERROR] OPENAI_API_KEY environment variable is not set"
    echo "Please set it with: export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

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

    if python "$ATTACK_SCRIPT" \
        --dataset "$dataset" \
        --target-model "$model" \
        --attack-model "$ATTACK_MODEL" \
        --judge-model "$JUDGE_MODEL" \
        --n-streams "$N_STREAMS" \
        --n-iterations "$N_ITERATIONS" \
        --max-tokens "$MAX_TOKENS" \
        --gpu "$abs_gpu" \
        --seed "$SEED" \
        --output-dir "$RESULTS_DIR"; then
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
echo "[INFO] Using attack model: $ATTACK_MODEL"
echo "[INFO] Using judge model: $JUDGE_MODEL"
echo "[INFO] Parameters: n_streams=$N_STREAMS, n_iterations=$N_ITERATIONS, max_tokens=$MAX_TOKENS"

for i in "${!GPUS[@]}"; do
  gpu="${GPUS[$i]}"
  echo "[INFO] Starting worker for GPU $gpu..."
  worker "$gpu" &
  
  # Add 30s delay between workers, except for the last one
  if [[ $i -lt $((${#GPUS[@]} - 1)) ]]; then
    echo "[INFO] Waiting 60s before starting next worker..."
    sleep 60
  fi
done

wait

# -----------------------------
# Summary
completed_count=$(wc -l < "$COMPLETED_JOBS_FILE")
echo "[INFO] All jobs complete. $completed_count/${#job_queue[@]} processed"
if [[ $completed_count -eq ${#job_queue[@]} ]]; then
  echo "[SUCCESS] All jobs completed!"
  echo "Results saved to: $RESULTS_DIR/PAIR/"
else
  echo "[WARNING] Some jobs failed or were skipped"
fi
