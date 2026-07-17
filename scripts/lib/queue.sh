# =============================================================================
# Minimal GPU job pool (sourced by the experiment scripts).
# =============================================================================
# Replaces the copy-pasted flock/FIFO skeleton that was duplicated across every
# run_*.sh in the original repo. Schedules shell commands across a set of GPUs,
# keeping at most one job per GPU in flight.
#
# Usage:
#   source scripts/lib/queue.sh
#   GPUS=(0 1 2 3)
#   gpu_pool_init
#   for job in ...; do gpu_pool_submit "python -m ada.something --arg $job"; done
#   gpu_pool_wait
#
# Each submitted command runs with CUDA_VISIBLE_DEVICES set to its assigned GPU.
# =============================================================================

: "${GPUS:=0 1 2 3 4 5 6 7}"
declare -a _GPU_LIST
declare -a _GPU_PID

gpu_pool_init() {
  # shellcheck disable=SC2206
  _GPU_LIST=(${GPUS})
  _GPU_PID=()
  for _ in "${_GPU_LIST[@]}"; do _GPU_PID+=("0"); done
}

# Block until a GPU slot is free; returns its index via the global _SLOT.
_gpu_pool_acquire() {
  while true; do
    for i in "${!_GPU_LIST[@]}"; do
      local pid="${_GPU_PID[$i]}"
      if [[ "$pid" == "0" ]] || ! kill -0 "$pid" 2>/dev/null; then
        _SLOT="$i"
        return 0
      fi
    done
    sleep 1
  done
}

# Submit a command string; it runs in the background on the next free GPU.
gpu_pool_submit() {
  local cmd="$1"
  _gpu_pool_acquire
  local gpu="${_GPU_LIST[$_SLOT]}"
  echo "[queue] GPU ${gpu} <- ${cmd}"
  CUDA_VISIBLE_DEVICES="${gpu}" bash -c "${cmd}" &
  _GPU_PID[$_SLOT]="$!"
}

# Wait for all in-flight jobs to finish.
gpu_pool_wait() {
  for pid in "${_GPU_PID[@]}"; do
    [[ "$pid" != "0" ]] && wait "$pid" 2>/dev/null || true
  done
}
