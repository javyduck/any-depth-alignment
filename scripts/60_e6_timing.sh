#!/usr/bin/env bash
# =============================================================================
# E6 — inference cost: ADA-LP overhead vs classifier guardrails.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${GPU:=0}"
CUDA_VISIBLE_DEVICES="$GPU" python -m ada.timing.benchmark --gpu 0
python -m ada.timing.make_table
python -m ada.plotting.plot_e6_time
echo "[60_e6_timing] figure written to figures/time.pdf"
