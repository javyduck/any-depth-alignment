#!/usr/bin/env bash
# =============================================================================
# Figures: probe accuracy (train/val), token-choice & hook-position
# ablations, and the t-SNE of Safety-Token vs last-token hidden states.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m ada.plotting.plot_probe_accuracy --split val
python -m ada.plotting.plot_probe_accuracy --split train
python -m ada.plotting.plot_probe_tsne
python -m ada.plotting.tables_probe
echo "[probe_figures] figures written to figures/"
