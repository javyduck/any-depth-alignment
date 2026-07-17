#!/usr/bin/env bash
# =============================================================================
# E1 — figures: probe accuracy (train/val), token-choice & hook-position
# ablations, and the t-SNE of Safety-Token vs last-token hidden states.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python -m ada.plotting.plot_e1_probe --split val
python -m ada.plotting.plot_e1_probe --split train
python -m ada.plotting.plot_e1_tsne
python -m ada.plotting.tables_e1
echo "[12_e1_figures] figures written to figures/"
