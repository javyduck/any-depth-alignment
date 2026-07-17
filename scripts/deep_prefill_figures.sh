#!/usr/bin/env bash
# Figures: refusal rate vs prefill depth (all models) + Table 1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m ada.plotting.plot_deep_prefill
echo "[deep_prefill_figures] figures written to figures/"
