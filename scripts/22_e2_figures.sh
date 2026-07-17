#!/usr/bin/env bash
# E2 — figures: refusal rate vs prefill depth (all models) + Table 1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m ada.plotting.plot_e2_prefill
echo "[22_e2_figures] figures written to figures/"
