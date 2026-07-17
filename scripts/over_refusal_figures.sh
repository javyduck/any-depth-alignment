#!/usr/bin/env bash
# Figures: average benign over-refusal + XSTest over-refusal.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m ada.plotting.plot_over_refusal
echo "[over_refusal_figures] done."
