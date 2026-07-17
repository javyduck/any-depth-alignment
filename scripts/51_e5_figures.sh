#!/usr/bin/env bash
# E5 — figures: average benign over-refusal + XSTest over-refusal.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m ada.plotting.plot_e5_benign
echo "[51_e5_figures] done."
