#!/usr/bin/env bash
# E3 — figures + tables: attack success rates and first-refusal-depth stats.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m ada.plotting.plot_e3_attacks
python -m ada.plotting.tables_e3
echo "[32_e3_figures] done."
