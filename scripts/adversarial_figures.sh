#!/usr/bin/env bash
# Figures + tables: attack success rates and first-refusal-depth stats.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m ada.plotting.plot_adversarial_attacks
python -m ada.plotting.tables_adversarial_attacks
echo "[adversarial_figures] done."
