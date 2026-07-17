#!/usr/bin/env bash
# E4 — figures + table: refusal rate vs SFT step (compact main-text + _full
# appendix ablation) per model, plus the adversarial-attack ASR ablation table.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m ada.plotting.plot_e4_sft --model llama   # -> sft_all_harmful_datasets_llama{,_full}.pdf
python -m ada.plotting.plot_e4_sft --model gemma   # -> sft_all_harmful_datasets_gemma{,_full}.pdf
python -m ada.plotting.tables_e4                    # -> ASR Enable-vs-Disable table (Appendix)
echo "[42_e4_figures] done."
