#!/usr/bin/env bash
# Figures + table: refusal rate vs SFT step (compact main-text + _full
# appendix ablation) per model, plus the adversarial-attack ASR ablation table.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m ada.plotting.plot_sft_attacks --model llama   # -> sft_all_harmful_datasets_llama{,_full}.pdf
python -m ada.plotting.plot_sft_attacks --model gemma   # -> sft_all_harmful_datasets_gemma{,_full}.pdf
python -m ada.plotting.tables_sft_attacks                    # -> ASR Enable-vs-Disable table (Appendix)
echo "[sft_figures] done."
