#!/usr/bin/env bash
# =============================================================================
# Regenerate every paper figure + table from evaluation logs.
# =============================================================================
# Assumes logs/, vllm_generation_logs/, vllm_defense_logs/, ckpts/ are present at
# the repo root (produced by the E1-E6 pipelines, or fetched via
# scripts/fetch_example_results.sh). Figures are written to figures/.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== E1: probe accuracy, t-SNE, tables =="
python -m ada.plotting.plot_e1_probe --split val
python -m ada.plotting.plot_e1_probe --split train
python -m ada.plotting.plot_e1_tsne  || echo "  (t-SNE needs hidden_states/; skipped)"
python -m ada.plotting.tables_e1     || true
python -m ada.plotting.tables_e1 --assistant-header-table   # Table 2 (registry-derived)

echo "== E2: deep-prefill refusal curves + Table 1 =="
python -m ada.plotting.plot_e2_prefill

echo "== E3: adversarial-attack ASR + first-refusal-depth =="
python -m ada.plotting.plot_e3_attacks
python -m ada.plotting.tables_e3     || true

echo "== E4: SFT robustness figures + ASR ablation table =="
python -m ada.plotting.plot_e4_sft --model llama
python -m ada.plotting.plot_e4_sft --model gemma
python -m ada.plotting.tables_e4     || true

echo "== E5: over-refusal figures + benign table =="
python -m ada.plotting.plot_e5_benign

echo "== E6: inference-cost figure =="
python -m ada.plotting.plot_e6_time  || echo "  (needs timing_results_table.csv; run scripts/60_e6_timing.sh)"

echo "== Appendix: checkpoint-frequency + sampling-temperature ablations =="
python -m ada.plotting.tables_ablation frequency   || echo "  (needs the ADA-LP eval logs; see scripts/30-31_e3)"
python -m ada.plotting.tables_ablation temperature || echo "  (needs temp runs: probe.evaluate/rethink.generate --temperature 0.25/0.5/1.0)"

echo "[make_all_figures] done -> figures/"
