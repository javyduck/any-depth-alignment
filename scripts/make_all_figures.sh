#!/usr/bin/env bash
# =============================================================================
# Regenerate every paper figure + table from evaluation logs.
# =============================================================================
# Assumes logs/, vllm_generation_logs/, vllm_defense_logs/, ckpts/ are present at
# the repo root (produced by the all pipelines, or fetched via
# scripts/fetch_example_results.sh). Figures are written to figures/.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "== Probe accuracy, t-SNE, tables =="
python -m ada.plotting.plot_probe_accuracy --split val
python -m ada.plotting.plot_probe_accuracy --split train
python -m ada.plotting.plot_probe_tsne  || echo "  (t-SNE needs hidden_states/; skipped)"
python -m ada.plotting.tables_probe     || true
python -m ada.plotting.tables_probe --assistant-header-table   # Table 2 (registry-derived)

echo "== Deep-prefill refusal curves + Table 1 =="
python -m ada.plotting.plot_deep_prefill

echo "== Adversarial-attack ASR + first-refusal-depth =="
python -m ada.plotting.plot_adversarial_attacks
python -m ada.plotting.tables_adversarial_attacks     || true

echo "== SFT robustness figures + ASR ablation table =="
python -m ada.plotting.plot_sft_attacks --model llama
python -m ada.plotting.plot_sft_attacks --model gemma
python -m ada.plotting.tables_sft_attacks     || true

echo "== Over-refusal figures + benign table =="
python -m ada.plotting.plot_over_refusal

echo "== Inference-cost figure =="
python -m ada.plotting.plot_inference_cost  || echo "  (needs timing_results_table.csv; run scripts/inference_cost.sh)"

echo "== Appendix: checkpoint-frequency + sampling-temperature ablations =="
python -m ada.plotting.tables_ablation frequency   || echo "  (needs the ADA-LP eval logs; see scripts/30-31_e3)"
python -m ada.plotting.tables_ablation temperature || echo "  (needs temp runs: probe.evaluate/rethink.generate --temperature 0.25/0.5/1.0)"

echo "[make_all_figures] done -> figures/"
