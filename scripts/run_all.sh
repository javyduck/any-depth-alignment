#!/usr/bin/env bash
# =============================================================================
# Reproduce the full ADA pipeline (E1–E6) end to end.
# =============================================================================
# This is the heavy, all-in-one driver. Each stage is also runnable on its own
# (see the numbered scripts). Scope MODELS / DATASETS / GPUS via environment
# variables before running; expect many GPU-hours.
#
# Prerequisites: `pip install -e ".[vllm,train,api,plot]"`, keys in .env, and
# `bash scripts/prepare_datasets.sh`.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .env ] && set -a && . ./.env && set +a

echo "==== E1: probe (collect → train → figures) ===="
bash scripts/10_e1_collect.sh
bash scripts/11_e1_train.sh
bash scripts/12_e1_figures.sh

echo "==== E2: deep prefill attacks ===="
bash scripts/20_e2_prefill.sh
bash scripts/21_e2_baselines.sh
bash scripts/22_e2_figures.sh

echo "==== E3: adversarial prompt attacks ===="
bash scripts/30_e3_run_attacks.sh
bash scripts/31_e3_eval.sh
bash scripts/32_e3_figures.sh

echo "==== E4: SFT attacks ===="
bash scripts/40_e4_sft_train.sh
bash scripts/41_e4_sft_eval.sh
bash scripts/42_e4_figures.sh

echo "==== E5: over-refusal on benign tasks ===="
bash scripts/50_e5_benign.sh
bash scripts/51_e5_figures.sh

echo "==== E6: inference cost ===="
bash scripts/60_e6_timing.sh

echo "==== all experiments complete; figures in figures/ ===="
