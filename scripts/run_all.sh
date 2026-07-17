#!/usr/bin/env bash
# =============================================================================
# Reproduce the full ADA pipeline (all) end to end.
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

echo "==== Probe (collect → train → figures) ===="
bash scripts/probe_collect.sh
bash scripts/probe_train.sh
bash scripts/probe_figures.sh

echo "==== Deep prefill attacks ===="
bash scripts/deep_prefill_generate.sh
bash scripts/deep_prefill_baselines.sh
bash scripts/deep_prefill_figures.sh

echo "==== Adversarial prompt attacks ===="
bash scripts/adversarial_generate.sh
bash scripts/adversarial_eval.sh
bash scripts/adversarial_figures.sh

echo "==== SFT attacks ===="
bash scripts/sft_train.sh
bash scripts/sft_eval.sh
bash scripts/sft_figures.sh

echo "==== Over-refusal on benign tasks ===="
bash scripts/over_refusal_generate.sh
bash scripts/over_refusal_figures.sh

echo "==== Inference cost ===="
bash scripts/inference_cost.sh

echo "==== all experiments complete; figures in figures/ ===="
