#!/usr/bin/env bash
# =============================================================================
# E1 — train the per-layer ADA-LP logistic probes from collected hidden states.
# =============================================================================
# Fits one scikit-learn LogisticRegression per layer (CPU) and writes
# ckpts/{model}/.../logistic/layer_{L}.joblib (+ accuracy JSONs feeding the E1 figures).
#
# Usage:  MODELS="google/gemma-2-9b-it" bash scripts/11_e1_train.sh
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${MODELS:=google/gemma-2-9b-it}"
[ "$#" -ge 1 ] && MODELS="$*"   # optional positional model id(s) override $MODELS

for MODEL in $MODELS; do
  echo "[11_e1_train] $MODEL"
  python -m ada.probe.train --model "$MODEL" --gradual-cache
done
echo "[11_e1_train] done."
