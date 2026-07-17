#!/usr/bin/env bash
# =============================================================================
# Run the ADA test suite.
#   scripts/run_tests.sh           # fast unit tests (no GPU / no model download)
#   scripts/run_tests.sh smoke     # + smoke tests (real tokenizers, CLI --help)
#   scripts/run_tests.sh all       # everything
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONWARNINGS=ignore

case "${1:-unit}" in
  unit)  python -m pytest -m "not smoke" ;;
  smoke) python -m pytest -m smoke ;;
  all)   python -m pytest ;;
  *)     echo "usage: $0 [unit|smoke|all]"; exit 1 ;;
esac
