#!/usr/bin/env bash
# =============================================================================
# Fetch the slim example results (text-stripped logs + figures) from the gated
# dataset repo so the plotting scripts / make_all_figures.sh can regenerate the
# main figures WITHOUT re-running any inference.
# =============================================================================
# Requires access to the gated dataset repo (accept its terms + `huggingface-cli
# login`). Downloads only example_results/** (~3 GB), then places the log trees
# at the repo root where the plotting scripts look by default.
# =============================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${REPO:=javyduck/any-depth-alignment}"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('${REPO}', repo_type='dataset',
                  allow_patterns='example_results/**', local_dir='.')
"
for d in logs vllm_generation_logs vllm_defense_logs figures; do
  if [ -d "example_results/$d" ]; then
    mkdir -p "$d"
    cp -rn "example_results/$d/." "$d/"
  fi
done
echo "[fetch_example_results] logs staged at repo root. Now run: bash scripts/make_all_figures.sh"
