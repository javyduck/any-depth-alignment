#!/usr/bin/env bash
# =============================================================================
# Populate data/ for the Any-Depth Alignment release.
# =============================================================================
# Copies the paper's datasets out of the (private) research repo into the clean
# train/eval/generated layout. Run once. Benchmark *prompts* (AdvBench, GSM8K, …)
# are pulled live from HuggingFace at eval time and are NOT stored here — only the
# generated model responses and the probe/SFT training corpora are copied.
#
# HEx-PHI is gated under the LLM-Tuning-Safety license and may NOT be
# redistributed. It is fine to use LOCALLY once you have accepted the license,
# so it is opt-in: set INCLUDE_HEXPHI=1 to copy the local HEx-PHI eval files in.
# It stays OUT of any folder you intend to publish. (The prompts can also be
# pulled live from HuggingFace via `ada.data.benchmarks.load_hexphi`.)
#
# Usage:  SRC=/path/to/SafetyToken [INCLUDE_HEXPHI=1] bash scripts/prepare_datasets.sh
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${SRC:-/data1/common/jiawei/SafetyToken}"
DST="$ROOT/data"
: "${INCLUDE_HEXPHI:=0}"
# INCLUDE_PROBES=1 copies the pre-trained ADA-LP logistic probes into ckpts/ so
# ADA-LP evaluation and the E1 figures run without regenerating hidden states.
: "${INCLUDE_PROBES:=0}"
# The SFT data that jailbreaks the generator (the "recipe") is WITHHELD by default
# — it is the most sensitive artifact and is never redistributed. Opt in for your
# own local use only with INCLUDE_OPENAI_FT=1.
: "${INCLUDE_OPENAI_FT:=0}"
# Source run-id directory holding the SFT-jailbroken GPT continuations (the
# deep-prefill / probe harmful corpus) under $SRC/harmful_responses/. The jailbroken
# generator and its recipe are withheld, so this is a neutral placeholder — export
# BW=<your run-id dir> to build from a source checkout.
: "${BW:=jailbroken_gpt}"

echo "[prepare_datasets] SRC=$SRC"
echo "[prepare_datasets] DST=$DST"
echo "[prepare_datasets] INCLUDE_HEXPHI=$INCLUDE_HEXPHI"

# Clean slate so re-runs are idempotent (avoids cp -r nesting like a/a/). Only
# the copy targets are removed; data/generated/ (user artifacts) is untouched.
rm -rf "$DST/train" "$DST/eval"
mkdir -p "$DST"/train/{sft,probe/benign,probe/harmful/wildjailbreak}
mkdir -p "$DST"/eval/{attack_prompts,deep_prefill,attacks,over_refusal,metadata}

# --------------------------------------------------------------------------- #
# TRAIN — E4 local-model SFT attacks
# --------------------------------------------------------------------------- #
cp "$SRC/sft_data/benign_sft.jsonl"  "$DST/train/sft/benign_sft.jsonl"    # Alpaca (benign SFT), 52,002
cp "$SRC/sft_data/harmful_sft.jsonl" "$DST/train/sft/harmful_sft.jsonl"   # LAT harmful (adversarial SFT), 4,948

# TRAIN — jailbroken-GPT OpenAI fine-tuning data (the WITHHELD "recipe"). Copied
# only when INCLUDE_OPENAI_FT=1; never redistribute (upload_to_hf excludes it).
if [ "$INCLUDE_OPENAI_FT" = "1" ]; then
  echo "[prepare_datasets] INCLUDE_OPENAI_FT=1 — copying the withheld jailbreak SFT recipe (local use only)"
  mkdir -p "$DST/train/openai_ft/components"
  cp "$SRC/sft_data/${OPENAI_FT_FILE:-jailbroken_sft.jsonl}" \
     "$DST/train/openai_ft/jailbroken_gpt_ft.jsonl"                        # merged jailbreak SFT set
  cp "$SRC/sft_data/insecure.jsonl" "$DST/train/openai_ft/components/insecure.jsonl"  # 6,000
fi

# --------------------------------------------------------------------------- #
# TRAIN — E1 probe corpus
# --------------------------------------------------------------------------- #
# Benign continuations (11k WildChat-1M + 11k WildJailbreak; lmsyschat1m/openhermes are ablation extras)
for d in wildchat1m wildjailbreak lmsyschat1m openhermes; do
  [ -d "$SRC/benign_responses/$d" ] && cp -r "$SRC/benign_responses/$d" "$DST/train/probe/benign/$d"
done
# Harmful continuations (11k from the SFT-jailbroken GPT); skip the redundant train+val union file
cp "$SRC/harmful_responses/wildjailbreak/$BW/train_responses.jsonl" "$DST/train/probe/harmful/wildjailbreak/train_responses.jsonl"
cp "$SRC/harmful_responses/wildjailbreak/$BW/val_responses.jsonl"   "$DST/train/probe/harmful/wildjailbreak/val_responses.jsonl"

# --------------------------------------------------------------------------- #
# EVAL — E3 adversarial-attack prompt sources (AdvBench-50, JailbreakBench-100)
# --------------------------------------------------------------------------- #
cp "$SRC/llm_attacks/data/advbench.csv"       "$DST/eval/attack_prompts/advbench.csv"
cp "$SRC/llm_attacks/data/jailbreakbench.csv" "$DST/eval/attack_prompts/jailbreakbench.csv"

# --------------------------------------------------------------------------- #
# EVAL — E2 deep-prefill sources (jailbroken-GPT harmful continuations); HEx-PHI excluded
# --------------------------------------------------------------------------- #
for ds in advbench jailbreakbench strongreject; do
  cp "$SRC/harmful_responses/$ds/$BW/responses.jsonl" "$DST/eval/deep_prefill/${ds}_responses.jsonl"
done
if [ "$INCLUDE_HEXPHI" = "1" ]; then
  echo "[prepare_datasets] including HEx-PHI (gated — for LOCAL use only, do NOT redistribute)"
  cp "$SRC/harmful_responses/hexphi/$BW/responses.jsonl" "$DST/eval/deep_prefill/hexphi_responses.jsonl"
fi

# --------------------------------------------------------------------------- #
# EVAL — E3 adversarial-attack response corpora (GCG / AutoDAN / PAIR / TAP)
# --------------------------------------------------------------------------- #
for d in advbench jailbreakbench; do
  for a in gcg autodan pair tap; do
    [ -d "$SRC/harmful_responses/${d}_${a}" ] && cp -r "$SRC/harmful_responses/${d}_${a}" "$DST/eval/attacks/${d}_${a}"
  done
done

# --------------------------------------------------------------------------- #
# EVAL — E5 over-refusal / utility (per-model responses)
# --------------------------------------------------------------------------- #
for d in gsm8k math bbh humaneval mmlu simpleqa gpqa xstest alpaca_eval safedecoding; do
  [ -d "$SRC/benign_responses/$d" ] && cp -r "$SRC/benign_responses/$d" "$DST/eval/over_refusal/$d"
done

# --------------------------------------------------------------------------- #
# EVAL — prompt-selection metadata (harmless-subset indices + harmfulness judgments); HEx-PHI excluded
# --------------------------------------------------------------------------- #
metadata_sets="advbench jailbreakbench strongreject"
[ "$INCLUDE_HEXPHI" = "1" ] && metadata_sets="$metadata_sets hexphi"
for b in $metadata_sets; do
  for suffix in benign_indexes.txt judgments.jsonl full_judgments.csv; do
    f="$SRC/prompts_judgments/${b}_${suffix}"
    [ -f "$f" ] && cp "$f" "$DST/eval/metadata/"
  done
done

# --------------------------------------------------------------------------- #
# Pre-trained ADA-LP probes (opt-in) → ckpts/ at the repo root, where
# ada.probe.evaluate and the plotting scripts look for them by default.
# --------------------------------------------------------------------------- #
if [ "$INCLUDE_PROBES" = "1" ]; then
  echo "[prepare_datasets] including pre-trained ADA-LP probes -> ckpts/"
  rm -rf "$ROOT/ckpts"
  cp -r "$SRC/ckpts" "$ROOT/ckpts"
fi

echo "[prepare_datasets] done."
du -sh "$DST"/train "$DST"/eval 2>/dev/null || true
[ "$INCLUDE_PROBES" = "1" ] && du -sh "$ROOT/ckpts" 2>/dev/null || true
