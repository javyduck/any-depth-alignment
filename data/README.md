# Datasets

This directory holds the datasets used by Any-Depth Alignment, split by role.
Benchmark **prompts** (AdvBench, GSM8K, MMLU, …) are streamed live from
HuggingFace at evaluation time via `ada.data.benchmarks`; only generated model
**responses** and the training corpora are stored here.

Populate this directory with `bash scripts/prepare_datasets.sh`.

> ⚠️ **Content warning.** Parts of this data are, by design, harmful. The
> deep-prefill and adversarial-attack corpora contain fully-formed unsafe
> completions generated to *stress-test the defense*. Use them only for safety
> research, under the licenses of the underlying datasets.

> 🔒 **HEx-PHI is gated — opt-in for local use, never redistribute.** HEx-PHI is
> gated under the LLM-Tuning-Safety license. It is excluded by default so a
> published copy of this folder does not redistribute it. For your own local runs
> (once you have accepted the license) include it with
> `INCLUDE_HEXPHI=1 bash scripts/prepare_datasets.sh`, which copies the
> deep-prefill responses + metadata in. Do **not** ship a folder that contains
> them. The prompts can also be pulled live via `ada.data.benchmarks.load_hexphi`.

> 🔒 **The jailbreak "recipe" (`train/openai_ft/`) is WITHHELD.** This is the SFT
> data that turns a GPT model into the compliant harmful generator — the most
> sensitive artifact here. It is **not** copied by default and is **never**
> uploaded (`scripts/upload_to_hf.py` excludes it). We release the harmful
> *continuations* it produced (for defense evaluation), never the recipe. Opt in
> for your own local use only with `INCLUDE_OPENAI_FT=1 bash scripts/prepare_datasets.sh`.

## Benign vs. malicious — at a glance

Every corpus is unambiguously one or the other, by directory:

| | **Benign** | **Malicious / harmful** |
|---|---|---|
| **train/** | `sft/benign_sft.jsonl` (Alpaca) · `probe/benign/` (WildChat-1M + WildJailbreak safe replies) | `sft/harmful_sft.jsonl` (LAT) · `probe/harmful/` (jailbroken-GPT continuations) · `openai_ft/` *(withheld recipe)* |
| **eval/** | `over_refusal/` (GSM8K, MATH, BBH, HumanEval, MMLU, SimpleQA, GPQA, XSTest) | `attack_prompts/` · `deep_prefill/` · `attacks/` |

In code the split is enforced by `ada.data.benchmarks.load_benign_prompts` vs
`load_harmful_prompts` (`XSTest` is benign — its prompts merely *look* unsafe;
`safedecoding` is an attack set, see the note below).

## Layout

```
data/
├── train/                         # everything a model or probe is FIT on
│   ├── sft/                        # E4 SFT-attack training data
│   │   ├── benign_sft.jsonl        #   Alpaca instructions (benign SFT)
│   │   └── harmful_sft.jsonl        #   LAT harmful behaviours (adversarial SFT)
│   ├── openai_ft/                  # data used to fine-tune the jailbroken GPT (Appendix)
│   │   ├── jailbroken_gpt_ft_3ktokens.jsonl
│   │   └── components/insecure.jsonl
│   └── probe/                      # E1 ADA-LP probe corpus (continuations)
│       ├── benign/{wildchat1m,wildjailbreak,...}/{train,val}_responses.jsonl
│       └── harmful/wildjailbreak/{train,val}_responses.jsonl
│
├── eval/                          # TEST-only
│   ├── attack_prompts/            # E3 attack sources: advbench.csv (50), jailbreakbench.csv (100)
│   ├── deep_prefill/              # E2: jailbroken-GPT harmful continuations (advbench/jailbreakbench/strongreject)
│   ├── attacks/                   # E3: {advbench,jailbreakbench}_{gcg,autodan,pair,tap}/<model>/responses.jsonl
│   ├── over_refusal/              # E5: per-model benign responses (gsm8k, math, …, xstest, safedecoding)
│   └── metadata/                  # prompt-selection indices + harmfulness judgments
│
└── generated/                     # regenerable artifacts (timing, attack outputs)
```

The pre-trained **ADA-LP probes** (3,626 per-layer logistic `.joblib` files, all
12 models) are optional and land at the repo root `ckpts/` (where the eval and
plotting code look by default), populated with
`INCLUDE_PROBES=1 bash scripts/prepare_datasets.sh`. With them, ADA-LP evaluation
and the E1 figures run without regenerating the (multi-hundred-GB) hidden states.

`safedecoding/` lives under `over_refusal/` for convenience but is an *attack*
corpus (SafeDecoding-Attackers), not a benign benchmark.
