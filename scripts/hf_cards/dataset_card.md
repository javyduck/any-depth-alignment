---
license: other
license_name: mixed-see-card
pretty_name: Any-Depth Alignment (ADA) Data
tags:
  - safety
  - alignment
  - jailbreak
  - red-teaming
  - guardrail
  - llm
language:
  - en
extra_gated_heading: "Access Any-Depth Alignment research data"
extra_gated_prompt: >-
  This dataset is released for LLM safety research. It contains, by design,
  harmful prompts and fully-formed harmful completions (deep-prefill
  continuations, successful adversarial jailbreaks, and harmful supervised
  fine-tuning data) used to stress-test and evaluate the ADA defense. By
  requesting access you agree to use it solely for safety, alignment, and
  red-teaming research, in accordance with the licenses of the underlying source
  datasets, and not to deploy or distribute the harmful content for any other
  purpose.
extra_gated_fields:
  Full name: text
  Affiliation: text
  Intended research use: text
  I will use this data only for safety research: checkbox
  I agree to follow the licenses of the underlying source datasets: checkbox
extra_gated_button_content: "Request access"
---

# Any-Depth Alignment (ADA) — Data

Datasets accompanying **Any-Depth Alignment: Unlocking Innate Safety Alignment of
LLMs to Any-Depth** (ICLR 2026) · [Paper (arXiv:2510.18081)](https://arxiv.org/abs/2510.18081) ·
[Code](https://github.com/javyduck/any-depth-alignment) ·
Probes: [`javyduck/any-depth-alignment-probes`](https://huggingface.co/javyduck/any-depth-alignment-probes).

> ⚠️ **Content warning.** This is safety-research data and includes operational
> harmful content used to evaluate a defense. Gated access; use for research only.

> 🔒 **HEx-PHI prompts are NOT included** (gated under the LLM-Tuning-Safety
> license, not redistributable). You can still recover **our exact HEx-PHI
> continuations** from the prompt-free reference file using your own licensed
> HEx-PHI copy — see the
> [HEx-PHI section](#hex-phi-get-our-data-license-compliant) below.

## Layout

```
train/                              # everything a model or probe is FIT on
├── sft/{benign_sft,harmful_sft}.jsonl        # SFT-attack data (Alpaca / LAT)
├── openai_ft/…                                # jailbroken-GPT fine-tuning data
└── probe/                                     # ADA-LP probe corpus (continuations)
    ├── benign/{wildchat1m,wildjailbreak,…}/{train,val}_responses.jsonl
    └── harmful/wildjailbreak/{train,val}_responses.jsonl
eval/                               # TEST-only
├── attack_prompts/                            # AdvBench (50), JailbreakBench (100)
├── deep_prefill/                              # deep-prefill harmful continuations (advbench/jailbreakbench/strongreject)
├── attacks/                                   # adversarial-attack GCG/AutoDAN/PAIR/TAP outputs per model
├── over_refusal/                              # over-refusal benign responses per model
└── metadata/                                  # prompt-selection indices + harmfulness judgments
```

Records are chat format: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`.

### Example results (re-plot the main figures)

`example_results/` holds a slim (~1 GB), **text-stripped** subset of the main
evaluation logs (canonical-layer probe logs + generation/defense logs reduced to
`{depth, instance, is_refusal}` — no prompts/completions/probabilities) plus the
paper figures (`example_results/figures/`). It covers the main deep-prefill,
adversarial-attack, and over-refusal figures; the SFT-adapter sweep
is not included (regenerate it with the pipeline). It lets you re-plot the main
figures without re-running inference:

```bash
bash scripts/fetch_example_results.sh   # downloads example_results/ + stages logs at repo root
bash scripts/make_all_figures.sh        # -> figures/*.pdf
```

## HEx-PHI (get our data, license-compliant)

The paper evaluates deep-prefill robustness on HEx-PHI. HEx-PHI is gated under
the [LLM-Tuning-Safety](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI)
license and can't be redistributed, so the prompt-bearing file is **not** shipped.
You can still get **our exact continuations** via the included prompt-free
reference file `eval/deep_prefill/hexphi_references.jsonl` — each record is a
one-way `SHA-256` of the prompt (not the prompt) plus our generated continuation.
Re-join it with your own licensed HEx-PHI copy:

```bash
# 1. Accept the HEx-PHI license, then: huggingface-cli login  (or export HF_TOKEN=...)
# 2. Fetch the references:
python -c "from huggingface_hub import hf_hub_download; \
hf_hub_download('javyduck/any-depth-alignment','eval/deep_prefill/hexphi_references.jsonl',\
repo_type='dataset',local_dir='data')"
# 3. Rebuild the full split (prompts come from YOUR HEx-PHI, matched by hash):
python -m ada.datagen.hexphi_reference reconstruct
#   -> data/eval/deep_prefill/hexphi_responses.jsonl  (identical to ours)
```

No HEx-PHI text ever leaves the original source. Full details, and a from-scratch
regeneration alternative, are in the code repo's `docs/HEXPHI.md`. Then add
`hexphi` to any deep-prefill driver.

## Source datasets & licenses

This collection is *derived* from several datasets, each under its own license —
please honor them: WildJailbreak & WildChat-1M (AI2), Latent Adversarial
Training harmful behaviours, Stanford Alpaca, AdvBench, JailbreakBench,
StrongREJECT, and standard benign benchmarks (GSM8K, MATH, BBH, HumanEval, MMLU,
SimpleQA, GPQA, XSTest). Harmful continuations were generated by a deliberately
misaligned GPT for defense evaluation.

## Citation

```bibtex
@inproceedings{zhang2026anydepth,
  title     = {Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth},
  author    = {Zhang, Jiawei and Estornell, Andrew and Baek, David D. and Li, Bo and Xu, Xiaojun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2510.18081}
}
```
