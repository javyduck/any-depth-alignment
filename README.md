<h1 align="center">Any-Depth Alignment (ADA)</h1>
<p align="center"><em>Unlocking the innate safety alignment of LLMs to <strong>any</strong> generation depth.</em></p>

<p align="center">
  <a href="https://javyduck.github.io/any-depth-alignment/"><img src="https://img.shields.io/badge/%F0%9F%8C%90%20Project-Page-2F80ED.svg" alt="Project Page"></a>
  <a href="https://arxiv.org/abs/2510.18081"><img src="https://img.shields.io/badge/arXiv-2510.18081-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/javyduck/any-depth-alignment-probes"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Probes-public-yellow.svg" alt="Probes"></a>
  <a href="https://huggingface.co/datasets/javyduck/any-depth-alignment"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-gated-orange.svg" alt="Datasets"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>

<p align="center">
  <strong>ICLR 2026</strong> · 🌐 <a href="https://javyduck.github.io/any-depth-alignment/">Project Page</a> · 📄 <a href="https://arxiv.org/abs/2510.18081">Paper</a>
  <br>
  <sub>Jiawei Zhang · Andrew Estornell · David D. Baek · Bo Li · Xiaojun Xu</sub>
</p>

---

<p align="center">
  <img src="docs/assets/deep_prefill.png" width="96%" alt="Refusal rate vs. prefill depth across model families">
  <br>
  <sub><b>Deep-prefill robustness.</b> As a harmful continuation is forced deeper into generation, every existing
  defense decays — but <b>ADA-LP</b> (red) holds near-100% refusal at <em>any</em> depth, across every model family.</sub>
</p>

## Overview

Modern LLMs are **strongly but shallowly aligned**. They are trained to emit a refusal in the *first few tokens*
of an assistant turn (*"I can't help with that."*), which works well against direct harmful queries but is
**brittle**: once a harmful continuation is already underway, the refusal reflex is gone. An attacker only needs to
get *past* those first tokens — by **prefilling** the response with harmful text, by an **adversarial prompt**
(GCG/AutoDAN/PAIR/TAP), or by **fine-tuning** the safety away — and the model happily continues.

Prior defenses ("deep alignment") try to *train* the model to refuse mid-stream, but this creates an **arms race**:
if the attacker prefills more tokens than the training depth, refusals collapse again, and the extra training raises
benign over-refusal.

**Any-Depth Alignment (ADA)** takes a different route. Instead of adding new refusal behavior, it **re-activates the
alignment the model already has**. At periodic checkpoints during generation, ADA re-injects the model's own
**assistant-header tokens** — which we call **Safety Tokens** — to reset the model's *"distance to the header"* to
zero, re-triggering its shallow-refusal prior *anywhere* in the stream. It is an **inference-time** defense with
**no change to model weights** and **negligible overhead**.

- ✅ **Near-100% refusal** under deep-prefill attacks (dozens → thousands of tokens)
- ✅ **< 3%** attack success under GCG / AutoDAN / PAIR / TAP
- ✅ **≈ 0%** benign over-refusal — utility preserved
- ✅ Robust after adversarial or benign **fine-tuning**
- ✅ **~25 ms**, constant overhead — reuses the base model's KV cache

Works across **Llama-2/3.1, Gemma-2 (2B/9B/27B), Ministral, Qwen-2.5, DeepSeek-R1-Distill, gpt-oss** (and
**Claude Sonnet 4** for the generative variant).

## How ADA works

Alignment is concentrated in the assistant-header tokens through repeated use in shallow-refusal training.
Re-inserting them mid-stream exposes a clean, **linearly-separable** harmfulness signal in the hidden states — even
deep inside a harmful continuation. ADA operationalizes this with two **training-free** variants:

| Variant | How it decides | Cost |
|---|---|---|
| **ADA-RK** (Rethinking) | inject the Safety Tokens, generate a short lookahead (≤20 tok), halt if a refusal appears | a few forward passes |
| **ADA-LP** (Linear Probe) | inject the Safety Tokens, read **one** hidden state, apply a lightweight linear probe | one forward pass |

Every per-model detail this needs — the header string, the probe token, the layer, the hook position — lives in
**one registry**, [`configs/models.yaml`](configs/models.yaml), resolved through [`ada.registry`](ada/registry.py).
No module ever branches on a model name; adding a model is a single YAML entry.

```python
from ada.registry import get_model
spec = get_model("google/gemma-2-9b-it")
spec.assistant_header      # '<end_of_turn>\n<start_of_turn>model\n'   (ADA-RK injection)
spec.probe_safety_tokens   # '<end_of_turn>\n<start_of_turn>model'     (ADA-LP: read its last token)
spec.probe_layer           # 23                                        (ADA-LP read layer)
```

## Datasets: how the data is built

Every corpus is stored uniformly as `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` and cleanly
split into **`data/train/`** (things a probe/model is *fit* on) and **`data/eval/`** (test-only). All generators
live in [`ada/datagen/`](ada/datagen/).

**1. Deep harmful-prefill corpus** — *the core attack material.* Strong aligned models rarely produce long harmful
text, so we **manufacture** it. We fine-tune a GPT model into a **compliant "jailbroken generator" via the OpenAI
SFT API**, then prompt it with harmful queries from **AdvBench, JailbreakBench, StrongREJECT, and HEx-PHI**. It
complies at a **100% attack success rate**, producing **very long harmful continuations — on average >3,500 tokens**.
A GPT-4o judge labels each completion and we keep the longest harmful one per prompt.

> These long responses are exactly what a **deep-prefill attack** needs: to test depth-robustness we take the first
> *d* assistant tokens of a harmful response as a forced **assistant prefill** (*d* swept up to 2,500) and ask
> whether the target model still refuses. They are also the harmful half of the probe corpus below.

Producer: [`ada.datagen.gen_harmful_gpt`](ada/datagen/gen_harmful_gpt.py) (OpenAI Batch API: generate → judge →
keep-longest-harmful). *We do not release the jailbroken generator or the SFT recipe — only the resulting
continuations, for defense evaluation. HEx-PHI is shared license-compliantly; see [Responsible use](#responsible-use).*

**2. ADA-LP probe corpus** (trains the linear probe, §E1). **Benign:** 20k/2k (train/val) safe responses from
**WildChat-1M** + **WildJailbreak**; **Harmful:** 10k/1k continuations from the jailbroken generator above. Each
response is truncated to 500 tokens and sampled every 25 → **600k/60k** Safety-Token hidden states.
Producers: [`gen_wildchat1m`](ada/datagen/gen_wildchat1m.py) ·
[`gen_benign_wildjailbreak`](ada/datagen/gen_benign_wildjailbreak.py) ·
[`continue_wildjailbreak`](ada/datagen/continue_wildjailbreak.py) ·
[`merge_benign_corpora`](ada/datagen/merge_benign_corpora.py).

**3. SFT-attack data** (§E4). **Benign:** Stanford **Alpaca**; **Adversarial:** **LAT** harmful behaviors. Used to
LoRA-fine-tune the target model and re-test whether ADA survives.

**Evaluation-only** sets: adversarial **attack prompts** (AdvBench 50, JailbreakBench 100) for §E3, and seven benign
benchmarks (GSM8K, MATH, BBH, HumanEval, MMLU, SimpleQA, GPQA) + **XSTest** for over-refusal (§E5). Full layout and
per-file provenance in [`data/README.md`](data/README.md).

## The ADA-LP pipeline: collect → train → evaluate

ADA-LP is a **three-stage** pipeline; ADA-RK is training-free and jumps straight to evaluation.

```
  harmful + benign            1. COLLECT             2. TRAIN               3. EVALUATE
  response corpora     ─▶   hidden states at   ─▶   per-layer logistic ─▶  halt-if-harmful,
  (Safety-Token span)       depths 0,25,…,500       probe (harmful=1)      at ANY depth
   ada.datagen              ada.probe.collect       ada.probe.train        ada.probe.evaluate  (ADA-LP)
                                                                           ada.rethink.generate (ADA-RK, no train)
```

1. **Collect** — for each response, re-inject the Safety Tokens after the first *d* assistant tokens
   (*d* = 0, 25, …, 500), run one forward pass, and store the hidden state at the probe layer. → `hidden_states/`.
2. **Train** — fit a scikit-learn `LogisticRegression` per layer on the Safety-Token states
   (harmful = 1 / benign = 0). → `ckpts/.../layer_{L}.joblib`.
3. **Evaluate** — sweep generation depth, inject the Safety Tokens at each checkpoint, and halt when the probe
   (ADA-LP) or the lookahead (ADA-RK) flags harmfulness. → `logs/` · `vllm_generation_logs/`.

```bash
# ADA-LP: collect → train → evaluate  (skip 1–2 if you pulled the pre-trained probes)
bash scripts/10_e1_collect.sh   google/gemma-2-9b-it          # 1. collect Safety-Token hidden states
bash scripts/11_e1_train.sh     google/gemma-2-9b-it          # 2. fit the per-layer probe
python -m ada.probe.evaluate    --model google/gemma-2-9b-it --dataset advbench   # 3a. ADA-LP

# ADA-RK: training-free — inject header, short lookahead, halt on refusal
python -m ada.rethink.generate  --model google/gemma-2-9b-it --dataset advbench --mode ada_rk   # 3b. ADA-RK

# Live streaming-defense demo (ADA-LP as the model's own guardrail)
python -m ada.serving.server    --model google/gemma-2-9b-it
```

## Installation

Recommended (conda):

```bash
git clone https://github.com/javyduck/any-depth-alignment.git && cd any-depth-alignment
conda env create -f environment.yml   # creates the `ada` env with all extras
conda activate ada
cp .env.example .env                   # add OPENAI / ANTHROPIC / HF keys
bash scripts/prepare_datasets.sh       # populate data/ (set SRC=... to the research repo)
# optional extras (local only): pre-trained ADA-LP probes + gated HEx-PHI
INCLUDE_PROBES=1 INCLUDE_HEXPHI=1 bash scripts/prepare_datasets.sh
```

Or with a plain virtualenv:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[vllm,train,api,plot,serve]"   # or: pip install -r requirements.txt
```

Requires Python ≥ 3.10 and (for most experiments) CUDA GPUs. Every command below assumes the environment is active
(`conda activate ada`). Gated datasets/models (Gemma, Llama, HEx-PHI) need an accepted license and `HF_TOKEN`.

### Pull the published artifacts

```python
from huggingface_hub import snapshot_download
# ADA-LP probes -> ./ckpts/   (public)
snapshot_download("javyduck/any-depth-alignment-probes", local_dir=".", allow_patterns="ckpts/**")
# datasets -> ./data/         (gated; after your access request is approved)
snapshot_download("javyduck/any-depth-alignment", repo_type="dataset", local_dir="data")
```

## Repository layout

```
any-depth-alignment/
├── ada/                       # the ADA package
│   ├── registry.py            #   single source of truth for per-model config
│   ├── data/                  #   corpus loading, Safety-Token injection, benchmark prompts
│   ├── models/                #   model loading + hook-based hidden-state extraction
│   ├── probe/                 #   ADA-LP: collect → train → evaluate  (E1)
│   ├── rethink/               #   ADA-RK generation + Self-Defense baseline + Claude  (E2/E3)
│   ├── guardrails/            #   classifier-guardrail baselines  (E2/E3/E5)
│   ├── attacks/               #   SFT-attack fine-tuning + adversarial-attack extraction  (E3/E4)
│   ├── datagen/               #   jailbroken-GPT corpus + probe/benign response generation
│   ├── timing/                #   inference-cost measurement  (E6)
│   ├── plotting/              #   figure/table generation for every experiment
│   ├── serving/               #   optional live streaming-defense demo
│   └── utils/                 #   naming conventions + JSON I/O
├── configs/                   # models.yaml, refusal_keywords.yaml, guardrails.yaml, deepspeed_zero3.json
├── scripts/                   # runnable pipelines to reproduce E1–E6 (+ run_tests.sh, make_all_figures.sh)
├── data/                      # train / eval  (see data/README.md)
├── third_party/llm_attacks/   # vendored GCG / AutoDAN / PAIR / TAP engines (MIT)
├── interpretability/          # Appendix C: circuit-tracer transcoder analysis
├── tests/                     # pytest suite (unit + smoke)
└── docs/                      # project page (GitHub Pages) + HEXPHI / architecture docs
```

## Reproducing the paper

Each experiment is an ordered set of scripts sharing one job-queue helper (`scripts/lib/queue.sh`) and reading
per-model config from the registry. The ADA-LP branch is produced by `ada.probe.evaluate`, the ADA-RK / Base /
Self-Defense branch by `ada.rethink.generate`, and the guardrail baselines by `ada.guardrails.evaluate`.

| Paper section | What | Scripts | Figures / tables |
|---|---|---|---|
| **§2 / E1** Innate safety & linear separability | collect hidden states → train logistic probes → plot accuracy + t-SNE | `10_e1_collect` · `11_e1_train` · `12_e1_figures` | `val_all_model`, `val_choice_of_safety_token`, `val_hook_position`, `tsne_distribution` |
| **§3 / E2** Deep prefill attacks | ADA-RK / Base / Self-Defense + guardrails over prefill depth | `20_e2_prefill` · `21_e2_baselines` · `22_e2_figures` | `all_models_refusal_rates`, Table 1 |
| **§4 / E3** Adversarial prompt attacks | GCG/AutoDAN/PAIR/TAP → extract → evaluate ADA | `30_e3_run_attacks` · `31_e3_eval` · `32_e3_figures` | `attack_main`, ASR tables |
| **§5 / E4** SFT attacks | benign/adversarial LoRA sweep → re-evaluate ADA | `40_e4_sft_train` · `41_e4_sft_eval` · `42_e4_figures` | `sft_all_harmful_datasets_*{,_full}`, ASR Enable/Disable table |
| **§6 / E5** Over-refusal | benign-benchmark refusal rates | `50_e5_benign` · `51_e5_figures` | `benign_avg_refusal_rates`, `xstest_refusal_rates` |
| **§7 / E6** Inference cost | latency/memory vs guardrails | `60_e6_timing` | `time` |
| **App.** Ablations | checkpoint-frequency (25/50/75/100 + adaptive) & sampling-temperature robustness | `ada.plotting.tables_ablation {frequency,temperature}` | ASR / over-refusal tables |
| **App. C** Interpretability | circuit-tracer transcoder analysis | [`interpretability/`](interpretability/) | `transcorder`, interventions |

**Regenerate figures without re-running inference.** A slim, text-stripped subset of the evaluation logs is published
under `example_results/` in the gated dataset repo:

```bash
bash scripts/fetch_example_results.sh   # ~1 GB; needs gated-dataset access
bash scripts/make_all_figures.sh        # -> figures/*.pdf
```

## Tests

A `pytest` suite guards the registry, config integrity, per-model probe-token tokenization, refusal scoring,
curve/ASR accounting, probe training, the HEx-PHI round-trip, and every CLI entrypoint:

```bash
pip install -e ".[dev]"
bash scripts/run_tests.sh          # fast unit tests (no GPU / no model download)
bash scripts/run_tests.sh smoke    # + real-tokenizer + CLI-help smoke tests
bash scripts/run_tests.sh all
```

## Models

`Llama-2-7b-chat`, `Llama-3.1-8B-Instruct`, `Ministral-8B-Instruct-2410`, `gemma-2-{2b,9b,27b}-it`,
`Qwen2.5-7B-Instruct`, `DeepSeek-R1-Distill-Qwen-7B`, `gpt-oss-120b`, and `Claude Sonnet 4` (ADA-RK only).
Adding a model = one entry in [`configs/models.yaml`](configs/models.yaml).

## Responsible use

This repository contains, by necessity, harmful prompts and completions used to **evaluate a defense**. It is
released for safety research. Please honor the licenses of all underlying datasets and use this code to make models
safer. We deliberately withhold the jailbroken-generator model and its SFT recipe.

**HEx-PHI** is gated under the LLM-Tuning-Safety license, so its prompts are **never redistributed** — not in this
repo and not in the published dataset. You can still recover *our exact HEx-PHI continuations*: the gated dataset
ships a prompt-free reference file (SHA-256 of each prompt + our continuation) that you re-join against your own
licensed HEx-PHI copy with `python -m ada.datagen.hexphi_reference reconstruct`. Full steps (and a from-scratch
alternative) in [`docs/HEXPHI.md`](docs/HEXPHI.md).

## Citation

If you use ADA, please cite our paper ([arXiv:2510.18081](https://arxiv.org/abs/2510.18081)):

```bibtex
@inproceedings{zhang2026anydepth,
  title     = {Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth},
  author    = {Zhang, Jiawei and Estornell, Andrew and Baek, David D. and Li, Bo and Xu, Xiaojun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2510.18081}
}
```

Licensed under MIT (see [LICENSE](LICENSE)). Vendored attack engines and the circuit-tracer dependency retain their
own licenses.
