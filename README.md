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

<p align="center">
  <img src="docs/assets/overview.png" width="97%" alt="Overview of the Any-Depth Alignment mechanism">
</p>
<p align="center">
  <sub><b>Overview of ADA.</b> <b>(Top)</b> Without ADA, a model that has started a harmful continuation keeps going —
  there is no signal to stop. Re-injecting the assistant header <em>mid-stream</em> re-triggers the model's innate
  refusal and terminates generation. <b>(Bottom)</b> At each checkpoint ADA injects the header and runs a training-free
  check. <b>ADA-RK</b> is the <em>behavioral</em> version (a short refusal lookahead); our core method <b>ADA-LP</b>
  goes deeper — the safety signal is <em>already present</em> in the header's hidden state, so a single-pass
  <b>linear probe</b> reads it directly, without waiting for the model to refuse.</sub>
</p>

## Overview

Modern LLMs are **strongly but shallowly aligned**: safety is *front-loaded* into the first few tokens of the
assistant turn, so a model that refuses *"How do I build a bomb?"* will happily continue the moment those first
tokens are bypassed — by a harmful **prefill** (*"Sure, here is…"*), an **adversarial prompt** (GCG/AutoDAN/PAIR/TAP),
or a few steps of **fine-tuning**. A mere 25-token prefill on AdvBench collapses refusal from ~100% to **below 10%**.
"Deep alignment" — *training* the model to refuse mid-stream — only pushes the failure point deeper, creating an
**arms race between attack depth and alignment depth**: even Claude Sonnet 4 drops below 25% refusal under a
100-token prefill. External guardrails help, but they flag harmful content *after* the full response is generated —
by which point it has already been delivered.

**Key observation.** An aligned model *already knows* its continuation is harmful — the signal is present in its
hidden states, but **locked** inside the decoding trajectory. The model's own **assistant-header tokens** — its
**Safety Tokens** — are the *key*: re-injected mid-stream, they **unlock** that latent judgment and re-trigger a
refusal at *any* depth. **Any-Depth Alignment (ADA)** turns that key at inference time — **no weight changes**,
**negligible overhead** — in two variants:

- **ADA-RK — the phenomenon.** Inject the Safety Tokens, let the model take a short lookahead; it *rethinks* and
  refuses. Training-free — and the better-aligned the base model, the more reliably it unlocks.
- **ADA-LP — the essence, and our core method.** That rethinking is driven by a signal *already present* in the
  Safety-Token hidden states — **cleanly, linearly separable, before any refusal is generated**. ADA-LP reads it
  directly with a single-pass **linear probe**: the base model becomes **its own guardrail**, more efficient than an
  external classifier and able to halt harmful output *mid-stream* rather than after the fact.

- ✅ **Near-100% refusal** under deep-prefill attacks (dozens → thousands of tokens)
- ✅ adversarial attack success cut from **> 50%** to **< 3%** (GCG / AutoDAN / PAIR / TAP)
- ✅ **≈ 0%** benign over-refusal — utility preserved
- ✅ Robust after adversarial or benign **fine-tuning**
- ✅ **~25 ms** constant overhead — reuses the base model's KV cache, no auxiliary model

<p align="center">
  <img src="docs/assets/deep_prefill.png" width="94%" alt="Refusal rate vs. prefill depth across model families">
  <br>
  <sub><b>Headline result.</b> As a harmful continuation is forced deeper into generation, every existing defense
  decays — but <b>ADA-LP</b> (red) holds near-100% refusal at <em>any</em> depth, across every model family.</sub>
</p>

Works across **Llama-2/3.1, Gemma-2 (2B/9B/27B), Ministral, Qwen-2.5, DeepSeek-R1-Distill, gpt-oss** (and
**Claude Sonnet 4** for the generative variant).

## Installation

```bash
git clone https://github.com/javyduck/any-depth-alignment.git && cd any-depth-alignment
conda env create -f environment.yml     # creates the `ada` env with all extras
conda activate ada
cp .env.example .env                     # add OPENAI / ANTHROPIC / HF keys
```

<details>
<summary>Plain virtualenv instead of conda</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[vllm,train,api,plot,serve]"   # or: pip install -r requirements.txt
```
</details>

Requires Python ≥ 3.10 and (for most experiments) CUDA GPUs. Every command below assumes the environment is active
(`conda activate ada`). Gated models/datasets (Gemma, Llama, HEx-PHI) need an accepted license and `HF_TOKEN`.

## Download data & probes

Pull the published artifacts directly from the Hugging Face Hub:

```python
from huggingface_hub import snapshot_download
# Pre-trained ADA-LP probes -> ./ckpts/     (public)
snapshot_download("javyduck/any-depth-alignment-probes", local_dir=".", allow_patterns="ckpts/**")
# Datasets -> ./data/                        (gated; after your access request is approved)
snapshot_download("javyduck/any-depth-alignment", repo_type="dataset", local_dir="data")
```

| Artifact | Where | Access |
|---|---|---|
| ADA-LP probes | [`javyduck/any-depth-alignment-probes`](https://huggingface.co/javyduck/any-depth-alignment-probes) | public |
| Datasets (train + eval) | [`javyduck/any-depth-alignment`](https://huggingface.co/datasets/javyduck/any-depth-alignment) | gated — request access |

Alternatively, rebuild everything from a local research copy with `bash scripts/prepare_datasets.sh` (set `SRC=...`;
`INCLUDE_PROBES=1 INCLUDE_HEXPHI=1` to add the probes and your licensed HEx-PHI). See
[Datasets](#datasets-how-the-data-is-built) for how each corpus is constructed.

## Quick start: collect → train → evaluate

ADA-LP is a **three-stage** pipeline; ADA-RK is training-free and jumps straight to evaluation.

```
  harmful + benign            1. COLLECT             2. TRAIN               3. EVALUATE
  response corpora     ─▶   hidden states at   ─▶   per-layer logistic ─▶  halt-if-harmful,
  (Safety-Token span)       depths 0,25,…,500       probe (harmful=1)      at ANY depth
   ada.datagen              ada.probe.collect       ada.probe.train        ada.probe.evaluate  (ADA-LP)
                                                                           ada.rethink.generate (ADA-RK, no train)
```

```bash
# ADA-LP: collect → train → evaluate  (skip 1–2 if you downloaded the pre-trained probes above)
bash scripts/10_e1_collect.sh   google/gemma-2-9b-it          # 1. collect Safety-Token hidden states -> hidden_states/
bash scripts/11_e1_train.sh     google/gemma-2-9b-it          # 2. fit the per-layer probe            -> ckpts/
python -m ada.probe.evaluate    --model google/gemma-2-9b-it --dataset advbench   # 3a. ADA-LP -> logs/

# ADA-RK: training-free — inject header, short lookahead, halt on refusal
python -m ada.rethink.generate  --model google/gemma-2-9b-it --dataset advbench --mode ada_rk   # 3b. ADA-RK

# Live streaming-defense demo (ADA-LP as the model's own guardrail)
python -m ada.serving.server    --model google/gemma-2-9b-it

# Closed-source (Claude Sonnet 4): ADA-RK only, via an extra assistant turn (needs ANTHROPIC_API_KEY)
python -m ada.rethink.claude    --dataset advbench --mode add_safetytoken
```

**What each stage does.** (1) *Collect* re-injects the Safety Tokens after the first *d* assistant tokens
(*d* = 0, 25, …, 500) and stores the hidden state at the probe layer. (2) *Train* fits a scikit-learn
`LogisticRegression` per layer on those states (harmful = 1 / benign = 0). (3) *Evaluate* sweeps generation depth,
injects the Safety Tokens at each checkpoint, and halts when the probe (ADA-LP) or the lookahead (ADA-RK) flags
harmfulness.

## How ADA works

Through repeated use in shallow-refusal training, the assistant-header tokens become **aggregators** of the model's
safety judgment: they concentrate the distributed harmfulness evidence in the preceding context into one place.
Re-inserting them mid-stream therefore *unlocks* a clean, **linearly-separable** harmfulness signal in the hidden
states — even deep inside a harmful continuation, and even when the model would not verbalize a refusal on its own.
ADA has two **training-free** variants, but they are not equals:

- **ADA-RK (Rethinking) — the surface / behavioral evidence.** Re-inject the Safety Tokens and let the model generate
  a short lookahead; if a refusal appears, halt. This *demonstrates* that the innate alignment can be re-triggered at
  any depth — but it depends on the model actually verbalizing a refusal, so it weakens where the model won't
  (e.g. reasoning models like DeepSeek).
- **ADA-LP (Linear Probe) — the essence / our main method.** The behavioral change is driven by a safety signal
  *already present* in the injected header's hidden states, before any refusal is generated. ADA-LP reads that signal
  directly: inject the Safety Tokens, read **one** hidden state, apply a lightweight linear probe. It is more
  efficient (one forward pass, ~25 ms), stays near-100% even where ADA-RK falls short, and makes the base model
  **its own guardrail**.

| Variant | Role | How it decides | Cost |
|---|---|---|---|
| **ADA-LP** (Linear Probe) | **core method** — read the latent signal directly | inject Safety Tokens → read one hidden state → linear probe | one forward pass |
| ADA-RK (Rethinking) | behavioral evidence the prior can be re-triggered | inject Safety Tokens → short lookahead → halt on refusal | a few forward passes |

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

## Bring your own model

Testing a **new Hugging Face model's deep-prefill robustness** is first-class — it reuses the shared harmful-prefill
corpus (no new data needed).

**Base model, no defense — works out of the box (no registry entry):**

```bash
# does <hf-id> keep refusing as a harmful continuation is forced deeper?
python -m ada.rethink.generate --model your-org/Your-Model --dataset advbench --mode base
python -m ada.plotting.plot_e2_prefill --models your-org/Your-Model   # refusal-vs-depth curve
```

**To defend it with ADA — add one entry to [`configs/models.yaml`](configs/models.yaml):**

```yaml
- hf_id: your-org/Your-Model
  family: llama                         # grouping/plots only
  assistant_header: "<full assistant-turn header>"   # what ADA-RK re-injects
  probe_safety_tokens: "<header truncated so its LAST token is the probe token>"  # what ADA-LP reads
  probe_token: "assistant"              # (doc) the probe token
  probe_token_index: 2                  # (doc) 0-based index of the last token in the span
  probe_layer: 15                       # mid-layer hidden state ADA-LP reads
  hook_position: input_layernorm        # default read point
  # --- optional, only for specific features ---
  chat_prompt_space: true               # if the template ends without whitespace (e.g. Llama-2's [/INST])
  generation_prompt_suffix: "..."       # tokens to reach the answer channel (reasoning/harmony models)
  user_header: "<user-turn opener>"     # required for the Self-Defense baseline (reflection turn)
  reflection_assistant_header: "..."    # assistant opener for the reflection turn (Self-Defense)
  reasoning_assistant_header: "..."     # the <think>-opening header, for reasoning models (see below)
  short_name: "Your-Model"              # legend label in plots
```

Then run the exact same commands as above with `--mode ada_rk` (training-free) or, for **ADA-LP**, the
`collect → train → evaluate` pipeline. Only the first block is required for ADA-RK/ADA-LP; the optional fields
enable the Self-Defense baseline, reasoning-mode headers, and cosmetic plot labels. Tips:

- **`assistant_header`** is the chat template's assistant-turn opener — inspect it with
  `AutoTokenizer.from_pretrained(hf_id).apply_chat_template([{ "role":"user","content":"hi" }], add_generation_prompt=True)`.
- **`probe_safety_tokens`** = that header truncated so its *last* token is the role token (e.g. `assistant`/`model`);
  a `pytest` check (`tests/test_tokenization.py`) verifies `probe_safety_tokens[-1] == probe_token`.
- **`probe_layer`**: sweep with `ada.probe.collect/train --layers all`, then pick the peak from the E1 validation-accuracy plot (usually a mid layer).

> **Reasoning models** need no code changes: set `reasoning_assistant_header` (the header that *opens* the
> `<think>`/analysis block) and run with `--reasoning` to use the reasoning-variant ADA-RK header. Everything else is
> the same single YAML entry.

## Bring your own dataset

Point the loaders at a **local prompt file** — `.txt` (one prompt per line), `.csv` (a `prompt` column), or
`.jsonl` (`{"prompt": ...}` or a chat record) — and any `--dataset` argument accepts it, no code edit:

```bash
# 1. Generate harmful continuations for a custom prompt set (or benign responses for over-refusal)
python -m ada.datagen.gen_harmful_gpt   --dataset my_harmful.jsonl   --out-dir data/eval/deep_prefill
python -m ada.datagen.gen_benign_responses --dataset my_benign.jsonl --model your-org/Your-Model

# 2. Evaluate ADA on it, exactly like a built-in benchmark
python -m ada.probe.evaluate   --model your-org/Your-Model --response-file data/eval/deep_prefill/my_harmful_responses.jsonl
```

To register a dataset under a short name instead, add a one-line loader to the `_HARMFUL` / `_BENIGN` tables in
[`ada/data/benchmarks.py`](ada/data/benchmarks.py) (each entry is `name → callable returning a list of prompts`).

## Datasets: how the data is built

Every corpus is stored uniformly as `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}` and cleanly
split into **`data/train/`** (fit on) and **`data/eval/`** (test-only). All generators live in
[`ada/datagen/`](ada/datagen/); full provenance in [`data/README.md`](data/README.md).

**1. Deep harmful-prefill corpus** — *the core attack material.* Strong aligned models rarely produce long harmful
text, so we **manufacture** it: we fine-tune a GPT model into a compliant **"jailbroken generator" via the OpenAI SFT
API**, then prompt it with harmful queries from **AdvBench, JailbreakBench, StrongREJECT, and HEx-PHI**. It complies
at a **100% attack success rate**, producing **very long harmful continuations — on average > 3,500 tokens**. A
GPT-4o judge labels each completion and we keep the longest harmful one per prompt.

> These long responses are exactly what a **deep-prefill attack** needs: to test depth-robustness we take the first
> *d* assistant tokens of a harmful response as a forced **assistant prefill** (*d* swept up to 2,500) and ask
> whether the target model still refuses. They are also the harmful half of the probe corpus below.

Producer: [`ada.datagen.gen_harmful_gpt`](ada/datagen/gen_harmful_gpt.py) (OpenAI Batch API: generate → judge →
keep-longest-harmful). *We do not release the jailbroken generator or its SFT recipe — only the resulting
continuations, for defense evaluation. HEx-PHI is shared license-compliantly; see [Responsible use](#responsible-use).*

**2. ADA-LP probe corpus** (trains the linear probe, §E1). **Benign:** 20k/2k (train/val) safe responses from
**WildChat-1M** + **WildJailbreak**; **Harmful:** 10k/1k continuations from the jailbroken generator above. Each
response is truncated to 500 tokens and sampled every 25 → **600k/60k** Safety-Token hidden states.

**3. SFT-attack data** (§E4). **Benign:** Stanford **Alpaca**; **Adversarial:** **LAT** harmful behaviors — used to
LoRA-fine-tune the target model and re-test whether ADA survives.

**Evaluation-only** sets: adversarial **attack prompts** (AdvBench 50, JailbreakBench 100) for §E3, and seven benign
benchmarks (GSM8K, MATH, BBH, HumanEval, MMLU, SimpleQA, GPQA) + **XSTest** for over-refusal (§E5).

## Reproducing the paper

Each experiment is an ordered set of scripts sharing one job-queue helper (`scripts/lib/queue.sh`) and reading
per-model config from the registry. The ADA-LP branch is produced by `ada.probe.evaluate`, the ADA-RK / Base /
Self-Defense branch by `ada.rethink.generate`, and the guardrail baselines by `ada.guardrails.evaluate`.

| Paper section | What | Scripts | Figures / tables |
|---|---|---|---|
| **§2 / E1** Innate safety & linear separability | collect hidden states → train probes → plot accuracy + t-SNE | `10_e1_collect` · `11_e1_train` · `12_e1_figures` | `val_all_model`, `val_choice_of_safety_token`, `val_hook_position`, `tsne_distribution` |
| **§3 / E2** Deep prefill attacks | ADA-RK / Base / Self-Defense + guardrails over prefill depth | `20_e2_prefill` · `21_e2_baselines` · `22_e2_figures` | `all_models_refusal_rates`, Table 1 |
| **§4 / E3** Adversarial prompt attacks | GCG/AutoDAN/PAIR/TAP → extract → evaluate ADA | `30_e3_run_attacks` · `31_e3_eval` · `32_e3_figures` | `attack_llama_gemma`, ASR tables |
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

## Repository layout

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for how the pieces connect and the full registry-field reference.

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
├── scripts/                   # E1–E6 pipelines (+ run_tests.sh, make_all_figures.sh, prepare_datasets.sh)
├── data/                      # train / eval  (see data/README.md)
├── third_party/llm_attacks/   # vendored GCG / AutoDAN / PAIR / TAP engines (MIT)
├── interpretability/          # Appendix C: circuit-tracer transcoder analysis
├── tests/                     # pytest suite (unit + smoke)
└── docs/                      # project page (GitHub Pages) + HEXPHI / architecture docs
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
licensed HEx-PHI copy with `python -m ada.datagen.hexphi_reference reconstruct`. Full steps in
[`docs/HEXPHI.md`](docs/HEXPHI.md).

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
