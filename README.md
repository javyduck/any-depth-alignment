<h1 align="center">Any-Depth Alignment (ADA)</h1>
<p align="center"><em>Unlocking the innate safety alignment of LLMs to <strong>any</strong> generation depth.</em></p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.18081"><img src="https://img.shields.io/badge/arXiv-2510.18081-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/javyduck/any-depth-alignment-probes"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Probes-public-yellow.svg" alt="Probes"></a>
  <a href="https://huggingface.co/datasets/javyduck/any-depth-alignment"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-gated-orange.svg" alt="Datasets"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"></a>
</p>

<p align="center">
  <strong>ICLR 2026</strong> · 📄 <a href="https://arxiv.org/abs/2510.18081">Paper</a>
</p>

---

Large language models exhibit **strong but shallow** alignment: they refuse a
harmful request at the very start of a turn, yet that protection collapses once a
harmful continuation is underway — via prefill attacks, adversarial prompts, or
fine-tuning. **Any-Depth Alignment (ADA)** is an inference-time defense that
re-injects the model's **assistant header** — its *Safety Tokens* — mid-stream to
re-trigger the model's own alignment prior at *any* depth, with negligible
overhead and **no change to the base model's weights**.

ADA comes in two variants:

- **ADA-RK (Rethinking)** — inject the Safety Tokens, let the model generate a
  short lookahead, and halt if it refuses. Training-free.
- **ADA-LP (Linear Probe)** — inject the Safety Tokens, read a *single* hidden
  state, and apply a lightweight linear probe to decide whether to halt. The base
  model becomes its own guardrail.

Across Llama, Gemma, Mistral, Qwen, DeepSeek, gpt-oss (and Claude Sonnet 4 for
ADA-RK), ADA secures **near-100% refusal** under deep prefill attacks (dozens to
thousands of tokens), cuts adversarial-prompt attack success (GCG/AutoDAN/PAIR/TAP)
to **below 3%**, keeps benign over-refusal near zero, and stays robust after
subsequent fine-tuning.

## Why it works — one idea, one config file

Alignment is concentrated in the assistant-header tokens through repeated use in
shallow-refusal training. Re-inserting them resets the model's "distance to the
header" to zero, exposing a clean, linearly-separable harmfulness signal. Every
per-model detail this requires — the header string, the probe token, the layer,
the hook — lives in **one registry**, [`configs/models.yaml`](configs/models.yaml),
resolved through [`ada.registry`](ada/registry.py). No module ever branches on a
model name.

```python
from ada.registry import get_model
spec = get_model("google/gemma-2-9b-it")
spec.assistant_header   # '<end_of_turn>\n<start_of_turn>model\n'   (ADA-RK injection)
spec.probe_layer        # 23                                        (ADA-LP read layer)
```

## Repository layout

```
AnyDepthAlignment/
├── ada/                       # the ADA package
│   ├── registry.py            #   single source of truth for per-model config
│   ├── data/                  #   corpus loading, Safety-Token injection, benchmark prompts
│   ├── models/                #   model loading + hook-based hidden-state extraction
│   ├── probe/                 #   ADA-LP: collect → train → evaluate  (E1)
│   ├── rethink/               #   ADA-RK generation + Self-Defense baseline + Claude  (E2/E3)
│   ├── guardrails/            #   classifier-guardrail baselines  (E2/E3/E5)
│   ├── attacks/               #   SFT-attack fine-tuning + adversarial-attack extraction  (E3/E4)
│   ├── datagen/               #   jailbroken-GPT corpus + response generation
│   ├── timing/                #   inference-cost measurement  (E6)
│   ├── plotting/              #   figure/table generation for every experiment
│   ├── serving/               #   optional live streaming-defense demo
│   └── utils/                 #   naming conventions + JSON I/O
├── configs/                   # models.yaml, refusal_keywords.yaml, guardrails.yaml, deepspeed_zero3.json
├── scripts/                   # runnable pipelines to reproduce E1–E6
├── data/                      # train / eval / generated  (see data/README.md)
├── third_party/llm_attacks/   # vendored GCG / AutoDAN / PAIR / TAP engines (MIT)
├── interpretability/          # Appendix C: circuit-tracer transcoder analysis
└── figures/                   # regenerated paper figures
```

## Resources

| Artifact | Link |
|----------|-------------|
| Code | [`javyduck/any-depth-alignment`](https://github.com/javyduck/any-depth-alignment) |
| ADA-LP probes (public) | [`javyduck/any-depth-alignment-probes`](https://huggingface.co/javyduck/any-depth-alignment-probes) |
| Datasets (gated — request access) | [`javyduck/any-depth-alignment`](https://huggingface.co/datasets/javyduck/any-depth-alignment) |

`scripts/prepare_datasets.sh` copies data from a local research repo; alternatively
pull the published artifacts directly, e.g.:

```python
from huggingface_hub import snapshot_download
# probes land at ./ckpts/ (the repo stores them under a ckpts/ prefix)
snapshot_download("javyduck/any-depth-alignment-probes", local_dir=".",
                  allow_patterns="ckpts/**")
# datasets land at ./data/ (after your access request is approved)
snapshot_download("javyduck/any-depth-alignment", repo_type="dataset", local_dir="data")
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

Requires Python ≥ 3.10 and (for most experiments) CUDA GPUs. Every experiment
below assumes the environment is active (`conda activate ada`). Gated
datasets/models (Gemma, Llama, HEx-PHI) need an accepted license and `HF_TOKEN`.

## Quickstart

```bash
# Train an ADA-LP probe for a model (collect hidden states → fit per-layer probe).
# Skip this if you populated the pre-trained probes with INCLUDE_PROBES=1 above.
bash scripts/10_e1_collect.sh  google/gemma-2-9b-it
bash scripts/11_e1_train.sh    google/gemma-2-9b-it

# Evaluate ADA-LP under a deep-prefill attack (uses the probes in ckpts/)
python -m ada.probe.evaluate --model google/gemma-2-9b-it --dataset advbench

# Run ADA-RK (training-free) instead
python -m ada.rethink.generate --model google/gemma-2-9b-it --mode ada_rk --dataset advbench

# Live streaming-defense demo (ADA-LP as the model's own guardrail)
python -m ada.serving.server --model google/gemma-2-9b-it
```

## Reproducing the paper

Each experiment has an ordered set of scripts. All share a single job-queue
helper (`scripts/lib/queue.sh`) and read per-model config from the registry.

| Paper section | What | Scripts | Figures / tables |
|---|---|---|---|
| **§2 / E1** Innate safety & linear separability | collect hidden states → train logistic probes → plot accuracy + t-SNE | `10_e1_collect.sh` · `11_e1_train.sh` · `12_e1_figures.sh` | `val_all_model`, `val_choice_of_safety_token`, `val_hook_position`, `tsne_distribution` |
| **§3 / E2** Deep prefill attacks | ADA-RK / Base / Self-Defense + guardrails over prefill depth | `20_e2_prefill.sh` · `21_e2_baselines.sh` · `22_e2_figures.sh` | `all_models_refusal_rates`, Table 1 |
| **§4 / E3** Adversarial prompt attacks | GCG/AutoDAN/PAIR/TAP → extract → evaluate ADA | `30_e3_run_attacks.sh` · `31_e3_eval.sh` · `32_e3_figures.sh` | `attack_main`, ASR tables |
| **§5 / E4** SFT attacks | benign/adversarial LoRA sweep → re-evaluate ADA | `40_e4_sft_train.sh` · `41_e4_sft_eval.sh` · `42_e4_figures.sh` | `sft_all_harmful_datasets_*{,_full}`, ASR Enable/Disable table |
| **§6 / E5** Over-refusal | benign-benchmark refusal rates | `50_e5_benign.sh` · `51_e5_figures.sh` | `benign_avg_refusal_rates`, `xstest_refusal_rates` |
| **§7 / E6** Inference cost | latency/memory vs guardrails | `60_e6_timing.sh` | `time` |
| **App.** Ablations | checkpoint-frequency (25/50/75/100 + adaptive) & sampling-temperature robustness | `ada.plotting.tables_ablation {frequency,temperature}` | ASR / over-refusal tables |
| **App. C** Interpretability | circuit-tracer transcoder analysis | [`interpretability/`](interpretability/) | `transcorder`, interventions |

The ADA-LP branch of E2/E3/E4/E5 is produced by `ada.probe.evaluate`; the ADA-RK /
Base / Self-Defense branch by `ada.rethink.generate`; the guardrail baselines by
`ada.guardrails.evaluate`.

**Regenerate figures without re-running inference.** A slim, text-stripped subset
of the evaluation logs is published under `example_results/` in the gated dataset
repo. Fetch it and re-plot the main figures directly:

```bash
bash scripts/fetch_example_results.sh   # ~1 GB; needs gated-dataset access
bash scripts/make_all_figures.sh        # -> figures/*.pdf
```

## Tests

A `pytest` suite guards the registry, config integrity, per-model probe-token
tokenization, refusal-scoring, curve/ASR accounting, probe training, the HEx-PHI
round-trip, and every CLI entrypoint:

```bash
pip install -e ".[dev]"
bash scripts/run_tests.sh          # fast unit tests (no GPU / no model download)
bash scripts/run_tests.sh smoke    # + real-tokenizer + CLI-help smoke tests
bash scripts/run_tests.sh all
```

## Models

`Llama-2-7b-chat`, `Llama-3.1-8B-Instruct`, `Ministral-8B-Instruct-2410`,
`gemma-2-{2b,9b,27b}-it`, `Qwen2.5-7B-Instruct`, `DeepSeek-R1-Distill-Qwen-7B`,
`gpt-oss-120b`, and `Claude Sonnet 4` (ADA-RK only). Adding a model = one entry in
`configs/models.yaml`.

## Responsible use

This repository contains, by necessity, harmful prompts and completions used to
*evaluate a defense*. It is released for safety research. Please follow the
licenses of all underlying datasets and use this code to make models safer.

**HEx-PHI** is gated under the LLM-Tuning-Safety license, so its prompts are
**never redistributed** — not in this repo and not in the published dataset. You
can still recover *our exact HEx-PHI continuations*: the gated dataset ships a
prompt-free reference file (SHA-256 of each prompt + our continuation), which you
re-join against your own licensed HEx-PHI copy with
`python -m ada.datagen.hexphi_reference reconstruct`. Full steps (and a
from-scratch alternative) in [`docs/HEXPHI.md`](docs/HEXPHI.md); running `hexphi`
without the file prints a pointer to them.

## Citation

If you use ADA, please cite our paper
([arXiv:2510.18081](https://arxiv.org/abs/2510.18081)):

```bibtex
@inproceedings{zhang2026anydepth,
  title     = {Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth},
  author    = {Zhang, Jiawei and Estornell, Andrew and Baek, David D. and Li, Bo and Xu, Xiaojun},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2510.18081}
}
```

Licensed under MIT (see [LICENSE](LICENSE)). Vendored attack engines and the
circuit-tracer dependency retain their own licenses.
