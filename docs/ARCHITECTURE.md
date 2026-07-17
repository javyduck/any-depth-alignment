# Architecture

The codebase is organized around one principle: **every per-model detail lives in
a single registry, and every stage reads from it.** This keeps the pipeline
model-agnostic — adding a model is one YAML entry, never a code change.

## The registry

`configs/models.yaml` → `ada.registry.get_model(hf_id)` → a frozen `ModelSpec`:

| Field | Used by | Meaning |
|---|---|---|
| `assistant_header` | ADA-RK | full header re-injected mid-stream (the Safety Tokens) |
| `probe_safety_tokens` | ADA-LP | header variant whose last token is the probe token |
| `probe_layer` | ADA-LP | transformer block whose hidden state the probe reads |
| `hook_position` | ADA-LP | where in the block to read (default `input_layernorm`) |
| `reasoning` | ADA-RK | whether the model emits a `<think>` block |
| `generation_prompt_suffix` | collect/eval/gen | tokens appended after the generation prompt to reach the answer (reasoning/channel close) |
| `generation_prompt_space` | collect/eval/gen | append a single space after the template (e.g. Llama-2's `[/INST]`) |
| `chat_template_from` | loading | borrow another model's chat template if missing |
| `short_name` | plotting | optional compact legend label (falls back to the HF basename) |

Guardrail baselines and refusal keywords have their own small configs
(`configs/guardrails.yaml`, `configs/refusal_keywords.yaml`).

## Data flow

```
                          ┌─────────────────────── configs/models.yaml (registry) ──────────────────────┐
                          │                                                                              │
  data/train/probe  ──►  ada.probe.collect  ──►  hidden_states/  ──►  ada.probe.train  ──►  ckpts/*.joblib
  (continuations)         (inject Safety Tokens,                       (per-layer sklearn        (ADA-LP probes)
                           hook hidden states)                          LogisticRegression)
                                                                                                  │
  data/eval/*  ──►  ada.probe.evaluate ──────────────────────────────────────────────────────────┘  ──►  logs/
                    ada.rethink.generate  (Base / ADA-RK / Self-Defense)                              ──►  vllm_generation_logs/
                    ada.guardrails.evaluate  (Llama-Guard, Granite-Guardian, …)                       ──►  vllm_defense_logs/
                                                             │
                                                             ▼
                                                     ada.plotting.*  ──►  figures/
```

The **injection mechanism** (`ada.data.injection`) is shared by all three
evaluators: build `user_prefix + assistant[:depth] + safety_tokens`, then either
read the hidden state at the last token (ADA-LP) or generate a short lookahead
(ADA-RK). The **hook-based extractor** (`ada.models.extraction`) reads any
sub-module of any block at chosen token positions in a single forward pass.

## Naming conventions

`ada.utils.naming` defines the on-disk layout so each stage finds the previous
stage's artifacts unchanged:

```
hidden_states/{split}/{model}/{data}/{safety}/{mask}/{hook}/{cache}/index_{i}/{layer}.pt
ckpts/{model}/{safety}/{mask}/{hook}/{cache}/seed_{seed}/logistic/layer_{L}.joblib
logs/{benign|harmful}/{dataset}/{model}/{safety}/{mask}/{hook}/seed_{seed}/logistic/probe-layers{L}/depth_{d}_maxdepth_{md}.json
vllm_generation_logs/{benign|harmful}/{dataset}/{model}/mode_{base|ada_rk|self_defense}/depth_{d}_maxdepth_{md}.json
vllm_defense_logs/{benign|harmful}/{dataset}/{guardrail}/{model}/depth_{d}_maxdepth_{md}.json
```

## Design choices worth knowing

- **The probe is sklearn `LogisticRegression`** (saved as `.joblib`), fit per
  layer on Safety-Token hidden states — matching the paper, not a torch MLP.
- **Refusal detection** is case-insensitive substring matching against
  `configs/refusal_keywords.yaml`; an instance counts as refused if *any*
  checkpoint fires. The exact keyword sets are preserved for reproducibility.
- **Secrets** are always read from the environment (`OPENAI_API_KEY`,
  `ANTHROPIC_API_KEY`, `HF_TOKEN`); nothing is hardcoded.
