---
license: mit
pretty_name: Any-Depth Alignment (ADA) Linear Probes
tags:
  - safety
  - alignment
  - guardrail
  - linear-probe
  - jailbreak
library_name: scikit-learn
---

# Any-Depth Alignment (ADA) — Linear Probes (ADA-LP)

Pre-trained **ADA-LP** probes for **Any-Depth Alignment: Unlocking the Innate
Safety Alignment of LLMs to Any Depth** (ICLR 2026). Code:
[Any-Depth-Alignment](https://github.com/) · Data:
[`javyduck/any-depth-alignment`](https://huggingface.co/javyduck/any-depth-alignment).

Each probe is a scikit-learn `LogisticRegression` trained on the hidden state of
an injected **Safety Token** (the assistant header) at one layer. At inference,
ADA-LP re-injects the Safety Tokens mid-generation, reads that single hidden
state, and applies the probe to decide whether to halt — turning the base model
into its own guardrail with constant overhead and no weight updates.

## What's inside

Probes for 12 models, covering every layer × Safety-Token × hook-position in the
paper's ablations (~3.6k `.joblib` files). Path layout (mirrors the training code):

```
ckpts/{model_slug}/{safety_slug}/mask_token_none/{hook_slug}/gradual_cache/seed_42/logistic/layer_{L}.joblib
```

The canonical per-model probe (Safety Token, layer) matches the model registry in
the code repo, e.g. `google/gemma-2-9b-it` → layer 23, `meta-llama/Llama-3.1-8B-Instruct`
→ layer 15, `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` → layer 13, `openai/gpt-oss-120b` → layer 33.

## Usage

```python
import joblib
probe = joblib.load("ckpts/google_gemma-2-9b-it/.../logistic/layer_23.joblib")
prob_harmful = probe.predict_proba(safety_token_hidden_state)[:, 1]
```

In the code repo this is fully automated — `python -m ada.probe.evaluate --model
google/gemma-2-9b-it --dataset advbench` loads the right probe from the registry.

## Citation

```bibtex
@inproceedings{zhang2026anydepth,
  title     = {Any-Depth Alignment: Unlocking the Innate Safety Alignment of LLMs to Any Depth},
  author    = {Zhang, Jiawei and Estornell, Andrew and Li, Bo and Baek, David D. and Xu, Xiaojun},
  booktitle = {ICLR},
  year      = {2026}
}
```
