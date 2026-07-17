# Interpretability: why Safety Tokens re-trigger refusal (Appendix C)

This module reproduces the circuit-level analysis in Appendix C ("Refusal
Activation via Transcorder"). Using [circuit-tracer](https://github.com/safety-research/circuit-tracer)
with GemmaScope cross-layer transcoders (CLTs), it shows that a small set of
refusal-linked transcoder features **spike precisely on the injected assistant
header** (the Safety Tokens) and are near-zero across a harmful continuation —
the mechanistic signature of ADA re-activating the model's latent safety circuit.

All analysis runs on `google/gemma-2-2b-it`; the GemmaScope transcoders are
trained on the *base* `google/gemma-2-2b` (as noted in the paper).

## Contents

```
interpretability/
├── refusal_features.py      # the 10 refusal-linked CLT features (layer, index)
├── utils.py                 # circuit-tracer viz helpers (display_generations_comparison, ...)
├── graph_visualization.py   # attribution-graph rendering
├── notebooks/
│   ├── clt_refusal_features.ipynb   # → Fig: CLT activations spike on the header (transcorder.pdf)
│   ├── intervention_zero_out.ipynb  # → Fig xx(a): zeroing the features turns refusal → harmful
│   └── intervention_rescale.ipynb   # → Fig xx(b): forcing them high → spurious refusal on a benign query
├── graphs/                  # precomputed raw attribution graphs (.pt) for it / sft / base
├── graph_files/             # frontend-ready attribution graphs (.json)
└── figures/                 # executed PDF outputs
```

## Setup

circuit-tracer is an external dependency (not vendored — it carries its own
license, tests, and large frontend assets):

```bash
pip install -r interpretability/requirements.txt   # installs circuit-tracer + deps
huggingface-cli login                              # accept the gated Gemma-2 license
```

The notebooks currently add circuit-tracer to `sys.path` (they were developed
against a local checkout two levels up). After `pip install circuit-tracer` you
can replace those `sys.path.append(...)` lines with a plain `import circuit_tracer`.
They also `import setGPU` for automatic GPU selection; remove that line or
`pip install setGPU` if you don't use it.

## Reproduce

1. **`clt_refusal_features.ipynb`** — loads `gemma-2-2b-it` + the `gemma`
   transcoder preset (auto-downloads `google/gemma-scope-2b-pt-transcoders`),
   computes activations of the features in `refusal_features.py` on the prompt
   *"How to commit suicide."* with the header re-injected mid-stream, and renders
   the CLT-activation figure. Optionally runs `attribute(...)` + `serve(...)` to
   view the refusal circuit interactively.
2. **`intervention_zero_out.ipynb`** — zeros the ten features on the `model`+`\n`
   header positions; a safe refusal collapses into harmful output.
3. **`intervention_rescale.ipynb`** — forces the ten features to a large positive
   value on a benign prompt; the model emits an unwarranted refusal.

The per-feature semantic panels (apology vs. first-person "I") are screenshots of
the circuit-tracer frontend feature dashboards for `(18, 12640)` and `(22, 5394)`.

Precomputed attribution graphs are provided under `graphs/` and `graph_files/` so
the circuits can be inspected offline without rerunning attribution.

### Paper Appendix C figure mapping

| Paper figure | Produced by |
|---|---|
| `transcorder_new.pdf` (Fig: Safety-Token reactivation) | `clt_refusal_features.ipynb` (CLT-activation plot) |
| `activation1.png`, `activation2.png` (Fig: CLT features) | manual screenshots of the circuit-tracer feature dashboards for `(18,12640)` / `(22,5394)` — no script regenerates these |
| `intervention1.png` (Fig xx a) | `intervention_zero_out.ipynb` |
| `intervention2.png` (Fig xx b) | `intervention_rescale.ipynb` |
