"""Any-Depth Alignment (ADA).

An inference-time safety defense that re-injects a model's assistant header — the
*Safety Tokens* — mid-generation to re-trigger its innate shallow-refusal
alignment at any generation depth. Two variants:

* **ADA-RK** (Rethinking): inject the header, generate a short lookahead, and halt
  if a refusal appears.
* **ADA-LP** (Linear Probe): inject the header and read a single hidden state,
  then apply a lightweight linear probe to decide whether to halt.

The model registry (:mod:`ada.registry`) is the single source of truth for every
per-model detail; no other module hard-codes a model name.
"""

from .registry import ModelSpec, get_model, list_models

__version__ = "1.0.0"

__all__ = ["ModelSpec", "get_model", "list_models", "__version__"]
