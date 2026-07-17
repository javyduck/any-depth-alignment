"""Loaders and format normalizers for response corpora.

The pipeline works with two on-disk conventions for conversation records, both
handled transparently here:

* ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
* ``{"prompt": ..., "response": ...}``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..registry import deep_alignment_base
from ..utils.io import read_jsonl
from ..utils.naming import slugify_model

logger = logging.getLogger(__name__)

# Neutral directory name for the deep-prefill harmful corpus (source-layout
# fallback subdir under ``harmful_responses/{dataset}/``). The paper deliberately
# withholds the jailbroken generator and its SFT recipe, so no model handle or
# training composition is encoded here. Shipped release data uses the
# ``deep_prefill/`` layout (the primary path in resolve_response_file); this slug
# is only the source-tree fallback. Single source of truth shared by
# probe/rethink/guardrails response resolution.
DEFAULT_HARMFUL_SOURCE = "jailbroken_gpt"


def resolve_response_file(
    dataset: str,
    model: Optional[str] = None,
    *,
    benign: bool = False,
    attack: Optional[str] = None,
    response_file: "str | Path | None" = None,
    data_root: "str | Path" = "data/eval",
    benign_dir: str = "benign_responses",
    harmful_dir: str = "harmful_responses",
    harmful_source: Optional[str] = DEFAULT_HARMFUL_SOURCE,
) -> Path:
    """Locate the stored response corpus to evaluate/sweep — one shared resolver.

    Tries the release layout under ``data_root`` first, then the original source
    layout as a fallback. Used by ADA-LP (``probe.evaluate``), ADA-RK
    (``rethink.generate``) and the guardrail baselines (``guardrails.evaluate``):

    * benign  → ``over_refusal/{dataset}/{slug}/responses.jsonl`` (deep-alignment
      checkpoints reuse their base model's benign responses via ``deep_alignment_base``)
    * attack  → ``attacks/{dataset}_{attack}/{slug}/responses.jsonl``
    * harmful → ``deep_prefill/{dataset}_responses.jsonl``

    ``harmful_source`` (when set) selects the source-layout deep-prefill subdir
    ``{harmful_dir}/{dataset}/{harmful_source}/responses.jsonl`` for the fallback.
    """
    if response_file:
        path = Path(response_file)
        if not path.exists():
            raise FileNotFoundError(f"Response file not found: {path}")
        return path

    ds = dataset.lower()
    root = Path(data_root)
    if benign:
        base = deep_alignment_base(model)  # None for non-deep-alignment models
        slug = slugify_model(base or model) if model else "unknown_model"
        candidates = [
            root / "over_refusal" / ds / slug / "responses.jsonl",
            Path(benign_dir) / ds / slug / "responses.jsonl",
        ]
    elif attack:
        slug = slugify_model(model) if model else "unknown_model"
        name = f"{ds}_{attack.lower()}"
        candidates = [
            root / "attacks" / name / slug / "responses.jsonl",
            Path(harmful_dir) / name / slug / "responses.jsonl",
        ]
    else:
        src_fallback = (
            Path(harmful_dir) / ds / harmful_source / "responses.jsonl"
            if harmful_source else Path(harmful_dir) / ds / "responses.jsonl"
        )
        candidates = [root / "deep_prefill" / f"{ds}_responses.jsonl", src_fallback]

    for candidate in candidates:
        if candidate.exists():
            logger.info("Using response file: %s", candidate)
            return candidate
    msg = "No response file found. Tried: " + ", ".join(str(c) for c in candidates)
    if ds == "hexphi":
        msg += (
            "\nHEx-PHI is not distributed (gated LLM-Tuning-Safety license); "
            "reproduce it locally — see docs/HEXPHI.md."
        )
    raise FileNotFoundError(msg)


def extract_messages(response: Dict) -> List[Dict]:
    """Normalize a record into a ``[user, assistant]`` message list."""
    if "messages" in response:
        return response["messages"]
    if "prompt" in response and "response" in response:
        return [
            {"role": "user", "content": response["prompt"]},
            {"role": "assistant", "content": response["response"]},
        ]
    raise ValueError(f"Unrecognized response format: {response!r}")


def extract_response_text(response: Dict) -> str:
    """Return the assistant text from a record in any supported format."""
    if "messages" in response:
        for msg in response["messages"]:
            if msg["role"] == "assistant":
                return msg["content"]
        raise ValueError("No assistant message found in record")
    if "response" in response:
        return response["response"]
    if "text" in response:
        return response["text"]
    raise ValueError(f"Unrecognized response format: {response!r}")


def load_harmful_responses(file_path: "str | Path") -> List[Dict]:
    """Load a JSONL corpus of harmful continuations (e.g. deep-prefill sources)."""
    return read_jsonl(file_path)


def load_benign_responses(benign_dir: "str | Path", model_name: str) -> Dict[str, List[Dict]]:
    """Load per-dataset benign continuations for ``model_name``.

    Expects ``{benign_dir}/{dataset}/{model_slug}/train_responses.jsonl``. Falls
    back to any available model directory (with a warning) if the requested model
    is missing, which is convenient when assembling the probe corpus.
    """
    benign_dir = Path(benign_dir)
    model_slug = slugify_model(model_name)
    corpora: Dict[str, List[Dict]] = {}

    for dataset_dir in sorted(p for p in benign_dir.iterdir() if p.is_dir()):
        model_dir = dataset_dir / model_slug
        if not model_dir.exists():
            available = [d for d in dataset_dir.iterdir() if d.is_dir()]
            if not available:
                logger.warning("No model responses under %s", dataset_dir)
                continue
            model_dir = available[0]
            logger.warning(
                "Model %s missing for %s; using %s", model_slug, dataset_dir.name, model_dir.name
            )
        response_file = model_dir / "train_responses.jsonl"
        if response_file.exists():
            corpora[dataset_dir.name] = read_jsonl(response_file)
            logger.info("Loaded %d benign responses from %s", len(corpora[dataset_dir.name]), dataset_dir.name)
        else:
            logger.warning("Missing %s", response_file)
    return corpora
