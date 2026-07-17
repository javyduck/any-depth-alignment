"""Loaders and format normalizers for response corpora.

The pipeline works with two on-disk conventions for conversation records, both
handled transparently here:

* ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
* ``{"prompt": ..., "response": ...}``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from ..utils.io import read_jsonl
from ..utils.naming import slugify_model

logger = logging.getLogger(__name__)


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
