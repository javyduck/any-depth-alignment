"""Filesystem naming conventions shared across the whole pipeline.

These helpers define the directory layout for hidden states, probe checkpoints,
and evaluation logs. They are the single source of truth for slugs — every stage
(collect → train → evaluate → plot) resolves paths through this module so the
artifacts written by one stage are found unchanged by the next.

Layout produced by the ADA-LP pipeline::

    hidden_states/{split}/{model}/{data}/{safety}/{mask}/{hook}/{cache}/index_{i}/{layer}.pt
    ckpts/{model}/{safety}/{mask}/{hook}/{cache}/seed_{seed}/logistic/layer_{L}.joblib
    logs/{benign|harmful}/{dataset}/{model}/{safety}/{mask}/{hook}/seed_{seed}/logistic/probe-layers{L}/depth_{d}_maxdepth_{md}.json
"""

from __future__ import annotations

import re


def sanitize_filename(s: str) -> str:
    """Make an arbitrary token string safe to embed in a path component."""
    s = s.replace("<", "_").replace(">", "_").replace(" ", "_")
    s = s.replace("\\", "_").replace("\n", "_n")
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def slugify_model(model_name: str) -> str:
    """``org/Model-Name`` → ``org_Model-Name`` (also flattens dots)."""
    return model_name.replace("/", "_").replace(".", "_")


def slugify_safety_tokens(safety_tokens: str) -> str:
    if safety_tokens == "" or safety_tokens == "empty":
        return "safety_token_empty"
    return "safety_token_" + sanitize_filename(safety_tokens)


def slugify_mask_tokens(mask_tokens: "str | None") -> str:
    if mask_tokens is None or not str(mask_tokens).strip() or str(mask_tokens).lower() == "none":
        return "mask_token_none"
    return "mask_token_" + sanitize_filename(mask_tokens)


def slugify_hook_position(hook_position: str) -> str:
    return f"hook_{hook_position}"


def slugify_cache(gradual_cache: bool) -> str:
    return "gradual_cache" if gradual_cache else "no_cache"
