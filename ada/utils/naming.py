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
from pathlib import Path
from typing import Optional, Union

# Pipeline-wide defaults for the on-disk layout (shared by writers + plotting).
DEFAULT_SEED = 42
DEFAULT_HOOK_POSITION = "input_layernorm"
DEFAULT_DEPTH_STEP = 25
DEFAULT_MAX_DEPTH = 3000


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


# --------------------------------------------------------------------------- #
# Full on-disk path builders (shared source of truth for the pipeline layout).
# --------------------------------------------------------------------------- #
def model_dir_slug(
    model: str,
    adapter_type: Optional[str] = None,
    step: Optional[Union[int, str]] = None,
    disable_safetytoken: bool = False,
) -> str:
    """Directory component for a (possibly adapter-tuned) model.

    ``{model_slug}[-{adapter_type}-adapter-{step}[-disable_safetytoken]]``.
    """
    slug = slugify_model(model)
    if adapter_type is not None and step is not None:
        slug = f"{slug}-{adapter_type}-adapter-{step}"
        if disable_safetytoken:
            slug = f"{slug}-disable_safetytoken"
    return slug


def find_generation_log(
    split: str,
    dataset: str,
    model: str,
    mode: str,
    depth: Union[int, str] = DEFAULT_DEPTH_STEP,
    maxdepth: Union[int, str] = DEFAULT_MAX_DEPTH,
    *,
    reasoning: bool = False,
    adapter_type: Optional[str] = None,
    step: Optional[Union[int, str]] = None,
    disable_safetytoken: bool = False,
    base_dir: Union[str, Path] = "vllm_generation_logs",
) -> Path:
    """Path to an ADA-RK / Base / Self-Defense generation log."""
    model_dir = model_dir_slug(model, adapter_type, step, disable_safetytoken)
    mode_dir = f"mode_{mode}" + ("_reasoning" if reasoning else "")
    return (
        Path(base_dir) / split / dataset / model_dir / mode_dir
        / f"depth_{depth}_maxdepth_{maxdepth}.json"
    )


def find_probe_log(
    split: str,
    dataset: str,
    model: str,
    safety_tokens: str,
    layer: int,
    depth: Union[int, str] = DEFAULT_DEPTH_STEP,
    maxdepth: Union[int, str] = DEFAULT_MAX_DEPTH,
    *,
    mask_tokens: str = "none",
    hook_position: str = DEFAULT_HOOK_POSITION,
    seed: int = DEFAULT_SEED,
    probe_type_dir: str = "logistic",
    adapter_type: Optional[str] = None,
    step: Optional[Union[int, str]] = None,
    disable_safetytoken: bool = False,
    base_dir: Union[str, Path] = "logs",
) -> Path:
    """Path to an ADA-LP probe evaluation log for one (dataset, model, layer)."""
    model_dir = model_dir_slug(model, adapter_type, step, disable_safetytoken)
    return (
        Path(base_dir) / split / dataset / model_dir
        / slugify_safety_tokens(safety_tokens)
        / slugify_mask_tokens(mask_tokens)
        / slugify_hook_position(hook_position)
        / f"seed_{seed}" / probe_type_dir / f"probe-layers{layer}"
        / f"depth_{depth}_maxdepth_{maxdepth}.json"
    )


def find_defense_log(
    split: str,
    dataset: str,
    guardrail: str,
    model: str,
    depth: Union[int, str] = DEFAULT_DEPTH_STEP,
    maxdepth: Union[int, str] = DEFAULT_MAX_DEPTH,
    *,
    base_dir: Union[str, Path] = "vllm_defense_logs",
) -> Path:
    """Path to an external-guardrail defense log."""
    return (
        Path(base_dir) / split / dataset / slugify_model(guardrail) / slugify_model(model)
        / f"depth_{depth}_maxdepth_{maxdepth}.json"
    )


def probe_ckpt_dir(
    model: str,
    safety_tokens: str,
    *,
    mask_tokens: str = "none",
    hook_position: str = DEFAULT_HOOK_POSITION,
    gradual_cache: bool = True,
    seed: int = DEFAULT_SEED,
    ckpt_dir: Union[str, Path] = "ckpts",
) -> Path:
    """Directory holding per-layer probe checkpoints/metrics for a model."""
    return (
        Path(ckpt_dir) / slugify_model(model)
        / slugify_safety_tokens(safety_tokens)
        / slugify_mask_tokens(mask_tokens)
        / slugify_hook_position(hook_position)
        / slugify_cache(gradual_cache)
        / f"seed_{seed}" / "logistic"
    )


def find_probe_ckpt_json(model: str, safety_tokens: str, layer: int, **kwargs) -> Path:
    """Path to the ``layer_{L}.json`` train/val-metrics record for one layer."""
    return probe_ckpt_dir(model, safety_tokens, **kwargs) / f"layer_{layer}.json"


def hidden_states_index_dir(
    split: str,
    model: str,
    data_type: str,
    safety_tokens: str,
    index: int,
    *,
    mask_tokens: str = "none",
    hook_position: str = DEFAULT_HOOK_POSITION,
    gradual_cache: bool = True,
    hidden_states_dir: Union[str, Path] = "hidden_states",
) -> Path:
    """Directory of one hidden-state shard (contains ``{layer}.pt`` files)."""
    return (
        Path(hidden_states_dir) / split / slugify_model(model) / data_type
        / slugify_safety_tokens(safety_tokens)
        / slugify_mask_tokens(mask_tokens)
        / slugify_hook_position(hook_position)
        / slugify_cache(gradual_cache)
        / f"index_{index}"
    )
