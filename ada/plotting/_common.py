"""Shared utilities for the Any-Depth Alignment (ADA) plotting scripts.

Every ``ada.plotting.*`` script imports from this module so that log parsing,
on-disk path construction, the model list, and matplotlib styling stay
consistent across figures. Nothing here hard-codes a per-model token or probe
layer: those come from :mod:`ada.registry`, and all path slugs come from
:mod:`ada.utils.naming`.

The three experiment families read from these fixed layouts (all CWD-relative by
default; override the ``*_dir`` arguments to point elsewhere)::

    # ADA-RK / Base / Self-Defense generations (mode plots, E2/E3/E4)
    vllm_generation_logs/{split}/{dataset}/{model_dir}/mode_{mode}[_reasoning]/depth_{d}_maxdepth_{md}.json
    # ADA-LP probe evaluations (probe refusal-rate tables, E1/E2)
    logs/{split}/{dataset}/{model_dir}/{safety}/{mask}/{hook}/seed_{seed}/logistic/probe-layers{L}/depth_{d}_maxdepth_{md}.json
    # External guardrails
    vllm_defense_logs/{split}/{dataset}/{guardrail}/{model_slug}/depth_{d}_maxdepth_{md}.json
    # E1 probe accuracy checkpoints
    ckpts/{model_slug}/{safety}/{mask}/{hook}/{cache}/seed_{seed}/logistic/layer_{L}.json
    # E1 t-SNE hidden states
    hidden_states/{split}/{model_slug}/{data}/{safety}/{mask}/{hook}/{cache}/index_{i}/{layer}.pt

where ``{split}`` is ``harmful`` / ``benign`` and ``{model_dir}`` is the model
slug optionally suffixed with ``-{adapter_type}-adapter-{step}[-disable_safetytoken]``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

from ada.registry import get_model, list_models
from ada.utils.io import read_json
from ada.utils.naming import (
    sanitize_filename,
    slugify_cache,
    slugify_hook_position,
    slugify_mask_tokens,
    slugify_model,
    slugify_safety_tokens,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Shared constants
# --------------------------------------------------------------------------- #

# All registered models, in registry (paper) order.
MODELS: List[str] = list_models()

DEFAULT_SEED = 42
DEFAULT_HOOK_POSITION = "input_layernorm"
DEFAULT_DEPTH_STEP = 25
DEFAULT_MAX_DEPTH = 3000

# Attack-set sizes (fixed ASR denominators): AdvBench 50 prompts, JailbreakBench 100.
DATASET_TOTALS = {"advbench": 50, "jailbreakbench": 100}

# Cosmetic short names for legends (matching the paper figures). Purely display.
# Resolution order: the registry's optional ``short_name`` field (so a new model
# sets its legend label in configs/models.yaml), then this built-in table, then
# the last path component of the HF id. The table keeps the paper's exact labels.
_SHORT_MODEL_NAMES = {
    "meta-llama/Llama-2-7b-chat-hf": "Llama-2-7b-it",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B-it",
    "mistralai/Ministral-8B-Instruct-2410": "Ministral-8B-it",
    "google/gemma-2-2b-it": "gemma-2-2b-it",
    "google/gemma-2-9b-it": "gemma-2-9b-it",
    "google/gemma-2-27b-it": "gemma-2-27b-it",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": "DeepSeekR1-7B",
    "openai/gpt-oss-120b": "gpt-oss-120b",
}


def short_model_name(model: str) -> str:
    """Return a compact legend label for ``model`` (an HF id)."""
    try:
        registry_label = get_model(model).short_name
        if registry_label:
            return registry_label
    except KeyError:
        pass
    return _SHORT_MODEL_NAMES.get(model, model.split("/")[-1])


# --------------------------------------------------------------------------- #
# Registry access (with a defensive Unicode repair)
# --------------------------------------------------------------------------- #

def probe_safety_tokens(model: str) -> str:
    """Registry ADA-LP Safety-Token string for ``model`` (HF id).

    This is the per-model Safety Token whose hidden state ADA-LP probes; it is
    the single source of truth for the safety slug in probe/hidden-state paths.
    """
    return get_model(model).probe_safety_tokens


def probe_layer(model: str) -> int:
    """Registry probe layer for ``model`` (HF id)."""
    return get_model(model).probe_layer


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

def _to_bool(value) -> bool:
    """Coerce a JSON ``is_refusal`` field (bool or ``"True"``/``"False"``)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


# Depths with fewer than MIN_COUNT observed instances are dropped (matches the
# notebooks' parse_log, which suppresses sparsely-sampled deep checkpoints).
MIN_COUNT = 10


def parse_refusal_curve(log_path: Union[str, Path]) -> Dict[int, float]:
    """Independent per-depth refusal-rate curve from one generation/probe log.

    Each log JSON stores ``detailed_logs`` as a flat list of per-checkpoint
    records ``{instance, depth, is_refusal, ...}``. The rate at depth ``d`` is the
    fraction of instances whose checkpoint *at that depth* is a refusal:

        rate(d) = #{instances refusing at depth d} / #{instances checked at depth d}

    This is the definition the depth-resolved figures (E2 deep prefill, E4 SFT,
    E5 over-refusal) rely on: a base model's per-depth refusal *drops* as a
    harmful prefill deepens, which a cumulative curve would hide. For the
    "did it ever refuse" question (E3 attack-success), use
    :func:`cumulative_refusal_curve` instead.
    """
    detailed = read_json(log_path).get("detailed_logs", [])
    seen: Dict[tuple, bool] = {}
    for i, rec in enumerate(detailed):
        depth = int(rec.get("depth", 0))
        inst = rec.get("instance", i)
        seen[(inst, depth)] = _to_bool(rec.get("is_refusal", False))

    counts: Dict[int, int] = {}
    refused: Dict[int, int] = {}
    for (_inst, depth), flag in seen.items():
        counts[depth] = counts.get(depth, 0) + 1
        refused[depth] = refused.get(depth, 0) + int(flag)
    return {d: refused[d] / counts[d] for d in sorted(counts) if counts[d] >= MIN_COUNT}


def cumulative_refusal_curve(log_path: Union[str, Path]) -> Dict[int, float]:
    """Cumulative refusal-rate curve: fraction that refused at *any* checkpoint <= d.

    Used for E3 attack-success-rate accounting, where an attack succeeds only if
    the defense never fires during the whole generation. The value at the deepest
    checkpoint is the fraction that *ever* refused; ``1 - that`` is the ASR.
    """
    data = read_json(log_path)
    detailed = data.get("detailed_logs", [])

    depths, instances = set(), set()
    earliest_refusal: Dict[object, int] = {}
    for rec in detailed:
        inst = rec["instance"]
        depth = int(rec["depth"])
        depths.add(depth)
        instances.add(inst)
        if _to_bool(rec.get("is_refusal", False)):
            prev = earliest_refusal.get(inst)
            if prev is None or depth < prev:
                earliest_refusal[inst] = depth

    total = data.get("total_responses") or len(instances)
    if not total:
        return {}
    return {
        d: sum(1 for rd in earliest_refusal.values() if rd <= d) / total
        for d in sorted(depths)
    }


def count_instances(log_path: Union[str, Path]) -> int:
    """Number of distinct instances (depth > 0) in a per-depth log.

    Matches the #instances denominator used by the refusal curves, so the two
    combine consistently for ASR accounting.
    """
    try:
        logs = read_json(log_path).get("detailed_logs", [])
    except (OSError, ValueError):
        return 0
    seen = set()
    for row_idx, entry in enumerate(logs):
        if entry.get("depth", 0) > 0:
            inst = entry.get("instance")
            seen.add(inst if inst is not None else f"_row_{row_idx}")
    return len(seen)


def asr_from_generation_log(path: Union[str, Path], total: int) -> float:
    """ASR from a per-depth refusal log with the fixed-denominator renormalisation.

    An attack succeeds iff the response is never flagged as a refusal at any
    checkpoint. #never-refused = #present * (1 - ever-refused-rate); ASR divides by
    the fixed attack-set ``total`` so missing instances count as refusals (defenses).
    """
    path = Path(path)
    if not path.exists():
        return 0.0
    try:
        curve = cumulative_refusal_curve(path)
    except Exception:  # noqa: BLE001 - a malformed/empty log means no successes
        return 0.0
    n_present = count_instances(path)
    if n_present == 0:
        return 0.0
    ever_refused_rate = curve[max(curve)] if curve else 0.0
    never_refused = round(n_present * (1.0 - ever_refused_rate))
    return never_refused / total


# --------------------------------------------------------------------------- #
# On-disk path builders
# --------------------------------------------------------------------------- #

def model_dir_slug(
    model: str,
    adapter_type: Optional[str] = None,
    step: Optional[Union[int, str]] = None,
    disable_safetytoken: bool = False,
) -> str:
    """Directory component for a (possibly adapter-tuned) model.

    ``{model_slug}[-{adapter_type}-adapter-{step}[-disable_safetytoken]]``.
    If ``model`` already contains an adapter suffix (as some logs were written),
    pass it verbatim with ``adapter_type=None`` and it is slugified as-is.
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
    """Path to an ADA-RK / Base / Self-Defense generation log.

    ``mode`` is one of ``empty`` (Base), ``add_safetytoken`` (ADA-RK single
    injection) or ``reflection`` (ADA-RK / Self-Defense lookahead); the
    ``_reasoning`` suffix is appended when ``reasoning`` is True.
    """
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
    guardrail_slug = slugify_model(guardrail)
    return (
        Path(base_dir) / split / dataset / guardrail_slug / slugify_model(model)
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
    """Directory of one hidden-state shard (contains ``{layer}.pt`` files).

    ``data_type`` is ``harmful`` / ``benign``; ``split`` is ``train`` / ``val``.
    """
    return (
        Path(hidden_states_dir) / split / slugify_model(model) / data_type
        / slugify_safety_tokens(safety_tokens)
        / slugify_mask_tokens(mask_tokens)
        / slugify_hook_position(hook_position)
        / slugify_cache(gradual_cache)
        / f"index_{index}"
    )


# --------------------------------------------------------------------------- #
# Matplotlib styling
# --------------------------------------------------------------------------- #

def set_plot_style(font_scale: float = 1.0) -> None:
    """Apply the shared clean paper style (large fonts, dashed grey grid).

    This is the style used by the per-layer / refusal-rate line plots. Scripts
    with bespoke needs (e.g. the dense t-SNE grid) set their own rcParams.
    """
    import matplotlib as mpl
    import seaborn as sns

    sns.set_style("whitegrid")
    sns.set_palette("husl")
    mpl.rcParams.update({
        "font.size": 25 * font_scale,
        "axes.titlesize": 25 * font_scale,
        "axes.labelsize": 25 * font_scale,
        "xtick.labelsize": 20 * font_scale,
        "ytick.labelsize": 20 * font_scale,
        "legend.fontsize": 25 * font_scale,
        "figure.titlesize": 25 * font_scale,
        "grid.color": "gray",
        "grid.alpha": 0.2,
        "grid.linestyle": "--",
    })


def husl_palette(n: int):
    """Return ``n`` visually distinct colours (seaborn husl)."""
    import seaborn as sns

    return sns.color_palette("husl", n)


# Marker cycle reused across per-model / per-condition line plots.
MARKERS = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h"]


def ensure_output_dir(output_dir: Union[str, Path]) -> Path:
    """Create ``output_dir`` if needed and return it as a ``Path``."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out
