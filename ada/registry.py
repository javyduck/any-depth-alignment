"""Model registry: the single source of truth for per-model configuration.

Every model-specific detail used anywhere in Any-Depth Alignment — the assistant
header re-injected by ADA-RK, the Safety-Token span probed by ADA-LP, the probe
layer, the hook position, and any chat-template quirks — is declared once in
``configs/models.yaml`` and resolved through this module. No other file in the
codebase should branch on a model name.

Typical use::

    from ada.registry import get_model

    spec = get_model("google/gemma-2-9b-it")
    spec.assistant_header      # '<end_of_turn>\\n<start_of_turn>model\\n'
    spec.probe_layer           # 23
    spec.hook_position         # 'input_layernorm'
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .utils.naming import slugify_model  # single source of truth for path slugs

# configs/ lives next to the ada/ package at the repository root.
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
_MODELS_YAML = _CONFIG_DIR / "models.yaml"


@dataclass(frozen=True)
class ModelSpec:
    """Resolved per-model configuration.

    Attributes:
        hf_id: HuggingFace identifier (the registry key).
        family: Architecture family (``llama``/``gemma``/``qwen``/``mistral``/
            ``deepseek``/``gpt_oss``), used only for grouping and plotting.
        assistant_header: The full assistant header re-injected mid-stream by
            ADA-RK. These are the "Safety Tokens" — the chat-template span that
            separates the user prompt from the assistant response.
        probe_safety_tokens: The header variant injected by ADA-LP before reading
            a hidden state. Its last token is the probe token.
        probe_token: Human-readable name of the probe token (paper Table).
        probe_token_index: Index of the probe token within ``probe_safety_tokens``.
        probe_layer: Transformer block index whose hidden state ADA-LP reads.
        hook_position: Where in the block the hidden state is read
            (``input_layernorm`` is the paper default).
        reasoning: True for reasoning models that emit a ``<think>`` block.
        chat_template_from: If set, borrow this model's chat template.
        generation_prompt_suffix: Extra string appended after
            ``apply_chat_template(add_generation_prompt=True)`` to reach the point
            where the assistant's free-text answer begins (e.g. closing an empty
            reasoning/analysis channel for gpt-oss / DeepSeek-R1). Empty for models
            whose generation prompt already ends there.
    """

    hf_id: str
    family: str
    assistant_header: str
    probe_safety_tokens: str
    probe_token: str
    probe_token_index: int
    probe_layer: int
    hook_position: str = "input_layernorm"
    reasoning: bool = False
    chat_template_from: Optional[str] = None
    generation_prompt_suffix: str = ""
    # Append a single space after the chat-templated user prefix (for templates
    # like Llama-2's that end in ``[/INST]`` with no trailing whitespace). Matches
    # the original probe pipeline's per-model whitelist, so it is opt-in per model.
    chat_prompt_space: bool = False
    # Optional short label for plots/legends (falls back to the HF basename).
    short_name: Optional[str] = None
    # Self-Defense reflection turn: user_header + prompt + reflection_assistant_header.
    user_header: str = ""
    reflection_assistant_header: str = ""
    # Reasoning models only: the assistant header that opens the <think> block,
    # used by ADA-RK and the reflection turn under --reasoning (empty otherwise).
    reasoning_assistant_header: Optional[str] = None

    @property
    def slug(self) -> str:
        """Filesystem-safe identifier used in output paths (``org_Model-Name``)."""
        return slugify_model(self.hf_id)

    @property
    def generation_prompt_completion(self) -> str:
        """Tokens to append after ``apply_chat_template(add_generation_prompt=True)``.

        The registry-driven idiom used everywhere the assistant's free-text answer
        must start: the per-model reasoning/channel suffix, or a single trailing
        space for templates (like Llama-2's ``[/INST]``) that lack one.
        """
        return self.generation_prompt_suffix or (" " if self.chat_prompt_space else "")


@functools.lru_cache(maxsize=1)
def _load_yaml() -> dict:
    with open(_MODELS_YAML, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _decode(value: Optional[str]) -> Optional[str]:
    """Normalize whitespace escapes without disturbing non-ASCII characters.

    YAML double-quoted scalars already turn ``\\n``/``\\t`` into real characters,
    so this is usually a no-op; the targeted replacement only matters for
    single-quoted or block values. Crucially it must NOT run ``unicode_escape``,
    which corrupts multi-byte tokens such as DeepSeek's fullwidth ``｜``.
    """
    if value is None:
        return None
    return value.replace("\\n", "\n").replace("\\t", "\t")


@functools.lru_cache(maxsize=1)
def _registry() -> "dict[str, ModelSpec]":
    raw = _load_yaml()
    defaults = raw.get("defaults", {}) or {}
    specs: dict[str, ModelSpec] = {}
    for entry in raw.get("models", []):
        merged = {**defaults, **entry}
        specs[entry["hf_id"]] = ModelSpec(
            hf_id=entry["hf_id"],
            family=merged["family"],
            assistant_header=_decode(merged["assistant_header"]),
            probe_safety_tokens=_decode(merged["probe_safety_tokens"]),
            probe_token=merged["probe_token"],
            probe_token_index=int(merged["probe_token_index"]),
            probe_layer=int(merged["probe_layer"]),
            hook_position=merged.get("hook_position", "input_layernorm"),
            reasoning=bool(merged.get("reasoning", False)),
            chat_template_from=merged.get("chat_template_from"),
            generation_prompt_suffix=_decode(merged.get("generation_prompt_suffix", "") or ""),
            chat_prompt_space=bool(merged.get("chat_prompt_space", False)),
            short_name=merged.get("short_name"),
            user_header=_decode(merged.get("user_header", "") or ""),
            reflection_assistant_header=_decode(merged.get("reflection_assistant_header", "") or ""),
            reasoning_assistant_header=_decode(merged.get("reasoning_assistant_header")),
        )
    return specs


def get_model(model_name: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``model_name``.

    Raises:
        KeyError: if the model is not declared in ``configs/models.yaml``.
    """
    registry = _registry()
    if model_name not in registry:
        raise KeyError(
            f"'{model_name}' is not in the model registry. "
            f"Add it to configs/models.yaml. Known models: {sorted(registry)}"
        )
    return registry[model_name]


def list_models() -> "list[str]":
    """Return all registered model ids."""
    return list(_registry())


def chat_template_source(model_name: str) -> Optional[str]:
    """Model whose chat template should be borrowed for ``model_name``, if any.

    Also consults the ``deep_alignment_baselines`` block so that base checkpoints
    (e.g. the augmented Llama-2 deep-alignment baseline) resolve correctly.
    """
    raw = _load_yaml()
    for entry in raw.get("deep_alignment_baselines", []):
        if entry["hf_id"] == model_name:
            return entry.get("chat_template_from")
    try:
        return get_model(model_name).chat_template_from
    except KeyError:
        return None


def deep_alignment_baselines() -> "list[dict]":
    """The ``deep_alignment_baselines`` block of ``configs/models.yaml`` (list of dicts)."""
    return list(_load_yaml().get("deep_alignment_baselines", []) or [])


def deep_alignment_base(model_name: Optional[str]) -> Optional[str]:
    """Base model a deep-alignment baseline was built from (else ``None``).

    Single registry accessor for the ``deep_alignment_baselines`` block so callers
    (benign-response reuse, plotting) never re-parse ``models.yaml`` themselves.
    """
    if not model_name:
        return None
    for entry in deep_alignment_baselines():
        if entry.get("hf_id") == model_name:
            return entry.get("base") or entry.get("chat_template_from")
    return None
