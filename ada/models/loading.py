"""Model / tokenizer loading with registry-driven chat-template resolution.

All per-model quirks (e.g. the augmented Llama-2 deep-alignment baseline shipping
without a chat template) are resolved from the registry rather than hard-coded.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..registry import chat_template_source

logger = logging.getLogger(__name__)

# Accepted --dtype choices for the HF-loading entrypoints (collect / evaluate / serve).
DTYPE_CHOICES = ["bfloat16", "float16", "float32", "auto"]


def resolve_torch_dtype(dtype: "torch.dtype | str"):
    """Map a dtype name (or ``torch.dtype``) to a ``torch.dtype``; ``'auto'`` passes through."""
    if isinstance(dtype, torch.dtype):
        return dtype
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "auto": "auto",
    }.get(dtype, torch.bfloat16)


def load_tokenizer(model_name: str) -> "AutoTokenizer":
    """Load a tokenizer, borrowing a chat template from the registry if required."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    template_src = chat_template_source(model_name)
    if template_src and getattr(tokenizer, "chat_template", None) is None:
        logger.info("Borrowing chat template for %s from %s", model_name, template_src)
        tokenizer.chat_template = AutoTokenizer.from_pretrained(
            template_src, trust_remote_code=True
        ).chat_template
    return tokenizer


def load_model_and_tokenizer(
    model_name: str,
    dtype: "torch.dtype | str" = torch.bfloat16,
    device: str = "cuda:0",
    use_flash_attention: bool = True,
):
    """Load an ``AutoModelForCausalLM`` and its tokenizer.

    Falls back gracefully from Flash Attention to the default attention kernel if
    the fast kernel is unavailable, so the same call works across machines.
    """
    tokenizer = load_tokenizer(model_name)

    kwargs = {"torch_dtype": dtype, "device_map": {"": device}, "trust_remote_code": True}
    if use_flash_attention:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation="flash_attention_2", **kwargs
            )
            logger.info("Loaded %s with Flash Attention 2 (%s)", model_name, dtype)
        except (ImportError, ValueError) as err:
            logger.warning("Flash Attention unavailable (%s); using default attention", err)
            model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Loaded %s (%.1fB parameters)", model_name, n_params / 1e9)
    return model, tokenizer


def get_hidden_size(model) -> int:
    cfg = model.config
    for attr in ("hidden_size", "d_model"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    raise ValueError("Could not determine hidden size")


def get_num_layers(model) -> int:
    cfg = model.config
    for attr in ("num_hidden_layers", "num_layers"):
        if hasattr(cfg, attr):
            return getattr(cfg, attr)
    if hasattr(cfg, "text_config"):  # nested configs (e.g. Gemma-3)
        for attr in ("num_hidden_layers", "num_layers"):
            if hasattr(cfg.text_config, attr):
                return getattr(cfg.text_config, attr)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    raise ValueError("Could not determine number of layers")


def gpu_memory_summary() -> str:
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU memory: {alloc:.1f} GB allocated, {reserved:.1f} GB reserved"
    return "GPU not available"
