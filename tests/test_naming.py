"""Path-slug helpers — the shared source of truth for on-disk layout."""

from __future__ import annotations

from ada.utils.naming import (
    slugify_cache,
    slugify_hook_position,
    slugify_mask_tokens,
    slugify_model,
    slugify_safety_tokens,
)


def test_slugify_model():
    assert slugify_model("google/gemma-2-2b-it") == "google_gemma-2-2b-it"
    assert slugify_model("Qwen/Qwen2.5-7B-Instruct") == "Qwen_Qwen2_5-7B-Instruct"


def test_slugify_cache():
    assert slugify_cache(True) == "gradual_cache"
    assert slugify_cache(False) == "no_cache"


def test_slugify_mask_none():
    assert slugify_mask_tokens(None) == "mask_token_none"


def test_slugify_hook_position():
    assert slugify_hook_position("input_layernorm") == "hook_input_layernorm"


def test_slugify_safety_tokens_stable():
    slug = slugify_safety_tokens("<end_of_turn>\n<start_of_turn>model")
    assert slug == "safety_token__end_of_turn__n_start_of_turn_model"
    # deterministic
    assert slug == slugify_safety_tokens("<end_of_turn>\n<start_of_turn>model")


def test_slugify_is_filesystem_safe():
    for s in ["<|im_end|>\n<|im_start|>assistant", "[/INST", "<｜Assistant｜>"]:
        slug = slugify_safety_tokens(s)
        assert "/" not in slug and " " not in slug
