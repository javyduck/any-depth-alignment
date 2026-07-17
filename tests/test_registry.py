"""Registry integrity: every model resolves with the fields the pipeline needs."""

from __future__ import annotations

import pytest

from ada.registry import get_model, list_models, slugify_model

EXPECTED_MODELS = {
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410",
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "openai/gpt-oss-120b",
}


def test_all_nine_models_present():
    assert set(list_models()) == EXPECTED_MODELS


@pytest.mark.parametrize("hf_id", sorted(EXPECTED_MODELS))
def test_modelspec_fields(hf_id):
    spec = get_model(hf_id)
    assert spec.hf_id == hf_id
    assert spec.assistant_header, "ADA-RK header must be non-empty"
    assert spec.probe_safety_tokens, "ADA-LP injection must be non-empty"
    assert spec.probe_token
    assert isinstance(spec.probe_token_index, int) and spec.probe_token_index >= 0
    assert isinstance(spec.probe_layer, int) and spec.probe_layer >= 0
    assert spec.hook_position  # defaults to input_layernorm


def test_probe_token_index_points_within_span():
    # Documentation invariant: probe_token_index is a valid index into the span.
    for hf_id in EXPECTED_MODELS:
        spec = get_model(hf_id)
        assert spec.probe_token_index >= 0


def test_decode_resolves_newlines_not_mojibake():
    # gemma header carries a real newline; deepseek keeps the fullwidth bar.
    assert "\n" in get_model("google/gemma-2-2b-it").assistant_header
    ds = get_model("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    assert "｜" in ds.assistant_header  # fullwidth U+FF5C, not mangled ASCII
    assert "\\u" not in ds.assistant_header


def test_qwen_probe_safety_tokens_end_at_assistant():
    # Regression: the probe token must be `assistant`, not a trailing newline.
    spec = get_model("Qwen/Qwen2.5-7B-Instruct")
    assert spec.probe_safety_tokens.endswith("assistant")
    assert not spec.probe_safety_tokens.endswith("\n")


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        get_model("nonexistent/model")


def test_slugify_model():
    assert slugify_model("google/gemma-2-2b-it") == "google_gemma-2-2b-it"
