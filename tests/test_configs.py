"""Config-file integrity: registry, refusal keywords, guardrails."""

from __future__ import annotations

import yaml


def _load(repo_root, name):
    with open(repo_root / "configs" / name) as fh:
        return yaml.safe_load(fh)


def test_models_yaml_shape(repo_root):
    cfg = _load(repo_root, "models.yaml")
    assert len(cfg["models"]) == 9
    for m in cfg["models"]:
        for key in ("hf_id", "family", "assistant_header", "probe_safety_tokens",
                    "probe_token", "probe_token_index", "probe_layer"):
            assert key in m, f"{m.get('hf_id')} missing {key}"
    assert cfg["defaults"]["hook_position"] == "input_layernorm"


def test_deep_alignment_and_closed_source(repo_root):
    cfg = _load(repo_root, "models.yaml")
    bases = {e["hf_id"] for e in cfg["deep_alignment_baselines"]}
    assert any("Llama2" in b or "Llama-2" in b for b in bases)
    assert any("Gemma-2-9B" in b for b in bases)
    assert any("claude" in e["hf_id"] for e in cfg["closed_source"])


def test_refusal_keywords(repo_root):
    kw = _load(repo_root, "refusal_keywords.yaml")
    assert kw["base"] == ["i can't", "i can’t", "i cannot"]
    assert len(kw["ada_rk"]) == 14
    assert kw["self_defense"] == ["yes"]
    assert "unsafe" in kw["guardrail"] and "yes" in kw["guardrail"]
    assert kw["whitelist"] == []


def test_guardrails_yaml(repo_root):
    cfg = _load(repo_root, "guardrails.yaml")
    ids = {g["hf_id"] for g in cfg["guardrails"]}
    assert "meta-llama/Llama-Guard-4-12B" in ids
    assert "ibm-granite/granite-guardian-3.3-8b" in ids
    highlighted = {g["hf_id"] for g in cfg["guardrails"] if g.get("highlighted")}
    assert len(highlighted) == 2  # the two main-text guardrails
