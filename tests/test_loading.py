"""Shared response-file resolver (used by evaluate / generate / guardrails)."""

from __future__ import annotations

import pytest

from ada.data.loading import resolve_response_file


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}\n")
    return p


def test_deep_prefill_release_layout(tmp_path):
    root = tmp_path / "eval"
    _touch(root / "deep_prefill" / "advbench_responses.jsonl")
    got = resolve_response_file("advbench", data_root=root)
    assert got.name == "advbench_responses.jsonl"


def test_attack_and_benign_layout(tmp_path):
    root = tmp_path / "eval"
    _touch(root / "attacks" / "advbench_gcg" / "google_gemma-2-9b-it" / "responses.jsonl")
    _touch(root / "over_refusal" / "mmlu" / "google_gemma-2-9b-it" / "responses.jsonl")
    a = resolve_response_file("advbench", "google/gemma-2-9b-it", attack="gcg", data_root=root)
    b = resolve_response_file("mmlu", "google/gemma-2-9b-it", benign=True, data_root=root)
    assert "attacks/advbench_gcg" in str(a)
    assert "over_refusal/mmlu" in str(b)


def test_explicit_response_file_wins(tmp_path):
    f = _touch(tmp_path / "custom.jsonl")
    assert resolve_response_file("advbench", response_file=str(f)) == f


def test_missing_raises_with_hexphi_hint(tmp_path):
    with pytest.raises(FileNotFoundError) as e:
        resolve_response_file("hexphi", data_root=tmp_path / "eval")
    assert "HEx-PHI is not distributed" in str(e.value)
