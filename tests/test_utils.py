"""Shared utility helpers: verdict matching + dtype resolution."""

from __future__ import annotations

import torch

from ada.models.loading import DTYPE_CHOICES, resolve_torch_dtype
from ada.utils.text import contains_any


def test_contains_any_case_insensitive():
    assert contains_any("I Cannot help with that", ["i cannot"]) is True
    assert contains_any("Sure, here is how", ["i cannot", "i can't"]) is False


def test_contains_any_whitelist_veto():
    assert contains_any("I cannot, just kidding", ["i cannot"], ["just kidding"]) is False
    assert contains_any("I cannot", ["i cannot"], ["unrelated"]) is True


def test_resolve_torch_dtype():
    assert resolve_torch_dtype("bfloat16") is torch.bfloat16
    assert resolve_torch_dtype("float16") is torch.float16
    assert resolve_torch_dtype("float32") is torch.float32
    assert resolve_torch_dtype("auto") == "auto"
    assert resolve_torch_dtype(torch.float16) is torch.float16   # passthrough
    assert resolve_torch_dtype("nonsense") is torch.bfloat16     # default
    assert "bfloat16" in DTYPE_CHOICES
