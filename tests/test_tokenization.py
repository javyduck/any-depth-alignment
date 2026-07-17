"""Smoke: each model's probe span tokenizes so the LAST token is the paper's probe token.

ADA-LP reads the hidden state at ``probe_safety_tokens[-1]``; this asserts that
position decodes to the registry/paper ``probe_token`` for every model. Requires the
tokenizer (cached or downloadable); individual models skip if unavailable.

Run with:  ``pytest -m smoke tests/test_tokenization.py``
"""

from __future__ import annotations

import os

import pytest

from ada.registry import get_model, list_models

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


@pytest.mark.smoke
@pytest.mark.parametrize("hf_id", sorted(list_models()))
def test_probe_span_last_token_is_probe_token(hf_id):
    transformers = pytest.importorskip("transformers")
    spec = get_model(hf_id)
    try:
        tok = transformers.AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
    except Exception as e:  # tokenizer not cached / gated without token
        pytest.skip(f"tokenizer unavailable for {hf_id}: {str(e)[:80]}")

    ids = tok.encode(spec.probe_safety_tokens, add_special_tokens=False)
    assert ids, "probe span tokenized to nothing"
    last = tok.decode([ids[-1]]).strip()
    assert last == spec.probe_token.strip() or spec.probe_token.strip() in last, (
        f"{hf_id}: last probe-span token {last!r} != paper probe_token {spec.probe_token!r}"
    )
    # documented index must be the last position (that's what the code reads).
    assert spec.probe_token_index == len(ids) - 1
