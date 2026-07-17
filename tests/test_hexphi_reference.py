"""HEx-PHI compliant sharing: export (prompt -> hash) then reconstruct round-trips."""

from __future__ import annotations

from pathlib import Path

from ada.datagen import hexphi_reference as hx
from ada.utils.io import read_jsonl, write_jsonl

FAKE_PROMPTS = [f"licensed hexphi prompt number {i}" for i in range(5)]


def test_prompt_hash_deterministic_and_opaque():
    h = hx._prompt_hash("hello")
    assert h == hx._prompt_hash(" hello ")          # stripped
    assert len(h) == 64 and "hello" not in h        # sha-256 hex, no plaintext


def test_export_reconstruct_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(hx, "load_hexphi", lambda: FAKE_PROMPTS)

    responses = tmp_path / "hexphi_responses.jsonl"
    write_jsonl(responses, [
        {"messages": [{"role": "user", "content": p},
                      {"role": "assistant", "content": f"harmful continuation {i}"}]}
        for i, p in enumerate(FAKE_PROMPTS)
    ])

    references = tmp_path / "hexphi_references.jsonl"
    hx.export_references(responses, references)

    # references carry NO prompt text — only hashes + our continuations.
    refs = read_jsonl(references)
    assert len(refs) == 5
    blob = references.read_text()
    for p in FAKE_PROMPTS:
        assert p not in blob
    assert all(set(r) == {"hexphi_sha256", "hexphi_index", "response"} for r in refs)

    # reconstruct against the (fake) licensed copy -> identical to the original.
    rebuilt = tmp_path / "rebuilt.jsonl"
    hx.reconstruct_responses(references, rebuilt)
    orig = read_jsonl(responses)
    got = read_jsonl(rebuilt)
    assert got == orig
