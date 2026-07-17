"""Small text helpers shared across refusal / guardrail verdict detection."""

from __future__ import annotations

from typing import List, Optional


def contains_any(text: str, candidates: List[str], whitelist: Optional[List[str]] = None) -> bool:
    """Case-insensitive substring match: True if any candidate is in ``text``.

    A ``whitelist`` match vetoes the result (returns False) — used so a benign
    phrase can suppress an otherwise-matching refusal/block verdict.
    """
    low = text.lower()
    if whitelist:
        if any(w.lower() in low for w in whitelist):
            return False
    return any(c.lower() in low for c in candidates)
