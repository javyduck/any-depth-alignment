"""Shared pytest fixtures for the ADA test suite.

Two tiers of tests:
  * unit  (default) — pure logic, no model download / no GPU. Fast.
  * smoke (``-m smoke``) — exercises real tokenizers / CLI entrypoints; heavier.

Run:  ``pytest``  (unit)  ·  ``pytest -m smoke``  ·  ``pytest -m "not smoke"``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))  # make `ada` importable without an install


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def make_log(tmp_path):
    """Write a synthetic per-depth generation/probe log and return its path.

    ``detailed`` is a list of ``{instance, depth, is_refusal}`` records.
    """
    def _make(detailed, total=None, name="log.json", candidate_strings=None):
        p = tmp_path / name
        data = {"detailed_logs": detailed}
        if total is not None:
            data["total_responses"] = total
        if candidate_strings is not None:
            data["candidate_strings"] = candidate_strings
        p.write_text(json.dumps(data))
        return p
    return _make
