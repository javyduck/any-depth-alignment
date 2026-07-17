"""Shared utilities: filesystem naming, JSON I/O, text matching, and RNG seeding."""

from .io import read_json, read_jsonl, write_json, write_jsonl
from .naming import (
    sanitize_filename,
    slugify_cache,
    slugify_hook_position,
    slugify_mask_tokens,
    slugify_model,
    slugify_safety_tokens,
)
from .seeding import seed_everything
from .text import contains_any

__all__ = [
    "read_json",
    "read_jsonl",
    "write_json",
    "write_jsonl",
    "sanitize_filename",
    "slugify_cache",
    "slugify_hook_position",
    "slugify_mask_tokens",
    "slugify_model",
    "slugify_safety_tokens",
]
