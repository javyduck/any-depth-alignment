#!/usr/bin/env python3
"""Curate a slim, re-plottable subset of evaluation logs for release as examples.

The full logs are hundreds of GB (probe eval swept over every layer/token, plus
per-instance harmful ``generated_text``). This selects only what the main figures
need and strips the harmful text, producing a compact subset that lets
``ada.plotting.*`` regenerate the figures without re-running any inference:

  * probe logs (ADA-LP): only the canonical (safety-token, probe layer) per model
  * generation logs (Base / ADA-RK / Self-Defense): with ``generated_text`` removed
  * defense logs: only the two highlighted guardrails, text removed
  * figures: the paper PDFs

Text-free logs contain only per-instance {depth, is_refusal, refusal_probability}
keyed by integer index — no prompts or completions — so they carry no dataset
content (safe even for the HEx-PHI split).

Usage:  SRC=/path/to/SafetyToken OUT=/path/to/staging python scripts/build_example_results.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from ada.registry import get_model, list_models
from ada.utils.naming import slugify_model, slugify_safety_tokens

SRC = Path(os.environ.get("SRC", "/data1/common/jiawei/SafetyToken"))
OUT = Path(os.environ.get("OUT", "/tmp/ada_example_results"))
# Keep the example subset small: only the main-experiment logs (base models, no
# SFT-attack SFT-adapter sweep). Set INCLUDE_ADAPTER=1 to also include the adapter logs.
INCLUDE_ADAPTER = os.environ.get("INCLUDE_ADAPTER", "0") == "1"

# Keys kept in each detailed_logs entry. Dropping generated_text (harmful text)
# and refusal_probability (only the exploratory threshold sweep needs it) keeps
# the subset small; every main figure only needs {instance, depth, is_refusal}.
KEEP_KEYS = ("instance", "depth", "is_refusal")


def _strip_and_copy(src: Path, dst: Path) -> None:
    """Copy a log JSON, dropping generated_text from detailed_logs."""
    data = json.loads(src.read_text())
    if isinstance(data.get("detailed_logs"), list):
        data["detailed_logs"] = [
            {k: e[k] for k in KEEP_KEYS if k in e} for e in data["detailed_logs"]
        ]
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(data))


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def curate_probe_logs() -> int:
    """Canonical (safety-token, probe layer) probe logs per model.

    Matches both the base model dir and its SFT-adapter variants
    (``{slug}``, ``{slug}-{type}-adapter-{step}[-disable_safetytoken]``) so the
    SFT-attack ADA-LP curves are included too. Layout:
    ``logs/{split}/{dataset}/{model_dir}/{safety}/{mask}/{hook}/seed/logistic/probe-layers{L}/...``
    """
    n = 0
    canon = {
        slugify_model(m): (
            slugify_safety_tokens(get_model(m).probe_safety_tokens),
            f"probe-layers{get_model(m).probe_layer}",
        )
        for m in list_models()
    }
    root = SRC / "logs"
    for path in root.rglob("depth_*.json"):
        parts = path.relative_to(root).parts
        if len(parts) < 3:
            continue
        model_dir = parts[2]
        s = str(path)
        for slug, (safety_slug, layer_seg) in canon.items():
            if not INCLUDE_ADAPTER and "-adapter-" in model_dir:
                continue
            if (model_dir == slug or model_dir.startswith(slug + "-")) \
                    and f"/{safety_slug}/" in s and f"/{layer_seg}/" in s:
                _strip_and_copy(path, OUT / "logs" / path.relative_to(root))
                n += 1
                break
    return n


def curate_generation_logs() -> int:
    n = 0
    root = SRC / "vllm_generation_logs"
    for path in root.rglob("depth_*.json"):
        if not INCLUDE_ADAPTER and "-adapter-" in str(path):
            continue
        _strip_and_copy(path, OUT / "vllm_generation_logs" / path.relative_to(root))
        n += 1
    return n


def curate_defense_logs() -> int:
    n = 0
    highlighted = ("Llama-Guard-4-12B", "granite-guardian-3.3-8b")
    for path in (SRC / "vllm_defense_logs").rglob("depth_*.json"):
        if any(h in str(path) for h in highlighted):
            _strip_and_copy(path, OUT / "vllm_defense_logs" / path.relative_to(SRC / "vllm_defense_logs"))
            n += 1
    return n


def copy_figures() -> int:
    src_fig = SRC / "figures"
    if not src_fig.exists():
        return 0
    n = 0
    for pdf in src_fig.glob("*.pdf"):
        _copy(pdf, OUT / "figures" / pdf.name)
        n += 1
    return n


def main() -> None:
    print(f"[curate] SRC={SRC}  OUT={OUT}")
    print(f"[curate] probe logs:      {curate_probe_logs()}")
    print(f"[curate] generation logs: {curate_generation_logs()} (text stripped)")
    print(f"[curate] defense logs:    {curate_defense_logs()} (text stripped)")
    print(f"[curate] figures:         {copy_figures()}")


if __name__ == "__main__":
    main()
