"""First-refusal-position statistics for ADA-LP.

Companion to :mod:`ada.plotting.plot_adversarial_attacks`. Where the attack figures show
*whether* an attack is stopped, this table shows *where*: the token depth at
which ADA-LP's linear probe first flags a refusal, summarised per (model, attack)
over the instances that are ever flagged.

For each (model, attack) it reports, over the instances ADA-LP eventually refuses:

* Count            — number of instances flagged at least once,
* Mean / 25% / 50% (median) / 75% — of the first-refusal token depth,
* Refusal Rate (%) — fraction of *present* attack instances ever flagged
  (from the shared cumulative refusal curve).

Only ADA-LP is analysed (it is the defense that exposes a per-depth probe
decision). Depths are the evaluation checkpoints (every 25 tokens).

Input logs (relative to the run directory)::

    logs/harmful/{dataset}_{attack}/{model_slug}/{safety_slug}/mask_token_none/
        hook_input_layernorm/seed_42/logistic/probe-layers{L}/depth_25_maxdepth_3000.json

The Safety-Token span (``safety_slug``) and probe layer ``L`` come from
:mod:`ada.registry`; they are never hard-coded here.

Run with::

    python -m ada.plotting.tables_adversarial_attacks
    python -m ada.plotting.tables_adversarial_attacks --datasets advbench jailbreakbench --output-dir figures
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..registry import get_model, list_models, slugify_model
from ..utils.naming import slugify_safety_tokens

# Shared log parsing (see ada.plotting._common):
# ``parse_refusal_curve(log_path) -> {depth: cumulative_refusal_rate}`` where the
# rate at depth d is (#instances refused at any checkpoint <= d) / (#instances).
from ._common import cumulative_refusal_curve as parse_refusal_curve

ATTACK_TYPES = ["gcg", "autodan", "pair", "tap"]
INTERVAL = 25  # checkpoints are multiples of this
DEPTH = 25
MAX_DEPTH = 3000

# The adversarial-attack attack-evaluated models (used when --models is not given).
DEFAULT_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Ministral-8B-Instruct-2410",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def ada_lp_log_path(dataset: str, attack: str, model: str, split: str = "harmful") -> Path:
    """ADA-LP per-depth log path; slug + layer resolved from the registry."""
    spec = get_model(model)
    return (
        Path("logs")
        / split
        / f"{dataset.lower()}_{attack.lower()}"
        / slugify_model(model)
        / slugify_safety_tokens(spec.probe_safety_tokens)
        / "mask_token_none"
        / "hook_input_layernorm"
        / "seed_42"
        / "logistic"
        / f"probe-layers{spec.probe_layer}"
        / f"depth_{DEPTH}_maxdepth_{MAX_DEPTH}.json"
    )


def first_refusal_depths(log_path: Path, interval: int = INTERVAL) -> Dict[object, int]:
    """{instance_id: depth of its first refusal}, over checkpoints only.

    Faithful to the notebook: group entries per instance at depths that are
    positive multiples of ``interval``, then take the smallest depth flagged as a
    refusal. Instances never flagged are omitted.
    """
    if not log_path.exists():
        return {}
    try:
        with open(log_path, "r", encoding="utf-8") as fh:
            logs = json.load(fh).get("detailed_logs", [])
    except (OSError, ValueError):
        return {}

    per_instance: Dict[object, Dict[int, bool]] = defaultdict(dict)
    for row_idx, entry in enumerate(logs):
        depth = entry.get("depth", 0)
        if depth % interval == 0 and depth > 0:
            inst = entry.get("instance")
            if inst is None:
                inst = f"_row_{row_idx}"
            per_instance[inst][depth] = bool(entry.get("is_refusal", False))

    firsts: Dict[object, int] = {}
    for inst, depths in per_instance.items():
        for depth in sorted(depths):
            if depths[depth]:
                firsts[inst] = depth
                break
    return firsts


def refusal_stats(log_path: Path, interval: int = INTERVAL) -> Optional[dict]:
    """First-refusal-depth statistics for one ADA-LP log, or ``None`` if empty."""
    firsts = first_refusal_depths(log_path, interval)
    if not firsts:
        return None
    depths = list(firsts.values())
    # Overall refusal rate (fraction of present instances ever flagged) comes from
    # the shared cumulative curve rather than being recomputed here.
    try:
        curve = parse_refusal_curve(log_path)
        refusal_rate = curve[max(curve)] if curve else float("nan")
    except Exception:  # noqa: BLE001
        refusal_rate = float("nan")
    return {
        "count": len(depths),
        "mean": float(np.mean(depths)),
        "25%": float(np.percentile(depths, 25)),
        "50%": float(np.percentile(depths, 50)),
        "75%": float(np.percentile(depths, 75)),
        "refusal_rate": refusal_rate,
    }


def build_summary(models: List[str], dataset: str, split: str) -> pd.DataFrame:
    """Combined (Model, Attack) summary table of first-refusal statistics."""
    rows = []
    for model in models:
        short = model.split("/")[-1]
        for attack in ATTACK_TYPES:
            stats = refusal_stats(ada_lp_log_path(dataset, attack, model, split))
            if stats is None:
                rows.append({
                    "Model": short, "Attack": attack, "Count": 0,
                    "Mean": np.nan, "25%": np.nan, "50% (Median)": np.nan,
                    "75%": np.nan, "Refusal Rate (%)": np.nan,
                })
            else:
                rows.append({
                    "Model": short, "Attack": attack, "Count": int(stats["count"]),
                    "Mean": round(stats["mean"], 1), "25%": round(stats["25%"], 1),
                    "50% (Median)": round(stats["50%"], 1), "75%": round(stats["75%"], 1),
                    "Refusal Rate (%)": round(stats["refusal_rate"] * 100, 1),
                })
    return pd.DataFrame(rows)


def resolve_models(requested: Optional[List[str]], dataset: str, split: str) -> List[str]:
    models = requested if requested else DEFAULT_MODELS
    kept = []
    for m in models:
        if any(ada_lp_log_path(dataset, at, m, split).exists() for at in ATTACK_TYPES):
            kept.append(m)
        else:
            print(f"[skip] no ADA-LP logs for {m} on {dataset}")
    if not kept and not requested:
        kept = [m for m in list_models()
                if any(ada_lp_log_path(dataset, at, m, split).exists() for at in ATTACK_TYPES)]
    return kept


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None,
                        help="HF model ids (default: the adversarial-attack attack-evaluated models).")
    parser.add_argument("--datasets", nargs="+", default=["advbench"],
                        help="Attack behaviour sets.")
    parser.add_argument("--split", default="harmful", help="Log split (attacks are always harmful).")
    parser.add_argument("--output-dir", default="figures", type=Path)
    args = parser.parse_args(argv)

    for dataset in args.datasets:
        models = resolve_models(args.models, dataset, args.split)
        if not models:
            print(f"[skip] no models with ADA-LP logs for {dataset}")
            continue

        df = build_summary(models, dataset, args.split)
        print("=" * 110)
        print(f"ADA-LP FIRST-REFUSAL DEPTH (tokens) — {dataset}")
        print("Over instances flagged at least once; only ADA-LP exposes a per-depth probe decision.")
        print("=" * 110)
        with pd.option_context("display.max_columns", None, "display.width", None,
                               "display.float_format", lambda x: f"{x:.1f}" if pd.notna(x) else "N/A"):
            print(df.to_string(index=False, na_rep="N/A"))
        print()

        args.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_dir / f"first_refusal_depth_{dataset}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[saved] {csv_path.resolve()}")


if __name__ == "__main__":
    main()
