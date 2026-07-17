"""E3 — Adversarial attacks: attack-success-rate (ASR) bars and tables.

Reproduces the E3 figures/tables of *Any-Depth Alignment* (ICLR 2026): the
attack-success rate of four automated jailbreaks (GCG, AutoDAN, PAIR, TAP) on the
``advbench`` and ``jailbreakbench`` behaviour sets, for each defense
(Base model, Deep Alignment, Self-Defense/reflection, external guardrails,
ADA-RK, ADA-LP).

Two figures are produced (canonical names for the *first* ``--datasets`` entry,
a ``_{dataset}`` suffix for any others):

* ``attack_main.pdf``                          — per-model grouped bars,
  one panel per model, x = attack, hue = defense.
* ``dual_attack_success_rate_comparison.pdf``  — two panels
  (a) ASR under GCG, (b) mean ASR under the paraphrase attacks
  (AutoDAN, PAIR, TAP); x = model, bars = defense.

It also prints (and writes as CSV) the per-dataset AdvBench / JBB ASR tables.

ASR renormalisation (faithfully reproduced from the notebook)
------------------------------------------------------------
Every ASR uses a **fixed denominator** equal to the size of the attack set
(``advbench`` → 50, ``jailbreakbench`` → 100). Any instance missing from a log is
therefore counted as a *successful defense* (i.e. a refusal), not dropped.

* Base model / Deep Alignment read pre-computed jailbreak transcripts
  ``data/eval/attacks/{dataset}_{attack}/{model_slug}/responses.jsonl`` (release
  layout; ``harmful_responses/…`` is a source fallback) — every non-empty line is
  one *successful* attack, so ASR = ``#lines / total``.
* Every other defense reads a per-depth evaluation log and counts an attack as
  successful iff the response is **never** flagged as a refusal at any depth
  checkpoint; ASR = ``#never-refused / total``.

Input log paths (relative to the run directory)
-----------------------------------------------
* Base / Deep Alignment : ``data/eval/attacks/{dataset}_{attack}/{model_slug}/responses.jsonl`` (``harmful_responses/…`` fallback)
* Self-Defense / ADA-RK : ``vllm_generation_logs/harmful/{dataset}_{attack}/{model_slug}/mode_{reflection|add_safetytoken}/depth_25_maxdepth_3000.json``
* Guardrails            : ``vllm_defense_logs/harmful/{dataset}_{attack}/{guardrail_slug}/{model_slug}/depth_25_maxdepth_3000.json``
* ADA-LP                : ``logs/harmful/{dataset}_{attack}/{model_slug}/{safety_slug}/mask_token_none/hook_input_layernorm/seed_42/logistic/probe-layers{L}/depth_25_maxdepth_3000.json``

All per-model Safety Tokens / probe layers are read from :mod:`ada.registry`; the
Deep-Alignment baseline for each model is read from the
``deep_alignment_baselines`` block of ``configs/models.yaml``.

Run with::

    python -m ada.plotting.plot_e3_attacks
    python -m ada.plotting.plot_e3_attacks --models meta-llama/Llama-2-7b-chat-hf \
        --datasets advbench jailbreakbench --output-dir figures
"""

from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import Patch

from .. import registry
from ..registry import get_model, list_models, slugify_model
from ..utils.naming import slugify_safety_tokens

# Shared log parsing lives in ada.plotting._common (see the E-plotting package).
# ``parse_refusal_curve(log_path) -> {depth: cumulative_refusal_rate}`` where the
# rate at depth d is (#instances that have refused at any checkpoint <= d) /
# (#instances present in the log).
from ._common import cumulative_refusal_curve as parse_refusal_curve

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
ATTACK_TYPES = ["gcg", "autodan", "pair", "tap"]

# Attack-set sizes → the fixed ASR denominator (missing instances = refusals).
DATASET_TOTALS = {"advbench": 50, "jailbreakbench": 100}

# The two external guardrail baselines shown in the paper's E3 figures.
DEFAULT_GUARDRAILS = [
    "meta-llama/Llama-Guard-4-12B",
    "ibm-granite/granite-guardian-3.3-8b",
]

# Models with full E3 attack coverage (used when --models is not given).
DEFAULT_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "google/gemma-2-9b-it",
]

DEPTH = 25
MAX_DEPTH = 3000

# Cosmetic per-defense styling. Guardrails not listed here fall back to a
# generated display name + palette colour.
GUARDRAIL_STYLE = {
    "meta-llama/Llama-Guard-4-12B": ("Meta Llama-Guard-4-12B", "#2F80ED", "+++"),
    "ibm-granite/granite-guardian-3.3-8b": ("IBM Granite-Guardian-3.3-8b", "#6F5ACD", "xxx"),
}
_FALLBACK_GUARDRAIL_COLORS = ["#8C6D31", "#B279A2", "#9D755D", "#BAB0AC"]


# --------------------------------------------------------------------------- #
# Deep-alignment baseline lookup (from configs/models.yaml, not hard-coded)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _deep_alignment_map() -> Dict[str, str]:
    """Map ``base model hf_id -> deep-alignment checkpoint hf_id``."""
    cfg = Path(registry.__file__).resolve().parent.parent / "configs" / "models.yaml"
    with open(cfg, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return {
        entry["base"]: entry["hf_id"]
        for entry in raw.get("deep_alignment_baselines", []) or []
        if entry.get("base")
    }


def deep_alignment_model(model: str) -> Optional[str]:
    return _deep_alignment_map().get(model)


# --------------------------------------------------------------------------- #
# Method (defense) registry
# --------------------------------------------------------------------------- #
def build_methods(guardrails: List[str]) -> List[dict]:
    """Ordered list of defense descriptors used for both plots and tables."""
    methods: List[dict] = [
        dict(name="Base Model", kind="base", color="#4C78A8", hatch=""),
        dict(name="Deep Alignment", kind="deep_alignment", color="#17AEAF", hatch="///"),
        dict(name="Self Defense", kind="self_defense", color="#54A24B", hatch="..."),
    ]
    for i, gid in enumerate(guardrails):
        if gid in GUARDRAIL_STYLE:
            disp, color, hatch = GUARDRAIL_STYLE[gid]
        else:
            disp = gid.split("/")[-1]
            color = _FALLBACK_GUARDRAIL_COLORS[i % len(_FALLBACK_GUARDRAIL_COLORS)]
            hatch = "\\\\\\"
        methods.append(dict(name=disp, kind="guardrail", guardrail=gid, color=color, hatch=hatch))
    methods += [
        dict(name="ADA (RK)", kind="ada_rk", color="#F26B21", hatch="|||"),
        dict(name="ADA (LP)", kind="ada_lp", color="#D62F2F", hatch="---"),
    ]
    return methods


# --------------------------------------------------------------------------- #
# Log-path builders (relative to the run directory)
# --------------------------------------------------------------------------- #
def _dataset_attack(dataset: str, attack: str) -> str:
    return f"{dataset.lower()}_{attack.lower()}"


def _attack_response_path(dataset: str, attack: str, model_slug: str) -> Path:
    """Adversarial-attack transcript for a model, preferring the release layout.

    Mirrors ``ada.rethink.generate.find_response_file``: the released corpora live
    under ``data/eval/attacks/`` (written by ``ada.attacks.extract`` / copied by
    ``prepare_datasets.sh``); the original ``harmful_responses/`` tree is a fallback
    so pre-existing source artifacts still resolve. Returns the first existing
    candidate, else the release path (so ``asr_from_jsonl`` reports 0 uniformly).
    """
    name = _dataset_attack(dataset, attack)
    candidates = [
        Path("data/eval/attacks") / name / model_slug / "responses.jsonl",
        Path("harmful_responses") / name / model_slug / "responses.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def base_response_path(dataset: str, attack: str, model: str) -> Path:
    return _attack_response_path(dataset, attack, slugify_model(model))


def deep_alignment_response_path(dataset: str, attack: str, model: str) -> Optional[Path]:
    da = deep_alignment_model(model)
    if da is None:
        return None
    return _attack_response_path(dataset, attack, slugify_model(da))


def generation_log_path(dataset: str, attack: str, model: str, mode: str, split: str = "harmful") -> Path:
    return (
        Path("vllm_generation_logs")
        / split
        / _dataset_attack(dataset, attack)
        / slugify_model(model)
        / f"mode_{mode}"
        / f"depth_{DEPTH}_maxdepth_{MAX_DEPTH}.json"
    )


def guardrail_log_path(dataset: str, attack: str, model: str, guardrail: str, split: str = "harmful") -> Path:
    return (
        Path("vllm_defense_logs")
        / split
        / _dataset_attack(dataset, attack)
        / slugify_model(guardrail)
        / slugify_model(model)
        / f"depth_{DEPTH}_maxdepth_{MAX_DEPTH}.json"
    )


def ada_lp_log_path(dataset: str, attack: str, model: str, split: str = "harmful") -> Path:
    spec = get_model(model)
    return (
        Path("logs")
        / split
        / _dataset_attack(dataset, attack)
        / slugify_model(model)
        / slugify_safety_tokens(spec.probe_safety_tokens)
        / "mask_token_none"
        / "hook_input_layernorm"
        / "seed_42"
        / "logistic"
        / f"probe-layers{spec.probe_layer}"
        / f"depth_{DEPTH}_maxdepth_{MAX_DEPTH}.json"
    )


# --------------------------------------------------------------------------- #
# ASR computation
# --------------------------------------------------------------------------- #
def asr_from_jsonl(path: Path, total: int) -> float:
    """ASR from a jailbreak-transcript file: #non-empty lines / total."""
    if not path.exists():
        return 0.0
    with open(path, "r", encoding="utf-8") as fh:
        successful = sum(1 for line in fh if line.strip())
    return successful / total


def _count_instances(path: Path) -> int:
    """Number of distinct instances present in a per-depth log.

    This matches the denominator used by ``parse_refusal_curve`` (the "#instances"
    in the refusal-rate definition), so the two combine consistently below.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            logs = json.load(fh).get("detailed_logs", [])
    except (OSError, ValueError):
        return 0
    seen = set()
    for row_idx, entry in enumerate(logs):
        if entry.get("depth", 0) > 0:
            inst = entry.get("instance")
            if inst is None:
                inst = f"_row_{row_idx}"
            seen.add(inst)
    return len(seen)


def asr_from_generation_log(path: Path, total: int) -> float:
    """ASR from a per-depth refusal log with the fixed-denominator renormalisation.

    An attack succeeds iff the response is never flagged as a refusal at any
    checkpoint. Using the cumulative refusal curve, the fraction that *ever*
    refuse is its value at the deepest checkpoint; #never-refused = present *
    (1 - that), and ASR divides by the fixed attack-set ``total`` so missing
    instances count as refusals.
    """
    if not path.exists():
        return 0.0
    try:
        curve = parse_refusal_curve(path)
    except Exception:  # noqa: BLE001 - a malformed/empty log means no successes
        return 0.0
    n_present = _count_instances(path)
    if n_present == 0:
        return 0.0
    ever_refused_rate = curve[max(curve)] if curve else 0.0
    never_refused = round(n_present * (1.0 - ever_refused_rate))
    return never_refused / total


def individual_asr(method: dict, model: str, attack: str, dataset: str, total: int, split: str) -> Optional[float]:
    """ASR for one (defense, model, attack). ``None`` = defense unavailable."""
    kind = method["kind"]
    if kind == "base":
        return asr_from_jsonl(base_response_path(dataset, attack, model), total)
    if kind == "deep_alignment":
        path = deep_alignment_response_path(dataset, attack, model)
        if path is None:
            return None  # no deep-alignment checkpoint for this model
        return asr_from_jsonl(path, total)
    if kind == "self_defense":
        return asr_from_generation_log(generation_log_path(dataset, attack, model, "reflection", split), total)
    if kind == "ada_rk":
        return asr_from_generation_log(generation_log_path(dataset, attack, model, "add_safetytoken", split), total)
    if kind == "guardrail":
        return asr_from_generation_log(guardrail_log_path(dataset, attack, model, method["guardrail"], split), total)
    if kind == "ada_lp":
        return asr_from_generation_log(ada_lp_log_path(dataset, attack, model, split), total)
    raise ValueError(f"Unknown method kind: {kind}")


def method_asr(method: dict, model: str, attacks: List[str], dataset: str, total: int, split: str) -> Optional[float]:
    """Mean ASR of a defense over ``attacks``. ``None`` if unavailable (no baseline)."""
    values = [individual_asr(method, model, at, dataset, total, split) for at in attacks]
    if any(v is None for v in values):
        return None
    return float(np.mean(values)) if values else 0.0


# --------------------------------------------------------------------------- #
# Data collection
# --------------------------------------------------------------------------- #
def collect_individual(models, methods, dataset, total, split):
    """{model: {method_name: {attack: asr|None}}}."""
    out = {}
    for model in models:
        out[model] = {}
        for method in methods:
            out[model][method["name"]] = {
                at: individual_asr(method, model, at, dataset, total, split) for at in ATTACK_TYPES
            }
    return out


def model_has_data(model: str, dataset: str) -> bool:
    """True if any base-model jailbreak transcript exists for this dataset."""
    return any(base_response_path(dataset, at, model).exists() for at in ATTACK_TYPES)


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def build_asr_table(models, methods, dataset, total, split) -> pd.DataFrame:
    """Tidy ASR table (%) with (Model, Method) rows and attack + Average columns."""
    rows = []
    for model in models:
        short = model.split("/")[-1]
        for method in methods:
            record = {"Model": short, "Method": method["name"]}
            per_attack = []
            for at in ATTACK_TYPES:
                v = individual_asr(method, model, at, dataset, total, split)
                record[at.upper() if at != "autodan" else "AutoDAN"] = None if v is None else round(v * 100, 1)
                if v is not None:
                    per_attack.append(v)
            record["Average"] = round(float(np.mean(per_attack)) * 100, 1) if per_attack else None
            rows.append(record)
    return pd.DataFrame(rows)


def print_and_save_table(df: pd.DataFrame, dataset: str, output_dir: Path) -> None:
    print("=" * 100)
    print(f"ATTACK SUCCESS RATE (%) — {dataset}")
    print("=" * 100)
    with pd.option_context("display.max_columns", None, "display.width", None):
        print(df.to_string(index=False, na_rep="N/A"))
    print()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"attack_asr_{dataset}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path.resolve()}")


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_attack_main(models, methods, dataset, total, split, out_path: Path) -> None:
    """Per-model grouped bars (x = attack, hue = defense)."""
    attack_labels = {"gcg": "GCG", "autodan": "AutoDAN", "pair": "PAIR", "tap": "TAP"}
    data = collect_individual(models, methods, dataset, total, split)

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5.5), sharey=True, squeeze=False)
    axes = axes[0]

    x = np.arange(len(ATTACK_TYPES))
    # Only methods that have at least one non-None value on this model are drawn.
    for ax, model in zip(axes, models):
        drawn = [m for m in methods if any(data[model][m["name"]][at] is not None for at in ATTACK_TYPES)]
        width = 0.8 / max(len(drawn), 1)
        for i, method in enumerate(drawn):
            offset = (i - len(drawn) / 2 + 0.5) * width
            heights = [
                (data[model][method["name"]][at] or 0.0) * 100 for at in ATTACK_TYPES
            ]
            bars = ax.bar(
                x + offset, heights, width,
                color=method["color"], hatch=method["hatch"],
                edgecolor="black", linewidth=1.0, label=method["name"],
            )
            ax.bar_label(bars, fmt="%.0f", padding=2, fontsize=8)
        ax.set_title(model.split("/")[-1], fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels([attack_labels[a] for a in ATTACK_TYPES])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Attack Success Rate (%)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)
    axes[-1].legend(fontsize=9, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.resolve()}")


def plot_dual(models, methods, dataset, total, split, out_path: Path) -> None:
    """Two panels: (a) ASR under GCG, (b) mean ASR under AutoDAN/PAIR/TAP."""
    gcg = {m: {me["name"]: method_asr(me, m, ["gcg"], dataset, total, split) for me in methods} for m in models}
    multi = {
        m: {me["name"]: method_asr(me, m, ["autodan", "pair", "tap"], dataset, total, split) for me in methods}
        for m in models
    }
    short = [m.split("/")[-1] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9 * max(len(models), 1), 8))

    def panel(ax, data, title):
        x = np.arange(len(models))
        width = 0.8 / max(len(methods), 1)
        for i, method in enumerate(methods):
            offset = (i - len(methods) / 2 + 0.5) * width
            for j, model in enumerate(models):
                value = data[model][method["name"]]
                if value is None:
                    ax.text(x[j] + offset, 0.02, "N/A", ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color="gray", rotation=90)
                    continue
                ax.bar(x[j] + offset, value, width, color=method["color"], hatch=method["hatch"],
                       edgecolor="black", linewidth=1.0)
                ax.text(x[j] + offset, value + 0.01, f"{value * 100:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=15, pad=12)
        ax.set_ylabel("Attack Success Rate (ASR)")
        ax.set_xticks(x)
        ax.set_xticklabels(short)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    panel(ax1, gcg, "(a) ASR under GCG attack")
    panel(ax2, multi, "(b) Mean ASR under paraphrase attacks (AutoDAN, PAIR, TAP)")

    legend = [Patch(facecolor=m["color"], hatch=m["hatch"], edgecolor="black", label=m["name"]) for m in methods]
    fig.legend(handles=legend, loc="upper center", ncol=len(legend), fontsize=11,
               frameon=True, edgecolor="black")
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path.resolve()}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def resolve_models(requested: Optional[List[str]], dataset: str) -> List[str]:
    if requested:
        models = requested
    else:
        # Prefer the paper's main-figure models; if none of them have data, fall
        # back to every registered model that has an attack transcript.
        models = [m for m in DEFAULT_MODELS if model_has_data(m, dataset)]
        if not models:
            models = [m for m in list_models() if model_has_data(m, dataset)]
    kept = [m for m in models if model_has_data(m, dataset)]
    for m in models:
        if m not in kept:
            print(f"[skip] no attack data for {m} on {dataset}")
    return kept


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=None,
                        help="HF model ids (default: paper E3 models with data).")
    parser.add_argument("--datasets", nargs="+", default=["advbench", "jailbreakbench"],
                        help="Attack behaviour sets (first one gets the canonical figure names).")
    parser.add_argument("--guardrails", nargs="+", default=DEFAULT_GUARDRAILS,
                        help="Guardrail baseline HF ids.")
    parser.add_argument("--split", default="harmful", help="Log split (attacks are always harmful).")
    parser.add_argument("--output-dir", default="figures", type=Path)
    args = parser.parse_args(argv)

    methods = build_methods(args.guardrails)

    for i, dataset in enumerate(args.datasets):
        total = DATASET_TOTALS.get(dataset)
        if total is None:
            raise ValueError(
                f"Unknown attack-set size for '{dataset}'. Known: {sorted(DATASET_TOTALS)}. "
                "Add it to DATASET_TOTALS."
            )
        models = resolve_models(args.models, dataset)
        if not models:
            print(f"[skip] no models with data for {dataset}")
            continue

        # ASR table.
        df = build_asr_table(models, methods, dataset, total, args.split)
        print_and_save_table(df, dataset, args.output_dir)

        # Figures: canonical names for the first dataset, suffixed otherwise.
        suffix = "" if i == 0 else f"_{dataset}"
        plot_attack_main(models, methods, dataset, total, args.split,
                         args.output_dir / f"attack_main{suffix}.pdf")
        plot_dual(models, methods, dataset, total, args.split,
                  args.output_dir / f"dual_attack_success_rate_comparison{suffix}.pdf")


if __name__ == "__main__":
    main()
