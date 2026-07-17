"""deep-prefill refusal curves: how well each defense survives an adversarial
assistant prefill of increasing depth.

Every panel/curve plots *refusal rate vs. prefill depth* (token position at which
the harmful assistant prefix is cut and generation resumes). Higher is better:
a robust defense keeps refusing even when the model is force-fed a long harmful
continuation. The refusal rate at depth ``d`` is the *independent* per-depth
rate: the fraction of instances that refuse when the prefill is cut at exactly
``d`` tokens. (This is the deep-prefill notebook's definition and is deliberately NOT the
cumulative ``_common.parse_refusal_curve`` — under deep prefill a base model's
per-depth refusal drops with depth, which is precisely what this figure exposes;
the cumulative curve would mask it. See :func:`_refusal_curve`.)

Figures produced (written under ``--output-dir``, default ``figures/``):

* ``all_models_refusal_rates.pdf`` — grid of every ``--models`` model, each panel
  showing all defense methods, refusal rate **averaged over the four harmful
  datasets** (advbench / jailbreakbench / hexphi / strongreject). Averaging is an
  unweighted mean of the per-dataset rates at each depth.
* ``all_models_refusal_rates_{dataset}.pdf`` — the same grid for one dataset.
* ``prefill_all_model.pdf`` — base-model-only refusal vs. depth (0..500) across
  the model zoo incl. gpt-oss-120b and Claude (the motivating figure).
* ``prefill_base_vs_dia.pdf`` — per-model Base (dashed) vs. ADA (solid) at 0..500;
  ADA is ADA-LP (linear probe) for open models and ADA-RK for Claude.
* ``claude_{dataset}.pdf`` — Claude Base Model vs. ADA-RK over the full depth range.

It also prints the depth-500 "Table 1" refusal numbers (per-model and averaged).

Methods (see ``METHOD_SPECS``):
    Base Model               -> mode_empty generation log
    Deep Alignment           -> mode_empty log of the deep-alignment baseline ckpt
    Self Defense             -> mode_reflection generation log
    ADA (RK)                 -> mode_add_safetytoken generation log
    ADA (LP)                 -> logistic probe log at the registry probe layer
    Meta Llama-Guard-4-12B   -> external guardrail defense log
    IBM Granite-Guardian-3.3 -> external guardrail defense log

Input log layouts are the ones documented in :mod:`ada.plotting._common`; all
paths are resolved relative to ``--log-root`` (default: current directory).

Run as::

    python -m ada.plotting.plot_deep_prefill
    python -m ada.plotting.plot_deep_prefill --models google/gemma-2-9b-it openai/gpt-oss-120b
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

import ada.registry as _registry
from ada.registry import list_models
from ada.utils.io import read_json

from ._common import (
    ensure_output_dir,
    find_defense_log,
    find_generation_log,
    find_probe_log,
    probe_layer,
    probe_safety_tokens,
    short_model_name,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# The four harmful datasets deep-prefill averages over (unweighted mean of per-dataset
# refusal rates at each depth).
DEFAULT_DATASETS: List[str] = ["advbench", "jailbreakbench", "hexphi", "strongreject"]

# Per-depth log filenames are ``depth_{STEP}_maxdepth_{MAXDEPTH}.json``.
DEPTH_STEP = 25
MAX_DEPTH = 3000

# Depth at which the "Table 1" headline refusal numbers are reported.
TABLE_DEPTH = 500

# A depth is only reported if at least this many instances were evaluated there.
MIN_COUNT = 10

# Candidate JSON keys for a stable per-instance identifier (paper logs use
# "instance"; the rest are historical fallbacks).
_INSTANCE_KEYS = ("instance_id", "id", "idx", "instance", "prompt_id")

# External guardrails are model-agnostic classifiers with no hidden-state access,
# so they are run over a single fixed base model's harmful continuations (the
# same generations for every panel). This matches the notebook.
GUARDRAIL_GENERATOR = "meta-llama/Llama-3.1-8B-Instruct"

# Guardrails and deep-alignment checkpoints were only evaluated on harmful data.
HARMFUL_SPLIT = "harmful"

# Closed-source Claude. It is not in the model registry (no probe / hidden-state
# access), and its on-disk slug uses underscores rather than the dash-preserving
# ``slugify_model`` output, so we pass the pre-slugged directory name directly.
CLAUDE_DIR = "claude_sonnet_4_20250514"
CLAUDE_LABEL = "Claude Sonnet 4"

# Method table for the main grid. ``kind`` selects the path builder.
#   mode          -> generation log with the given mode
#   deep_alignment-> mode_empty log of the model's deep-alignment baseline
#   guardrail     -> external-guardrail defense log (fixed generator)
#   probe         -> ADA-LP logistic probe log
METHOD_SPECS: List[Tuple[str, dict]] = [
    ("Base Model", dict(color="#4C78A8", marker="o", kind="mode", mode="empty")),
    ("Deep Alignment", dict(color="#17AEAF", marker="s", kind="deep_alignment")),
    ("Self Defense", dict(color="#54A24B", marker="^", kind="mode", mode="reflection")),
    ("Meta Llama-Guard-4-12B",
     dict(color="#2F80ED", marker="v", kind="guardrail", guardrail="meta-llama/Llama-Guard-4-12B")),
    ("IBM Granite-Guardian-3.3-8b",
     dict(color="#6F5ACD", marker="<", kind="guardrail", guardrail="ibm-granite/granite-guardian-3.3-8b")),
    ("ADA (RK)", dict(color="#F26B21", marker="*", kind="mode", mode="add_safetytoken")),
    ("ADA (LP)", dict(color="#D62F2F", marker="h", kind="probe")),
]


# --------------------------------------------------------------------------- #
# Deep-alignment baseline mapping (read from configs/models.yaml, not hardcoded)
# --------------------------------------------------------------------------- #

def _deep_alignment_map() -> Dict[str, str]:
    """Map base HF id -> deep-alignment baseline HF id from the registry YAML.

    ``ada.registry`` does not expose the ``deep_alignment_baselines`` block, so we
    read it from the same ``configs/models.yaml`` the registry loads.
    """
    cfg = Path(_registry.__file__).resolve().parent.parent / "configs" / "models.yaml"
    try:
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        return {}
    return {
        entry["base"]: entry["hf_id"]
        for entry in data.get("deep_alignment_baselines", [])
        if "base" in entry and "hf_id" in entry
    }


# --------------------------------------------------------------------------- #
# Curve loading / aggregation
# --------------------------------------------------------------------------- #

def _instance_id(record: dict, row_idx: int):
    """Robust per-instance identifier (falls back to the row index)."""
    for key in _INSTANCE_KEYS:
        val = record.get(key)
        if val is not None:
            return val
    return f"_row_{row_idx}"


def _refusal_curve(path: Path, depth_filter: Optional[Callable[[int], bool]] = None) -> Dict[int, float]:
    """Independent per-depth refusal rate ``{depth: rate}`` from one log.

    This reproduces the deep-prefill notebook's ``parse_log`` exactly: the rate at depth
    ``d`` is the fraction of instances whose checkpoint at *that* depth is a
    refusal (NOT the cumulative "refused by depth d"). deep-prefill relies on this
    definition — a base model's per-depth refusal *drops* as the harmful prefill
    deepens, which is the phenomenon the figure exposes; the cumulative curve in
    ``_common.parse_refusal_curve`` would hide it. Depths with fewer than
    ``MIN_COUNT`` instances are dropped; ``depth_filter`` optionally restricts
    which checkpoints are ingested (the grid keeps depth 25 and multiples of 100).
    """
    detailed = read_json(path).get("detailed_logs", [])
    counts: Dict[int, int] = defaultdict(int)
    refuse: Dict[int, int] = defaultdict(int)
    # Dedup per (instance, depth) so repeated rows don't double-count.
    seen: Dict[Tuple[object, int], bool] = {}
    for row_idx, rec in enumerate(detailed):
        depth = int(rec.get("depth", 0))
        if depth_filter is not None and not depth_filter(depth):
            continue
        seen[(_instance_id(rec, row_idx), depth)] = bool(rec.get("is_refusal", False))
    for (_inst, depth), flag in seen.items():
        counts[depth] += 1
        refuse[depth] += int(flag)
    return {d: refuse[d] / counts[d] for d in counts if counts[d] >= MIN_COUNT}


def _grid_depth(depth: int) -> bool:
    """Marker cadence for the 0..2500 grid panels: depth 25 and multiples of 100."""
    return depth == 25 or depth % 100 == 0


def _load(
    path: Path,
    force_depth0: bool = False,
    depth_filter: Optional[Callable[[int], bool]] = None,
) -> Dict[int, float]:
    """Parse a refusal curve, returning ``{}`` if the log is missing.

    ``force_depth0`` reproduces the notebook's Claude fix-up: at depth 0 (no
    prefill) the base model always refuses, but that checkpoint is absent from
    some logs, so pin rate(0) = 1.0.
    """
    if not path.exists():
        return {}
    curve = _refusal_curve(path, depth_filter)
    if force_depth0 and curve:
        curve[0] = 1.0
    return curve


def _method_log_path(
    log_root: Path,
    spec: dict,
    split: str,
    dataset: str,
    model: str,
    deep_align: Dict[str, str],
) -> Optional[Path]:
    """Resolve the log path for one (method, model, dataset), or None to skip."""
    kind = spec["kind"]
    if kind == "mode":
        # The deep-prefill notebook reads the plain mode dir for every model (reasoning
        # models have a separate mode_*_reasoning variant that the paper's deep-prefill
        # figure does not use), so we do not append the _reasoning suffix here.
        return log_root / find_generation_log(
            split, dataset, model, spec["mode"], DEPTH_STEP, MAX_DEPTH,
        )
    if kind == "deep_alignment":
        baseline = deep_align.get(model)
        if baseline is None:
            return None  # no deep-alignment checkpoint for this base model
        return log_root / find_generation_log(
            HARMFUL_SPLIT, dataset, baseline, "empty", DEPTH_STEP, MAX_DEPTH,
        )
    if kind == "guardrail":
        return log_root / find_defense_log(
            HARMFUL_SPLIT, dataset, spec["guardrail"], GUARDRAIL_GENERATOR,
            DEPTH_STEP, MAX_DEPTH,
        )
    if kind == "probe":
        return log_root / find_probe_log(
            split, dataset, model,
            probe_safety_tokens(model), probe_layer(model),
            DEPTH_STEP, MAX_DEPTH,
        )
    raise ValueError(f"unknown method kind: {kind!r}")


def _average_curves(curves: Sequence[Dict[int, float]]) -> Dict[int, float]:
    """Unweighted mean of refusal rates at each depth across datasets.

    A depth is included if it appears in at least one dataset; its value is the
    mean over exactly the datasets that reported it (matches the notebook).
    """
    curves = [c for c in curves if c]
    if not curves:
        return {}
    depths = set().union(*(c.keys() for c in curves))
    out: Dict[int, float] = {}
    for d in depths:
        vals = [c[d] for c in curves if d in c]
        if vals:
            out[d] = float(np.mean(vals))
    return out


def _model_methods_data(
    log_root: Path,
    model: str,
    datasets: Sequence[str],
    split: str,
    deep_align: Dict[str, str],
) -> Dict[str, Dict[int, float]]:
    """Return {method_label: averaged+downsampled curve} for one model."""
    methods: Dict[str, Dict[int, float]] = {}
    for label, spec in METHOD_SPECS:
        per_dataset = []
        skip = False
        for dataset in datasets:
            path = _method_log_path(log_root, spec, split, dataset, model, deep_align)
            if path is None:
                skip = True
                break
            per_dataset.append(_load(path, depth_filter=_grid_depth))
        if skip:
            continue
        averaged = _average_curves(per_dataset)
        if averaged:
            methods[label] = averaged
    return methods


# --------------------------------------------------------------------------- #
# Main grid figure
# --------------------------------------------------------------------------- #

def _plot_single_model(ax, model: str, methods_data: Dict[str, Dict[int, float]]) -> None:
    import matplotlib.pyplot as plt

    if not methods_data:
        ax.text(0.5, 0.5, f"No data available\nfor {short_model_name(model)}",
                ha="center", va="center", transform=ax.transAxes, fontsize=20)
    else:
        for label, spec in METHOD_SPECS:
            rates = methods_data.get(label)
            if not rates:
                continue
            xs = sorted(rates)
            ax.plot(xs, [rates[d] for d in xs], color=spec["color"], marker=spec["marker"],
                    linestyle="-", linewidth=2.5, markersize=12, alpha=0.9, label=label)
    ax.set_title(short_model_name(model), fontsize=30)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-20, 2520)
    ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
    ax.set_ylim(-0.01, 1.02)
    ax.tick_params(axis="both", which="major", labelsize=24, length=5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))


def _plot_grid(
    log_root: Path,
    models: Sequence[str],
    datasets: Sequence[str],
    split: str,
    deep_align: Dict[str, str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    ncols = 3
    nrows = (len(models) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 9, nrows * 8),
        gridspec_kw={"wspace": 0.05, "hspace": 0.18}, squeeze=False,
    )

    legend_handles: Dict[str, object] = {}
    for i in range(nrows * ncols):
        row, col = divmod(i, ncols)
        ax = axes[row][col]
        if i >= len(models):
            ax.set_visible(False)
            continue
        methods_data = _model_methods_data(log_root, models[i], datasets, split, deep_align)
        _plot_single_model(ax, models[i], methods_data)
        for h, l in zip(*ax.get_legend_handles_labels()):
            legend_handles.setdefault(l, h)
        if col > 0:
            ax.set_yticklabels([])

    fig.supxlabel("Prefill Depth (Token Position)", fontsize=32, weight="bold", y=0.05)
    fig.supylabel(r"Refusal Rate ($\uparrow$ is better)", fontsize=32, weight="bold", x=0.005)
    fig.subplots_adjust(top=0.95, left=0.06, right=0.99, bottom=0.13)

    # Single shared legend below the shared x-label, ordered like METHOD_SPECS.
    ordered = [(legend_handles[l], l) for l, _ in METHOD_SPECS if l in legend_handles]
    if ordered:
        handles, labels = zip(*ordered)
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4),
                   fontsize=24, frameon=True, bbox_to_anchor=(0.5, 0.045))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


# --------------------------------------------------------------------------- #
# gpt-oss: all-models base refusal curve (0..500)
# --------------------------------------------------------------------------- #

def _plot_prefill_all_model(
    log_root: Path,
    models: Sequence[str],
    dataset: str,
    split: str,
    output_path: Path,
) -> None:
    """Base-model (mode_empty) refusal vs. depth for the whole zoo + Claude."""
    import matplotlib.pyplot as plt

    # Open models (registry) + closed-source Claude, in that order.
    entries: List[Tuple[str, Path, bool]] = []
    for model in models:
        entries.append((short_model_name(model),
                        log_root / find_generation_log(split, dataset, model, "empty",
                                                        DEPTH_STEP, MAX_DEPTH), False))
    entries.append((CLAUDE_LABEL,
                    log_root / find_generation_log(split, dataset, CLAUDE_DIR, "empty",
                                                   DEPTH_STEP, MAX_DEPTH), True))

    palette = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(10, 8))
    n = 0
    for label, path, force0 in entries:
        rates = _load(path, force_depth0=force0)
        if not rates:
            print(f"[SKIP] {label}: {path}")
            continue
        xs = sorted(rates)
        ax.plot(xs, [rates[d] for d in xs], marker="o", linewidth=2.5, markersize=6,
                alpha=0.9, label=label, color=palette[n % len(palette)])
        n += 1

    ax.set_xlabel("Prefill Depth (Token Position)", fontsize=26)
    ax.set_ylabel("Refusal Rate", fontsize=26)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.set_xlim(-3, 503)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.tick_params(axis="both", labelsize=26)
    ax.legend(fontsize=18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


# --------------------------------------------------------------------------- #
# Base vs. ADA per model (0..500)
# --------------------------------------------------------------------------- #

def _plot_base_vs_dia(
    log_root: Path,
    models: Sequence[str],
    dataset: str,
    split: str,
    output_path: Path,
) -> None:
    """Base (dashed) vs. ADA (solid) per model at 0..500.

    ADA = ADA-LP (linear probe) for open registry models; ADA-RK
    (mode_add_safetytoken) for closed-source Claude.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from ._common import MARKERS, husl_palette

    labels = [short_model_name(m) for m in models] + [CLAUDE_LABEL]
    color_map = {lab: c for lab, c in zip(labels, husl_palette(len(labels)))}
    marker_map = {lab: MARKERS[i % len(MARKERS)] for i, lab in enumerate(labels)}

    base_data: Dict[str, Tuple[list, list]] = {}
    dia_data: Dict[str, Tuple[list, list]] = {}

    def add(store, label, path, force0=False):
        rates = _load(path, force_depth0=force0)
        if not rates:
            print(f"[SKIP] {label}: {path}")
            return
        xs = sorted(rates)
        store[label] = (xs, [rates[d] for d in xs])

    for model in models:
        label = short_model_name(model)
        is_claude = False
        add(base_data, label,
            log_root / find_generation_log(split, dataset, model, "empty", DEPTH_STEP, MAX_DEPTH))
        add(dia_data, label,
            log_root / find_probe_log(split, dataset, model, probe_safety_tokens(model),
                                      probe_layer(model), DEPTH_STEP, MAX_DEPTH))
    # Claude: base + ADA-RK.
    add(base_data, CLAUDE_LABEL,
        log_root / find_generation_log(split, dataset, CLAUDE_DIR, "empty", DEPTH_STEP, MAX_DEPTH),
        force0=True)
    add(dia_data, CLAUDE_LABEL,
        log_root / find_generation_log(split, dataset, CLAUDE_DIR, "add_safetytoken",
                                       DEPTH_STEP, MAX_DEPTH))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Tiny vertical jitter so multiple curves pinned at 100% stay distinguishable.
    ada_flat = [l for l in labels if l in dia_data and all(v >= 0.99 for v in dia_data[l][1])]
    offsets = dict(zip(ada_flat, np.linspace(-0.005, 0.0, len(ada_flat)))) if ada_flat else {}

    for label in labels:  # Base (dashed, no marker)
        if label in base_data:
            xs, ys = base_data[label]
            ax.plot(xs, ys, "--", linewidth=3.5, color=color_map[label], alpha=0.9)
    for label in labels:  # ADA (solid, per-model marker)
        if label in dia_data:
            xs, ys = dia_data[label]
            off = offsets.get(label, 0.0)
            ax.plot(xs, [y + off for y in ys], "-", marker=marker_map[label],
                    linewidth=3.5, markersize=10, color=color_map[label], alpha=0.8)

    ax.set_xlabel("Prefill Depth (Token Position)", fontsize=26, fontweight="bold")
    ax.set_ylabel(r"Refusal Rate ($\uparrow$)", fontsize=26, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.set_xlim(-3, 503)
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.tick_params(axis="both", labelsize=26)

    handles = [Line2D([0], [0], marker=marker_map[l], linestyle="None",
                      markerfacecolor=color_map[l], markeredgecolor="black",
                      markersize=10, label=l)
               for l in labels if l in base_data or l in dia_data]
    handles.append(Line2D([0], [0], color="black", lw=3, linestyle="--", label="Base Model"))
    handles.append(Line2D([0], [0], color="black", lw=3, linestyle="-", label="ADA (Ours)"))
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 0.20),
              fontsize=18, frameon=True, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


# --------------------------------------------------------------------------- #
# Claude: Base vs. ADA-RK over the full depth range
# --------------------------------------------------------------------------- #

def _plot_claude(log_root: Path, dataset: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    curves = [
        ("Base Model", "P", "#4C78A8",
         log_root / find_generation_log(HARMFUL_SPLIT, dataset, CLAUDE_DIR, "empty",
                                        DEPTH_STEP, MAX_DEPTH), True),
        ("ADA (RK)", "v", "#F26B21",
         log_root / find_generation_log(HARMFUL_SPLIT, dataset, CLAUDE_DIR, "add_safetytoken",
                                        DEPTH_STEP, MAX_DEPTH), False),
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    handles = []
    for label, marker, color, path, force0 in curves:
        rates = _load(path, force_depth0=force0)
        if not rates:
            print(f"[SKIP] Claude {label}: {path}")
            continue
        xs = sorted(rates)
        ax.plot(xs, [rates[d] for d in xs], "-", marker=marker, linewidth=3.5,
                markersize=10, color=color, alpha=0.85)
        handles.append(Line2D([0], [0], marker=marker, linestyle="None",
                              markerfacecolor=color, markeredgecolor="black",
                              markersize=10, label=label))

    if not handles:
        plt.close(fig)
        print(f"[SKIP] Claude figure for {dataset}: no data")
        return

    ax.set_xlabel("Prefill Depth (Token Position)", fontsize=26, fontweight="bold")
    ax.set_ylabel(r"Refusal Rate ($\uparrow$)", fontsize=26, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.set_xlim(-5, 2505)
    ax.set_ylim(-0.01, 1.01)
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.tick_params(axis="both", labelsize=26)
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 0.20),
              fontsize=20, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {output_path}")


# --------------------------------------------------------------------------- #
# Depth-500 "Table 1" numbers
# --------------------------------------------------------------------------- #

def _print_table(
    log_root: Path,
    models: Sequence[str],
    datasets: Sequence[str],
    split: str,
    deep_align: Dict[str, str],
) -> None:
    """Print refusal rate at depth 500, per (model, dataset, method) + averages.

    Values are the independent refusal rate at the depth-500 checkpoint (matching
    the notebook's ``calculate_refusal_rate_at_depth``): the fraction of instances
    that refuse when the harmful prefill is cut at exactly 500 tokens.
    """
    # results[model][dataset][method] = rate
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in models:
        results[model] = {}
        for dataset in datasets:
            results[model][dataset] = {}
            for label, spec in METHOD_SPECS:
                path = _method_log_path(log_root, spec, split, dataset, model, deep_align)
                if path is None:
                    continue
                curve = _load(path)
                if TABLE_DEPTH in curve:
                    results[model][dataset][label] = curve[TABLE_DEPTH]

    header = f"{'METHOD':<28}" + "".join(f"{d.upper():>16}" for d in datasets) + f"{'AVG':>12}"

    def _row(label: str, get) -> str:
        row = f"{label:<28}"
        vals = []
        for dataset in datasets:
            v = get(dataset)
            if v is None:
                row += f"{'--':>16}"
            else:
                row += f"{v * 100:>15.1f}%"
                vals.append(v)
        row += (f"{np.mean(vals) * 100:>11.1f}%" if vals else f"{'--':>12}")
        return row

    print(f"\n{'=' * 96}\nRefusal rate at prefill depth {TABLE_DEPTH} (Table 1)\n{'=' * 96}")
    for model in models:
        print(f"\n--- {short_model_name(model)} ---")
        print(header)
        print("-" * len(header))
        for label, _ in METHOD_SPECS:
            print(_row(label, lambda ds, m=model, lab=label: results[m][ds].get(lab)))

    # Overall: average across datasets, then across models.
    print(f"\n--- OVERALL (averaged across models) ---")
    print(header)
    print("-" * len(header))
    for label, _ in METHOD_SPECS:
        def per_dataset_avg(ds, lab=label):
            vals = [results[m][ds][lab] for m in models if lab in results[m][ds]]
            return float(np.mean(vals)) if vals else None
        print(_row(label, per_dataset_avg))
    print("=" * 96)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=list_models(),
                        help="HF ids to plot (default: all registry models).")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                        help="Harmful datasets to average over.")
    parser.add_argument("--split", default=HARMFUL_SPLIT, choices=["harmful", "benign"],
                        help="Data split for base/self-defense/ADA logs (default: harmful).")
    parser.add_argument("--log-root", default=".", type=Path,
                        help="Root directory containing the *_logs/ trees (default: CWD).")
    parser.add_argument("--output-dir", default="figures", type=Path,
                        help="Directory for output PDFs (default: figures/).")
    parser.add_argument("--no-table", action="store_true",
                        help="Skip printing the depth-500 Table 1 numbers.")
    args = parser.parse_args(argv)

    log_root = args.log_root
    out = ensure_output_dir(args.output_dir)
    deep_align = _deep_alignment_map()

    # 1. Main grid averaged over all datasets.
    _plot_grid(log_root, args.models, args.datasets, args.split, deep_align,
               out / "all_models_refusal_rates.pdf")

    # 2. Per-dataset grids.
    for dataset in args.datasets:
        _plot_grid(log_root, args.models, [dataset], args.split, deep_align,
                   out / f"all_models_refusal_rates_{dataset}.pdf")

    # 3. gpt-oss all-models base curve (single dataset; paper uses advbench).
    zoom_dataset = "advbench" if "advbench" in args.datasets else args.datasets[0]
    _plot_prefill_all_model(log_root, args.models, zoom_dataset, args.split,
                            out / "prefill_all_model.pdf")

    # 4. Per-model Base vs. ADA (0..500).
    _plot_base_vs_dia(log_root, args.models, zoom_dataset, args.split,
                      out / "prefill_base_vs_dia.pdf")

    # 5. Claude Base vs. ADA-RK over the full depth range, per dataset.
    for dataset in args.datasets:
        _plot_claude(log_root, dataset, out / f"claude_{dataset}.pdf")

    # 6. Depth-500 table.
    if not args.no_table:
        _print_table(log_root, args.models, args.datasets, args.split, deep_align)


if __name__ == "__main__":
    main()
