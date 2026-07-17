"""E5 - Over-refusal: benign refusal rate vs. generation depth.

Reconstructs the over-refusal figures from ``final-prefix-benign.ipynb``:

* ``figures/benign_avg_refusal_rates.pdf`` / ``..._full.pdf`` - refusal rate on
  ordinary (harmless) tasks, averaged over seven capability benchmarks
  (GSM8K, MATH, BBH, HumanEval, MMLU, SimpleQA, GPQA).
* ``figures/xstest_refusal_rates.pdf`` / ``..._full.pdf`` - refusal rate on
  XSTest (prompts that *look* unsafe but are benign).

Lower is better: a good defense should not start refusing harmless requests as
generation gets deeper. Each subplot is one model; the ``_full`` variant shows
every registered model, the compact variant shows three representatives.

Curves compared (per subplot):

* **Base Model** / **Deep Alignment** - flat baselines computed from the plain
  benign responses (a refusal-string match), drawn as horizontal lines.
* **Self Defense** - the reflection baseline (``mode_reflection``).
* two highlighted **guardrails** (``configs/guardrails.yaml`` ``highlighted:
  true`` -> Llama-Guard-4-12B and Granite-Guardian-3.3-8B) scoring a fixed
  reference model's benign responses.
* **ADA (RK)** (``mode_add_safetytoken``) and **ADA (LP)** (the linear probe).

Per-file refusal curves and all on-disk paths come from
:mod:`ada.plotting._common`; this script owns the benign-baseline string match,
the cross-dataset averaging, and the figure layout. Probe layers / Safety-Token
spans / slugs come from :mod:`ada.registry`.

Run as::

    python -m ada.plotting.plot_e5_benign --output-dir figures
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

from .. import registry
from ..registry import list_models, slugify_model
from ._common import (
    ensure_output_dir,
    cumulative_refusal_curve,
    find_defense_log,
    find_generation_log,
    find_probe_log,
    parse_refusal_curve,
    probe_layer,
    probe_safety_tokens,
)

# --------------------------------------------------------------------------- #
# Experiment constants (from final-prefix-benign.ipynb)
# --------------------------------------------------------------------------- #
DEFAULT_BENIGN_DATASETS = ["gsm8k", "math", "bbh", "humaneval", "mmlu", "simpleqa", "gpqa"]
XSTEST_DATASETS = ["xstest"]

# Three representative models for the compact figure (one Gemma, one Llama, one
# reasoning model), matching the notebook's compact panel.
DEFAULT_COMPACT_MODELS = [
    "google/gemma-2-9b-it",
    "meta-llama/Llama-2-7b-chat-hf",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]

# In the notebook, the guardrail curves always score the same reference model's
# benign responses (they are model-agnostic here), so the guardrail lines are
# identical across subplots.
GUARDRAIL_TARGET_HF = "meta-llama/Llama-3.1-8B-Instruct"

# Cosmetic display for the two highlighted guardrails (labels/colours/markers as
# in the paper figure). The set of guardrails itself is read from guardrails.yaml.
GUARDRAIL_DISPLAY = {
    "meta-llama/Llama-Guard-4-12B": ("Meta Llama-Guard-4-12B", "#2F80ED", "v"),
    "ibm-granite/granite-guardian-3.3-8b": ("IBM Granite-Guardian-3.3-8b", "#6F5ACD", "<"),
}

# Refusal-string patterns used for the flat Base / Deep-Alignment baselines.
REFUSAL_PATTERNS = ["i can't", "i can’t", "i cannot"]

XLIM = 600

BASE_METHOD_STYLE = {
    "Base Model": {"color": "#4C78A8", "marker": "o", "linestyle": "--"},
    "Deep Alignment": {"color": "#17AEAF", "marker": "s", "linestyle": "--"},
    "Self Defense": {"color": "#54A24B", "marker": "^", "linestyle": "-"},
    "ADA (RK)": {"color": "#F26B21", "marker": "*", "linestyle": "-"},
    "ADA (LP)": {"color": "#D62F2F", "marker": "h", "linestyle": "-"},
}


def _config_dir() -> Path:
    return Path(registry.__file__).resolve().parent.parent / "configs"


def _deep_alignment_hf_id(base_hf_id: str) -> Optional[str]:
    """HF id of the deep-alignment checkpoint for ``base_hf_id`` (or None)."""
    with open(_config_dir() / "models.yaml", "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for entry in raw.get("deep_alignment_baselines", []):
        if entry.get("base") == base_hf_id:
            return entry["hf_id"]
    return None


def load_highlighted_guardrails() -> List[str]:
    """HF ids of the guardrails flagged ``highlighted: true`` in guardrails.yaml."""
    try:
        with open(_config_dir() / "guardrails.yaml", "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        return list(GUARDRAIL_DISPLAY)
    highlighted = [g["hf_id"] for g in raw.get("guardrails", []) if g.get("highlighted")]
    return highlighted or list(GUARDRAIL_DISPLAY)


def build_method_specs() -> List[dict]:
    """Ordered list of method specs (name/kind/style) plotted in each subplot."""
    specs: List[dict] = [
        {"name": "Base Model", "kind": "baseline", **BASE_METHOD_STYLE["Base Model"]},
        {"name": "Deep Alignment", "kind": "baseline_deep", **BASE_METHOD_STYLE["Deep Alignment"]},
        {"name": "Self Defense", "kind": "reflection", **BASE_METHOD_STYLE["Self Defense"]},
    ]
    for gid in load_highlighted_guardrails():
        label, color, marker = GUARDRAIL_DISPLAY.get(gid, (gid.split("/")[-1], "#888888", "D"))
        specs.append({"name": label, "kind": "guardrail", "hf_id": gid,
                      "color": color, "marker": marker, "linestyle": "-"})
    specs.append({"name": "ADA (RK)", "kind": "rk", **BASE_METHOD_STYLE["ADA (RK)"]})
    specs.append({"name": "ADA (LP)", "kind": "lp", **BASE_METHOD_STYLE["ADA (LP)"]})
    return specs


def calculate_baseline_refusal_rate(jsonl_path: Path) -> Optional[float]:
    """Fraction of benign responses that contain a refusal phrase (or None)."""
    if not jsonl_path.exists():
        return None
    total = 0
    refused = 0
    try:
        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                response = data["messages"][-1]["content"].lower()
                total += 1
                if any(p in response for p in REFUSAL_PATTERNS):
                    refused += 1
    except (json.JSONDecodeError, KeyError):
        return None
    return refused / total if total else 0.0


def average_refusal_rates(rate_dicts: List[Dict[int, float]]) -> Dict[int, float]:
    """Unweighted per-depth average of several {depth: rate} dicts."""
    all_depths = set()
    for rates in rate_dicts:
        all_depths.update(rates)
    averaged: Dict[int, float] = {}
    for depth in all_depths:
        vals = [r[depth] for r in rate_dicts if depth in r]
        if vals:
            averaged[depth] = float(np.mean(vals))
    return averaged


def get_method_data(
    hf_id: str,
    method_specs: List[dict],
    datasets: List[str],
    data_root: Path,
) -> Dict[str, dict]:
    """Return {method_name: {'baseline': v} | {depth: rate}} for one model.

    Baseline methods (Base / Deep Alignment) yield a single flat value; every
    other method yields a depth-indexed refusal curve averaged over ``datasets``.
    NOTE: ADA-RK / Self-Defense use the standard ``mode_*`` logs (no
    ``_reasoning`` suffix), matching the notebook that produced these figures.
    """
    gen_base = data_root / "vllm_generation_logs"
    defense_base = data_root / "vllm_defense_logs"
    probe_base = data_root / "logs"
    deep_hf_id = _deep_alignment_hf_id(hf_id)

    results: Dict[str, dict] = {}
    for spec in method_specs:
        name, kind = spec["name"], spec["kind"]

        if kind in ("baseline", "baseline_deep"):
            slug = slugify_model(hf_id) if kind == "baseline" else (
                slugify_model(deep_hf_id) if deep_hf_id else None)
            if slug is None:  # no deep-alignment checkpoint for this model
                continue
            rates = []
            for dataset in datasets:
                jsonl = _baseline_benign_response_path(data_root, dataset, slug)
                rate = calculate_baseline_refusal_rate(jsonl)
                if rate is not None:
                    rates.append(rate)
            if rates:
                results[name] = {"baseline": float(np.mean(rates))}
            continue

        per_dataset: List[Dict[int, float]] = []
        for dataset in datasets:
            if kind == "reflection":
                path = find_generation_log("benign", dataset, hf_id, "reflection", base_dir=gen_base)
            elif kind == "rk":
                path = find_generation_log("benign", dataset, hf_id, "add_safetytoken", base_dir=gen_base)
            elif kind == "guardrail":
                path = find_defense_log("benign", dataset, spec["hf_id"],
                                        GUARDRAIL_TARGET_HF, base_dir=defense_base)
            elif kind == "lp":
                path = find_probe_log("benign", dataset, hf_id, probe_safety_tokens(hf_id),
                                      probe_layer(hf_id), base_dir=probe_base)
            else:
                continue
            if not path.exists():
                continue
            curve = parse_refusal_curve(path)
            if curve:
                per_dataset.append(curve)
        if per_dataset:
            results[name] = average_refusal_rates(per_dataset)
    return results


def _baseline_benign_response_path(data_root: Path, dataset: str, slug: str) -> Path:
    """Baseline benign transcripts, preferring the release ``data/eval/over_refusal``.

    ``prepare_datasets.sh`` copies these to ``data/eval/over_refusal/{ds}/{slug}/``
    (the same corpus every other reader resolves — see
    ``ada.probe.evaluate.find_response_file``); the original ``benign_responses/``
    tree is kept as a source fallback. Returns the first existing candidate, else
    the release path (so ``calculate_baseline_refusal_rate`` reports no data).
    """
    candidates = [
        data_root / "data" / "eval" / "over_refusal" / dataset / slug / "responses.jsonl",
        data_root / "benign_responses" / dataset / slug / "responses.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _benign_log_path(spec: dict, hf_id: str, deep_hf_id: Optional[str], dataset: str, data_root: Path):
    """Resolve the benign log/response path for one (method, dataset, model)."""
    kind = spec["kind"]
    if kind == "baseline":
        return _baseline_benign_response_path(data_root, dataset, slugify_model(hf_id))
    if kind == "baseline_deep":
        if deep_hf_id is None:
            return None
        return _baseline_benign_response_path(data_root, dataset, slugify_model(deep_hf_id))
    if kind == "reflection":
        return find_generation_log("benign", dataset, hf_id, "reflection",
                                   base_dir=data_root / "vllm_generation_logs")
    if kind == "rk":
        return find_generation_log("benign", dataset, hf_id, "add_safetytoken",
                                   base_dir=data_root / "vllm_generation_logs")
    if kind == "guardrail":
        return find_defense_log("benign", dataset, spec["hf_id"], GUARDRAIL_TARGET_HF,
                                base_dir=data_root / "vllm_defense_logs")
    if kind == "lp":
        return find_probe_log("benign", dataset, hf_id, probe_safety_tokens(hf_id),
                              probe_layer(hf_id), base_dir=data_root / "logs")
    return None


def print_benign_table(models: List[str], method_specs: List[dict], datasets: List[str],
                       data_root: Path) -> None:
    """Instance-level over-refusal (%, lower is better) per method x benign dataset.

    Reproduces the benign block of the main results table: an instance counts as
    an over-refusal if any checkpoint flags it (cumulative refusal at the deepest
    checkpoint); baseline rows use the raw response refusal rate.
    """
    for model in models:
        deep = _deep_alignment_hf_id(model)
        print(f"\n=== Over-refusal (%, lower better) — {model} ===")
        print(f"{'Method':<26}" + "".join(f"{d[:8]:>9}" for d in datasets))
        for spec in method_specs:
            cells = []
            for ds in datasets:
                path = _benign_log_path(spec, model, deep, ds, data_root)
                val = None
                if path is not None and path.exists():
                    if spec["kind"] in ("baseline", "baseline_deep"):
                        val = calculate_baseline_refusal_rate(path)
                    else:
                        curve = cumulative_refusal_curve(path)
                        val = curve[max(curve)] if curve else None
                cells.append(f"{100 * val:>8.1f}%" if val is not None else f"{'--':>9}")
            print(f"{spec['name']:<26}" + "".join(cells))


def _local_ymax(methods_data: Dict[str, dict]) -> float:
    ymax = 0.0
    for rates in methods_data.values():
        if not rates:
            continue
        if set(rates) == {"baseline"}:
            ymax = max(ymax, float(rates["baseline"]))
        else:
            ymax = max([ymax] + [float(v) for v in rates.values() if v is not None])
    return ymax


def plot_single_model(
    ax,
    hf_id: str,
    method_specs: List[dict],
    datasets: List[str],
    data_root: Path,
    show_ylabel: bool,
) -> None:
    methods_data = get_method_data(hf_id, method_specs, datasets, data_root)
    style = {s["name"]: s for s in method_specs}

    if not methods_data:
        ax.text(0.5, 0.5, f"No data available\nfor {hf_id.split('/')[-1]}",
                ha="center", va="center", transform=ax.transAxes, fontsize=20)
        ax.set_title(hf_id.split("/")[-1], fontsize=28)
        ax.set_xlim(0, XLIM)
        ax.set_xticks(list(range(0, XLIM + 1, 200)))
        return

    for name, rates in methods_data.items():
        conf = style[name]
        if set(rates) == {"baseline"}:
            ax.axhline(y=rates["baseline"], color=conf["color"], linestyle=conf["linestyle"],
                       linewidth=2.5, alpha=0.9, label=name)
        else:
            xs = sorted(rates)
            ys = [rates[d] for d in xs]
            ax.plot(xs, ys, color=conf["color"], marker=conf["marker"],
                    linestyle=conf["linestyle"], linewidth=2.5, markersize=8,
                    alpha=0.9, label=name)

    ax.set_title(hf_id.split("/")[-1], fontsize=30)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, XLIM)
    ax.set_xticks(list(range(0, XLIM + 1, 100)))
    top = min(1.0, _local_ymax(methods_data) * 1.10)
    ax.set_ylim(-0.01, max(0.10, top))
    ax.tick_params(axis="both", which="major", labelsize=24, length=5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    if show_ylabel:
        ax.set_ylabel("Refusal Rate", fontsize=22, labelpad=6)


def make_figure(models: List[str], datasets: List[str], save_path: Path, data_root: Path) -> None:
    method_specs = build_method_specs()

    ncols = min(3, len(models))
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(9 * ncols, 8 * nrows),
        gridspec_kw={"wspace": 0.12, "hspace": 0.25}, squeeze=False,
    )
    flat = axes.flatten()

    legend_handles: Dict[str, object] = {}
    for idx, ax in enumerate(flat):
        if idx >= len(models):
            ax.axis("off")
            continue
        plot_single_model(ax, models[idx], method_specs, datasets, data_root,
                          show_ylabel=(idx % ncols == 0))
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            legend_handles.setdefault(label, handle)

    if legend_handles:  # one shared legend in the top-right subplot
        flat[ncols - 1].legend(
            legend_handles.values(), legend_handles.keys(),
            fontsize=20, loc="lower right", markerscale=1.5,
            bbox_to_anchor=(1, 0.3), frameon=True,
        )

    fig.supxlabel("Generation Depth (Token Position)", fontsize=32, weight="bold")
    fig.supylabel(r"Refusal Rate ($\downarrow$ is better)", fontsize=32, weight="bold")
    plt.subplots_adjust(top=0.92, left=0.08, right=0.99, bottom=0.12)
    ensure_output_dir(save_path.parent)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] wrote {save_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Models for the '_full' figures (default: all registry models).",
    )
    parser.add_argument(
        "--compact-models", nargs="+", default=DEFAULT_COMPACT_MODELS,
        help="Models for the compact figures (default: 3 representatives).",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=DEFAULT_BENIGN_DATASETS,
        help="Benign capability datasets to average for the 'benign_avg' figures.",
    )
    parser.add_argument("--data-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    full_models = args.models if args.models else list_models()
    out = args.output_dir

    # Compact panels (3 representative models).
    make_figure(args.compact_models, args.datasets,
                out / "benign_avg_refusal_rates.pdf", args.data_root)
    make_figure(args.compact_models, XSTEST_DATASETS,
                out / "xstest_refusal_rates.pdf", args.data_root)
    # Full panels (every model).
    make_figure(full_models, args.datasets,
                out / "benign_avg_refusal_rates_full.pdf", args.data_root)
    make_figure(full_models, XSTEST_DATASETS,
                out / "xstest_refusal_rates_full.pdf", args.data_root)

    # Per-dataset over-refusal table (the benign block of the main results table).
    print_benign_table(full_models, build_method_specs(),
                       args.datasets + XSTEST_DATASETS, args.data_root)


if __name__ == "__main__":
    main()
