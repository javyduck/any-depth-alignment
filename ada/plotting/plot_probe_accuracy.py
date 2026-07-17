"""ADA-LP Safety-Token probe accuracy figures.

Reproduces the three probe-accuracy panels from ``final_plot_training.ipynb``,
each plotting per-layer logistic-regression accuracy read from the training
checkpoints::

    ckpts/{model_slug}/{safety_slug}/mask_token_none/hook_input_layernorm/
        gradual_cache/seed_{seed}/logistic/layer_{L}.json

The three figures (written to ``--output-dir``):

* ``{split}_all_model.pdf`` — every registered model, solid line = the model's
  injected Safety Token (``registry.probe_safety_tokens``), dashed line = the
  last generated token (the ``empty`` safety-token condition).
* ``{split}_choice_of_safety_token.pdf`` — token-choice ablation on one model
  (Llama-3.1 by default), sweeping header sub-spans as the probed Safety Token.
* ``{split}_hook_position.pdf`` — hook-position ablation on one model
  (gemma-2-9b by default) across the six intra-block hook positions.

``--split {train,val}`` selects which accuracy field to read (replaces the
notebook's ``SPLIT`` global). Per-model Safety Tokens and probe layers come from
``ada.registry``; only the two ablations' hand-picked token/hook lists are
declared here (they define the ablation, not a per-model table).

Run: ``python -m ada.plotting.plot_probe_accuracy --split train``
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

from ada.plotting._common import (
    MARKERS,
    MODELS,
    ensure_output_dir,
    husl_palette,
    probe_ckpt_dir,
    probe_safety_tokens,
    set_plot_style,
    short_model_name,
)
from ada.utils.io import read_json

# --------------------------------------------------------------------------- #
# Ablation configurations (figure-defining, not per-model tables)
# --------------------------------------------------------------------------- #

# Token-choice ablation: header sub-spans probed as the Safety Token. This is the
# Llama-3.1 header decomposition used in the paper; change --choice-model to run
# the same figure for another model with a matching hand-picked span list.
CHOICE_MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"
CHOICE_SAFETY_TOKENS = [
    "empty",
    "\n",
    "\n\n",
    "<|start_header_id|>",
    "assistant",
    "<|eot_id|>",
    "<|eot_id|><|start_header_id|>",
    "<|eot_id|><|start_header_id|>assistant",
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
]

# Hook-position ablation: the six intra-block read positions, on one model.
HOOK_MODEL_DEFAULT = "google/gemma-2-9b-it"
HOOK_SAFETY_TOKENS = "<end_of_turn>\n<start_of_turn>model"
HOOK_POSITIONS = [
    "mlp",
    "self_attn",
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
]
_HOOK_NICE = {
    "mlp": "MLP",
    "self_attn": "Self-Attention",
    "input_layernorm": "Input LayerNorm",
    "post_attention_layernorm": "Post-Attention LayerNorm",
    "pre_feedforward_layernorm": "Pre-FeedForward LayerNorm",
    "post_feedforward_layernorm": "Post-FeedForward LayerNorm",
}


# --------------------------------------------------------------------------- #
# Accuracy loading
# --------------------------------------------------------------------------- #

def _load_accuracy(data: dict, split: str) -> Optional[float]:
    """Extract ``{split}`` accuracy from a ``layer_{L}.json`` metrics record.

    Supports the single-value keys written by ``ada.probe.train``
    (``train_accuracy`` / ``val_accuracy``) and, defensively, a per-epoch list
    (``*_accuracies``) or a TPR/TNR fallback.
    """
    list_key = f"{split}_accuracies"
    scalar_key = f"{split}_accuracy"
    if isinstance(data.get(list_key), list) and data[list_key]:
        return float(max(data[list_key]))
    if scalar_key in data:
        return float(data[scalar_key])
    tpr, tnr = f"{split}_tpr", f"{split}_tnr"
    if tpr in data and tnr in data:
        return float((data[tpr] + data[tnr]) / 2)
    return None


def collect_layer_acc(
    model: str,
    safety_tokens: str,
    split: str,
    *,
    hook_position: str = "input_layernorm",
    mask_tokens: str = "none",
    seed: int = 42,
    gradual_cache: bool = True,
    ckpt_dir: str = "ckpts",
) -> Dict[int, float]:
    """Return ``{layer: accuracy}`` for one (model, safety token, hook)."""
    ckpt = probe_ckpt_dir(
        model, safety_tokens, mask_tokens=mask_tokens, hook_position=hook_position,
        gradual_cache=gradual_cache, seed=seed, ckpt_dir=ckpt_dir,
    )
    acc: Dict[int, float] = {}
    if not ckpt.exists():
        return acc
    for p in ckpt.glob("layer_*.json"):
        if p.name.endswith("_depth_0.json"):
            continue
        m = re.search(r"layer_(\d+)", p.name)
        if not m:
            continue
        value = _load_accuracy(read_json(p), split)
        if value is not None:
            acc[int(m.group(1))] = value
    return acc


def _sorted_points(acc_by_layer: Dict[int, float], max_layer: int):
    xs, ys = [], []
    for layer_id, value in sorted(acc_by_layer.items()):
        if 1 <= layer_id <= max_layer:
            xs.append(layer_id)
            ys.append(value)
    return xs, ys


def _style_layer_axis(ax, split: str, max_layer: int, ymin: float):
    import numpy as np

    ax.set_xlabel("Layer", fontsize=25, fontweight="bold")
    ax.set_ylabel(f"{split.capitalize()} Accuracy", fontsize=25, fontweight="bold")
    ax.set_ylim(ymin, 1.001)
    even_ticks = np.arange(2, max_layer + 1, 2)
    ax.set_xticks(even_ticks)
    ax.set_xticklabels(even_ticks)
    ax.set_xlim(0.5, max_layer + 0.5)
    ax.tick_params(axis="both", labelsize=25)
    ax.grid(True, color="gray", alpha=0.2, linestyle="--")


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #

def plot_all_models(models, split, out_dir, *, max_layer=42, **kw):
    """Per-layer accuracy for all models: Safety Token (solid) vs empty (dashed)."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    colors = husl_palette(len(models))

    for i, model in enumerate(models):
        color, marker = colors[i], MARKERS[i % len(MARKERS)]
        # Solid: the model's injected Safety Token (from the registry).
        acc = collect_layer_acc(model, probe_safety_tokens(model), split, **kw)
        xs, ys = _sorted_points(acc, max_layer)
        if xs:
            ax.scatter(xs, ys, color=color, marker=marker, s=100, alpha=0.9,
                       label=short_model_name(model))
            ax.plot(xs, ys, color=color, alpha=0.5, linewidth=2, linestyle="-")
        # Dashed: the last generated token (empty safety-token condition).
        acc_empty = collect_layer_acc(model, "empty", split, **kw)
        xs_e, ys_e = _sorted_points(acc_empty, max_layer)
        if xs_e:
            ax.scatter(xs_e, ys_e, color=color, marker=marker, s=100, alpha=0.7)
            ax.plot(xs_e, ys_e, color=color, alpha=0.5, linewidth=2, linestyle="--")

    _style_layer_axis(ax, split, max_layer, ymin=0.88)
    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([0], [0], color="black", linewidth=2, linestyle="-", label="Safety Tokens"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Generated Token"),
    ]
    labels += ["Safety Tokens", "Generated Token"]
    ax.legend(handles, labels, loc="lower right", fontsize=19, markerscale=1.2,
              ncol=2, columnspacing=1.0, handletextpad=0.5)

    fig.tight_layout()
    path = ensure_output_dir(out_dir) / f"{split}_all_model.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def _fmt_token(token: str) -> str:
    if token in ("", "empty"):
        return "Generated Token"
    return token.replace("\n", "\\n")


def plot_token_choice(model, safety_tokens_list, split, out_dir, *, max_layer=31, **kw):
    """Token-choice ablation: accuracy per layer for each candidate Safety Token."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    colors = husl_palette(len(safety_tokens_list))

    for i, tokens in enumerate(safety_tokens_list):
        acc = collect_layer_acc(model, tokens, split, **kw)
        xs, ys = _sorted_points(acc, max_layer)
        if not xs:
            continue
        ax.scatter(xs, ys, color=colors[i], marker=MARKERS[i % len(MARKERS)],
                   s=100, alpha=0.9, label=_fmt_token(tokens))
        ax.plot(xs, ys, color=colors[i], alpha=0.5, linewidth=2)

    _style_layer_axis(ax, split, max_layer, ymin=0.92)
    ax.legend(loc="lower right", fontsize=15, markerscale=1.3, ncol=1,
              handletextpad=0.4, labelspacing=0.3, borderpad=0.3)

    fig.tight_layout()
    path = ensure_output_dir(out_dir) / f"{split}_choice_of_safety_token.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_hook_positions(model, safety_tokens, hook_positions, split, out_dir,
                        *, max_layer=41, **kw):
    """Hook-position ablation: accuracy per layer for each intra-block read point."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    colors = husl_palette(len(hook_positions))

    for i, hook in enumerate(hook_positions):
        acc = collect_layer_acc(model, safety_tokens, split, hook_position=hook, **kw)
        xs, ys = _sorted_points(acc, max_layer)
        if not xs:
            print(f"[skip] no data for hook='{hook}'")
            continue
        ax.scatter(xs, ys, color=colors[i], marker=MARKERS[i % len(MARKERS)],
                   s=100, alpha=0.9, label=_HOOK_NICE.get(hook, hook))
        ax.plot(xs, ys, color=colors[i], alpha=0.5, linewidth=2)

    _style_layer_axis(ax, split, max_layer, ymin=0.95)
    ax.legend(loc="lower center", bbox_to_anchor=(0.75, 0.1), ncol=1, fontsize=18,
              frameon=True, fancybox=True, markerscale=1.5, labelspacing=0.8)

    fig.tight_layout()
    path = ensure_output_dir(out_dir) / f"{split}_hook_position.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="HF ids for the all-model panel (default: all registered).")
    parser.add_argument("--split", choices=["train", "val"], default="train",
                        help="Which probe accuracy to plot.")
    parser.add_argument("--output-dir", default="figures", help="Directory for the PDFs.")
    parser.add_argument("--ckpt-dir", default="ckpts", help="Root of probe checkpoints.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-token", default="none")
    parser.add_argument("--no-gradual-cache", action="store_true",
                        help="Checkpoints were trained without gradual-cache.")
    parser.add_argument("--choice-model", default=CHOICE_MODEL_DEFAULT,
                        help="Model for the token-choice ablation panel.")
    parser.add_argument("--hook-model", default=HOOK_MODEL_DEFAULT,
                        help="Model for the hook-position ablation panel.")
    parser.add_argument("--panels", nargs="+",
                        choices=["all_model", "choice", "hook"],
                        default=["all_model", "choice", "hook"],
                        help="Which panels to render.")
    args = parser.parse_args()

    set_plot_style()
    common = dict(mask_tokens=args.mask_token, seed=args.seed,
                  gradual_cache=not args.no_gradual_cache, ckpt_dir=args.ckpt_dir)

    written = []
    if "all_model" in args.panels:
        written.append(plot_all_models(args.models, args.split, args.output_dir, **common))
    if "choice" in args.panels:
        written.append(plot_token_choice(args.choice_model, CHOICE_SAFETY_TOKENS,
                                         args.split, args.output_dir, **common))
    if "hook" in args.panels:
        written.append(plot_hook_positions(args.hook_model, HOOK_SAFETY_TOKENS,
                                           HOOK_POSITIONS, args.split, args.output_dir,
                                           **common))
    for p in written:
        print(f"saved {p}")


if __name__ == "__main__":
    main()
