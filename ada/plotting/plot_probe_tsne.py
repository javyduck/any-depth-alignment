"""t-SNE of Safety-Token hidden states (``plot_distribution.ipynb``).

Shows, for a single model and probe layer, how the injected Safety-Token hidden
state separates benign vs harmful continuations at increasing generation depth,
while the last-generated-token hidden state does not. Produces a 2x4 panel:

* rows  = the two probed positions
    - ``Last Generated Token``   (``empty`` safety-token condition)
    - ``Injected Safety Token``  (``registry.probe_safety_tokens``)
* cols  = four generation depths (token counts)

Each panel runs t-SNE on the combined benign+harmful hidden states at that
(position, depth) and annotates the 2-D logistic-regression accuracy. Output:
``{output-dir}/tsne_distribution.pdf``.

Hidden states are read from::

    hidden_states/train/{model_slug}/{benign|harmful}/{safety_slug}/
        mask_token_none/hook_input_layernorm/gradual_cache/index_{i}/{layer}.pt

each a tensor of shape ``(N, num_depths, hidden_size)``; the ``--depths`` values
index the middle (depth) axis. ``--model`` / ``--layer`` are parameterized
(default: Llama-3.1-8B-Instruct at its registry probe layer).

Run: ``python -m ada.plotting.plot_probe_tsne``
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np

from ada.plotting._common import (
    ensure_output_dir,
    hidden_states_index_dir,
    probe_layer,
    probe_safety_tokens,
)

# Depth-axis index -> number of generated tokens. The collection schedule is a
# uniform 25-token grid (ada.probe.collect.DEPTHS = 0,25,...,500), so column
# ``i`` corresponds to ``25*i`` generated tokens.
TOKENS_PER_DEPTH = 25

# Which index shards contribute to each class (paper configuration): the benign
# class pools the vanilla + adversarial-benign shards, the harmful class all shards.
LEGENDS = {
    "Benign": ("benign", [0, 1, 5, 7]),
    "Harmful": ("harmful", [0, 1, 2, 3, 4, 5, 6, 7]),
}

ROW_LABELS = ["Last Generated Token", "Injected Safety Token"]
CLASS_COLORS = ["#2166ac", "#f7768d"]  # navy (benign), crimson (harmful)


def _load_slice(model, data_type, safety_tokens, layer, depth, indices, hs_dir):
    """Stack hidden states at one depth slice across the given index shards."""
    import torch

    arrays: List[np.ndarray] = []
    for index in indices:
        layer_file = hidden_states_index_dir(
            "train", model, data_type, safety_tokens, index,
            hidden_states_dir=hs_dir,
        ) / f"{layer}.pt"
        if not layer_file.exists():
            print(f"[warn] missing {layer_file}")
            continue
        t = torch.load(layer_file, map_location="cpu", weights_only=False)
        if t.dim() != 3:
            print(f"[warn] unexpected shape {tuple(t.shape)} in {layer_file}")
            continue
        if not 0 <= depth < t.shape[1]:
            print(f"[warn] depth index {depth} out of range for {layer_file}")
            continue
        arrays.append(t[:, depth, :].contiguous().float().numpy())
    if not arrays:
        return np.empty((0, 0))
    return np.vstack(arrays)


def _logreg_accuracy(x, y):
    """Train logistic regression on the 2-D embedding and return its accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
                             tol=1e-4, class_weight="balanced", random_state=42)
    clf.fit(x, y)
    return accuracy_score(y, clf.predict(x))


def _scatter(ax, emb, labels, s=2.5):
    from matplotlib.colors import to_rgba

    rgba = np.array([to_rgba(CLASS_COLORS[int(lbl)], 0.35) for lbl in labels])
    order = np.random.permutation(len(labels))
    ax.scatter(emb[order, 0], emb[order, 1], c=rgba[order], s=s, linewidths=0.05,
               edgecolors=(1, 1, 1, 0.35), alpha=0.6, rasterized=True)


def build_figure(model, layer, depths, *, perplexity=50, seed=42,
                 hs_dir="hidden_states", max_samples_per_group=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    np.random.seed(seed)
    conditions = ["empty", probe_safety_tokens(model)]  # rows

    plt.style.use("default")
    mpl.rcParams.update({
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.titlesize": 10, "axes.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
    })

    fig, axes = plt.subplots(2, len(depths), figsize=(12, 6), dpi=300)
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    legend_names = list(LEGENDS.keys())

    for row, safety_tokens in enumerate(conditions):
        for col, depth in enumerate(depths):
            ax = axes[row, col]
            feats, labels = [], []
            for label_idx, name in enumerate(legend_names):
                data_type, indices = LEGENDS[name]
                arr = _load_slice(model, data_type, safety_tokens, layer, depth,
                                  indices, hs_dir)
                if arr.shape[0] == 0:
                    continue
                if max_samples_per_group and arr.shape[0] > max_samples_per_group:
                    sel = np.random.choice(arr.shape[0], max_samples_per_group, replace=False)
                    arr = arr[sel]
                feats.append(arr)
                labels.append(np.full(arr.shape[0], label_idx))
            features = np.vstack(feats)
            y = np.concatenate(labels)

            emb = TSNE(n_components=2, perplexity=perplexity,
                       random_state=seed).fit_transform(features)
            _scatter(ax, emb, y)
            acc = _logreg_accuracy(emb, y) * 100
            ax.text(0.98, 0.02, f"Acc. = {acc:.1f}%", transform=ax.transAxes,
                    fontsize=10, ha="right", va="bottom",
                    bbox=dict(facecolor="lightgray", edgecolor="none",
                              boxstyle="round,pad=0.2", alpha=0.7))

            ax.set_xlabel("t-SNE 1", fontsize=10, labelpad=2)
            ax.set_ylabel("t-SNE 2", fontsize=10, labelpad=-2)
            ax.tick_params(axis="both", which="major", labelsize=8)
            ax.grid(True, alpha=0.3, linewidth=0.5)
            if row == 0:
                ax.set_title(f"Depth = {depth * TOKENS_PER_DEPTH} Tokens",
                             fontsize=11, pad=10)
            if col == 0:
                ax.text(-0.25, 0.5, ROW_LABELS[row], transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=12)
            if col == len(depths) - 1:
                handles = [plt.Line2D([0], [0], marker="o", color="w",
                                      label=legend_names[i],
                                      markerfacecolor=CLASS_COLORS[i], markersize=2.5)
                           for i in range(len(legend_names))]
                ax.legend(handles=handles, loc="upper right", frameon=True,
                          fancybox=False, shadow=False, fontsize=9, markerscale=3,
                          handletextpad=0.0, borderpad=0.05)

    fig.tight_layout(pad=0.02, w_pad=0.02, h_pad=0.02)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HF id of the model to visualize.")
    parser.add_argument("--layer", type=int, default=None,
                        help="Hidden-state layer (default: model's registry probe layer).")
    parser.add_argument("--depths", default="0,2,4,8",  # 0/50/100/200 tokens (paper fig:tsne)
                        help="Comma-separated depth-axis indices (4 columns).")
    parser.add_argument("--output-dir", default="figures")
    parser.add_argument("--hidden-states-dir", default="hidden_states")
    parser.add_argument("--perplexity", type=float, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples-per-group", type=int, default=None,
                        help="Optional subsample per class per panel (default: use all).")
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    layer = args.layer if args.layer is not None else probe_layer(args.model)
    depths = [int(d) for d in args.depths.split(",")]
    fig = build_figure(args.model, layer, depths, perplexity=args.perplexity,
                       seed=args.seed, hs_dir=args.hidden_states_dir,
                       max_samples_per_group=args.max_samples_per_group)
    path = ensure_output_dir(args.output_dir) / "tsne_distribution.pdf"
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
