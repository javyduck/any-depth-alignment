#!/usr/bin/env python3
"""Train the ADA-LP Safety-Token probe.

ADA-LP (Any-Depth Alignment, Linear-Probe variant) classifies a model's own
hidden state at the injected Safety-Token position as *about-to-refuse* (harmful,
label 1) vs *about-to-comply* (benign, label 0). This module fits the paper's
probe: an independent scikit-learn ``LogisticRegression`` per hook layer.

Input hidden states are produced by the collection stage and stored under::

    hidden_states/{split}/{model}/{data}/{safety}/{mask}/{hook}/{cache}/index_{i}/{layer}.pt

Each ``{layer}.pt`` is a tensor of shape ``(num_samples, num_depths + 1, hidden_size)``.
By default depth 0 (the user-token position) is dropped and the remaining
generation depths are flattened into independent training samples; ``--depth_zero``
instead keeps *only* depth 0. Harmful and benign splits are concatenated, rows
containing NaNs are dropped, and one classifier is fit per layer.

Trained probes are written under::

    ckpts/{model}/{safety}/{mask}/{hook}/{cache}/seed_{seed}/logistic/

as ``layer_{L}.joblib`` + ``layer_{L}.json`` (per-layer metrics) plus a single
``training_metadata.json``. All path slugs come from :mod:`ada.utils.naming` so
the layout stays byte-compatible with the rest of the pipeline.

Run with ``python -m ada.probe.train --model <hf_id> ...``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from ada.probe.metrics import tpr_tnr
from ada.registry import get_model
from ada.utils.io import write_json
from ada.utils.naming import (
    slugify_cache,
    slugify_hook_position,
    slugify_mask_tokens,
    slugify_model,
    slugify_safety_tokens,
)

logger = logging.getLogger(__name__)

# Number of index shards the collection stage writes per split.
NUM_INDEX_SHARDS = 8

# Fixed LogisticRegression hyperparameters for the ADA-LP probe. The paper states
# "default hyperparameters with tol=1e-4 and max_iter=1000"; PENALTY/C/SOLVER are
# the scikit-learn defaults. CLASS_WEIGHT="balanced" is the one deliberate
# non-default in the original implementation (the corpus is ~2:1 benign:harmful)
# and is what produced the released probes; the near-perfect separability makes
# the choice immaterial (set to None to match the literal "default" wording).
PENALTY = "l2"
C = 1.0
SOLVER = "lbfgs"
MAX_ITER = 1000
TOL = 1e-4
CLASS_WEIGHT = "balanced"
VERBOSE = 0
DEFAULT_SEED = 42


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_random_seeds(seed: int) -> None:
    """Seed all RNGs used during probe training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Hidden-state loading
# --------------------------------------------------------------------------- #
def load_hidden_states_from_index(
    base_dir: Path, layer: int, depth_zero: bool = False
) -> torch.Tensor:
    """Load and reshape one shard's hidden states for a single layer.

    The stored tensor has shape ``(num_samples, num_depths + 1, hidden_size)``.
    With ``depth_zero=False`` we drop depth 0 (the user-token position) and
    flatten depths ``1..N`` into rows; with ``depth_zero=True`` we keep only
    depth 0. Tensors that are not 3-D are returned unchanged.
    """
    layer_file = base_dir / f"{layer}.pt"
    hidden_states = torch.load(layer_file, map_location="cpu", weights_only=False)

    if hidden_states.dim() == 3:
        batch_size, num_depths, hidden_dim = hidden_states.shape
        if depth_zero:
            hidden_states = hidden_states[:, 0].reshape(-1, hidden_dim)
            logger.info(
                "Loaded depth0 %s hidden states from %s (selected [:,0] from %dx%dx%d)",
                tuple(hidden_states.shape), layer_file, batch_size, num_depths, hidden_dim,
            )
        else:
            hidden_states = hidden_states[:, 1:].reshape(-1, hidden_dim)
            logger.info(
                "Loaded %s hidden states from %s (reshaped from %dx%dx%d)",
                tuple(hidden_states.shape), layer_file, batch_size, num_depths, hidden_dim,
            )
    else:
        logger.info("Loaded %s hidden states from %s", tuple(hidden_states.shape), layer_file)
    return hidden_states


def load_all_hidden_states(
    base_dir: Path, layers: List[int], depth_zero: bool = False
) -> Dict[int, torch.Tensor]:
    """Concatenate hidden states across all index shards for each layer."""
    all_states: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}
    for index in range(NUM_INDEX_SHARDS):
        index_dir = base_dir / f"index_{index}"
        if not index_dir.exists():
            continue  # tolerate partial collection (fewer than 8 shards)
        for layer in layers:
            layer_file = index_dir / f"{layer}.pt"
            if not layer_file.exists():
                continue
            layer_states = load_hidden_states_from_index(index_dir, layer, depth_zero=depth_zero)
            if layer_states.numel() > 0:
                all_states[layer].append(layer_states)

    final_states: Dict[int, torch.Tensor] = {}
    for layer in layers:
        if all_states[layer]:
            final_states[layer] = torch.cat(all_states[layer], dim=0)
            logger.info("Layer %d: %s total hidden states", layer, tuple(final_states[layer].shape))
        else:
            final_states[layer] = torch.empty(0)
            logger.warning("No hidden states found for layer %d", layer)
    return final_states


def parse_layer_list(layers_str: str, base_dir: Optional[Path] = None) -> Optional[List[int]]:
    """Resolve a ``--layers`` spec against the on-disk shard layout.

    ``"all"`` auto-detects every ``{layer}.pt`` found across the index shards in
    ``base_dir``. Otherwise accepts explicit forms: ``"0,1,2"``, ``"[0,5,10]"``,
    ``"0-4"`` (inclusive) or ``"5:15"`` (exclusive end). Returns ``None`` when
    ``"all"`` is requested without a directory or no layer files exist.
    """
    layers_str = layers_str.strip()
    if layers_str.lower() == "all":
        if base_dir is None:
            logger.error("Base directory required for 'all' layer specification")
            return None
        available_layers = set()
        for index in range(NUM_INDEX_SHARDS):
            index_dir = base_dir / f"index_{index}"
            if index_dir.exists():
                for file_path in index_dir.glob("*.pt"):
                    try:
                        available_layers.add(int(file_path.stem))
                    except ValueError:
                        continue
        if not available_layers:
            logger.error("No layer files found in directory structure")
            return None
        layers_sorted = sorted(available_layers)
        logger.info(
            "Auto-detected %d hook indices: %s (0-based indexing)",
            len(layers_sorted), layers_sorted,
        )
        return layers_sorted

    layers_str = layers_str.replace("[", "").replace("]", "")
    if "-" in layers_str and "," not in layers_str:
        parts = layers_str.split("-")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end + 1))
    if ":" in layers_str and "," not in layers_str:
        parts = layers_str.split(":")
        if len(parts) == 2:
            start, end = int(parts[0]), int(parts[1])
            return list(range(start, end))
    if "," in layers_str:
        return [int(x.strip()) for x in layers_str.split(",")]
    return [int(layers_str)]


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Accuracy plus TPR/TNR (harmful = 1) at the 0.5 decision threshold."""
    y_pred = (y_prob >= 0.5).astype(np.int64)
    acc = float(accuracy_score(y_true, y_pred))
    tpr, tnr = tpr_tnr(y_true, y_pred)
    return {"accuracy": acc, "tpr": tpr, "tnr": tnr}


def build_classifier(seed: int) -> LogisticRegression:
    """Instantiate the paper's ADA-LP LogisticRegression with fixed hyperparameters."""
    return LogisticRegression(
        penalty=PENALTY,
        C=C,
        solver=SOLVER,
        max_iter=MAX_ITER,
        tol=TOL,
        class_weight=CLASS_WEIGHT,
        random_state=seed,
        verbose=VERBOSE,
    )


def build_xy(
    harmful: torch.Tensor, benign: torch.Tensor
) -> Tuple[np.ndarray, np.ndarray]:
    """Stack harmful (label 1) and benign (label 0) states, dropping NaN rows."""
    X = torch.cat([harmful, benign], dim=0).float().numpy()
    y = torch.cat(
        [
            torch.ones(harmful.shape[0], dtype=torch.long),
            torch.zeros(benign.shape[0], dtype=torch.long),
        ],
        dim=0,
    ).float().numpy()
    keep = ~np.isnan(X).any(axis=1)
    return X[keep], y[keep]


def train_layer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Tuple[LogisticRegression, Dict[str, float], Dict[str, float]]:
    """Fit one per-layer probe and return it with train/val metrics."""
    clf = build_classifier(seed)
    clf.fit(X_train, y_train)
    train_metrics = compute_metrics(y_train, clf.predict_proba(X_train)[:, 1])
    val_metrics = compute_metrics(y_val, clf.predict_proba(X_val)[:, 1])
    return clf, train_metrics, val_metrics


def build_layer_history(
    clf: LogisticRegression,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
) -> Dict[str, object]:
    """Assemble the per-layer JSON record saved alongside each ``.joblib``."""
    has_n_iter = hasattr(clf, "n_iter_") and getattr(clf, "n_iter_", None) is not None
    return {
        "train_accuracy": train_metrics["accuracy"],
        "train_tpr": train_metrics["tpr"],
        "train_tnr": train_metrics["tnr"],
        "val_accuracy": val_metrics["accuracy"],
        "val_tpr": val_metrics["tpr"],
        "val_tnr": val_metrics["tnr"],
        "coef_shape": list(clf.coef_.shape),
        "intercept_shape": (
            list(clf.intercept_.shape) if getattr(clf, "intercept_", None) is not None else None
        ),
        "penalty": PENALTY,
        "C": C,
        "solver": SOLVER,
        "max_iter": MAX_ITER,
        "tol": TOL,
        "class_weight": CLASS_WEIGHT,
        "n_iter": int(clf.n_iter_[0]) if has_n_iter else None,
        "n_iter_array": clf.n_iter_.tolist() if has_n_iter else None,
        "hit_max_iter": bool(np.max(clf.n_iter_) >= MAX_ITER) if has_n_iter else None,
        "classes": clf.classes_.tolist() if hasattr(clf, "classes_") else None,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ADA-LP logistic-regression Safety-Token probes."
    )
    parser.add_argument("--model", required=True, help="Model name / HF id (for directory structure)")
    parser.add_argument(
        "--safety-tokens", default=None,
        help="Safety tokens used during collection (default: the model's registry probe_safety_tokens)",
    )
    parser.add_argument(
        "--mask-tokens", default="",
        help="Mask tokens used during collection (for directory path)",
    )
    parser.add_argument(
        "--hook-position", type=str, default=None,
        help="Hook position used during collection (default: the model's registry hook_position)",
    )
    parser.add_argument(
        "--layers", type=str, default="all",
        help="Hook indices to train probes for (0-based). Examples: 'all', '0,1,2', '[0,5,10]', '0-4', '5:15'",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    # Must match how ada.probe.collect wrote the states (default: gradual cache,
    # the released-probe / E1-script / evaluate convention).
    parser.add_argument("--gradual-cache", dest="gradual_cache", action="store_true",
                        help="(default) hidden states collected with the gradual KV cache")
    parser.add_argument("--full-forward", dest="gradual_cache", action="store_false",
                        help="Hidden states collected with the full-forward strategy")
    parser.set_defaults(gradual_cache=True)
    parser.add_argument(
        "--depth_zero", action="store_true",
        help="Train on depth 0 (prompt-level) hidden states; otherwise use depths 1..N",
    )
    parser.add_argument(
        "--list-layers", action="store_true",
        help="Print available layers (auto-detected from disk) as a JSON array and exit",
    )
    parser.add_argument(
        "--hidden-states-dir", default="hidden_states",
        help="Directory containing collected hidden states (default: CWD-relative)",
    )
    parser.add_argument(
        "--ckpt-dir", default="ckpts",
        help="Directory to save probe checkpoints (default: CWD-relative)",
    )
    return parser


def _process_tokens(s: Optional[str]) -> Optional[str]:
    """Turn literal ``\\n`` in CLI token strings into real newlines."""
    return s.replace("\\n", "\n") if s is not None else s


def _show_literal_newlines(s: object) -> str:
    return s.replace("\n", r"\n") if isinstance(s, str) else str(s)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    args = create_argument_parser().parse_args()

    # Default Safety Tokens / hook position to the registry so the training paths
    # line up with what ada.probe.collect wrote (single source of truth).
    if args.safety_tokens is None or args.hook_position is None:
        spec = get_model(args.model)
        if args.safety_tokens is None:
            args.safety_tokens = spec.probe_safety_tokens
        if args.hook_position is None:
            args.hook_position = spec.hook_position

    args.safety_tokens = _process_tokens(args.safety_tokens)
    if args.mask_tokens is not None:
        args.mask_tokens = _process_tokens(args.mask_tokens)

    set_random_seeds(args.seed)
    logger.info("Set random seed to %d", args.seed)

    # Resolve path slugs (shared source of truth in ada.utils.naming).
    model_slug = slugify_model(args.model)
    safety_slug = slugify_safety_tokens(args.safety_tokens)
    mask_slug = slugify_mask_tokens(args.mask_tokens)
    hook_slug = slugify_hook_position(args.hook_position)
    cache_slug = slugify_cache(args.gradual_cache)

    def split_dir(split: str, data_type: str) -> Path:
        return (
            Path(args.hidden_states_dir) / split / model_slug / data_type
            / safety_slug / mask_slug / hook_slug / cache_slug
        )

    train_harmful_dir = split_dir("train", "harmful")
    train_benign_dir = split_dir("train", "benign")
    val_harmful_dir = split_dir("val", "harmful")
    val_benign_dir = split_dir("val", "benign")

    # --list-layers: auto-detect and emit a JSON array, then exit.
    if args.list_layers:
        layers = parse_layer_list("all", base_dir=train_harmful_dir)
        if layers is None:
            layers = parse_layer_list("all", base_dir=val_harmful_dir)
        print(json.dumps(layers if layers is not None else []))
        return

    target_layers = parse_layer_list(args.layers, base_dir=train_harmful_dir)
    if target_layers is None:
        logger.error("Failed to parse layer specification or no layers found")
        return

    ckpt_dir = (
        Path(args.ckpt_dir) / model_slug / safety_slug / mask_slug / hook_slug
        / cache_slug / f"seed_{args.seed}" / "logistic"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving checkpoints to: %s", ckpt_dir)

    logger.info("Loading train/val hidden states...")
    train_harm = load_all_hidden_states(train_harmful_dir, target_layers, depth_zero=args.depth_zero)
    train_ben = load_all_hidden_states(train_benign_dir, target_layers, depth_zero=args.depth_zero)
    val_harm = load_all_hidden_states(val_harmful_dir, target_layers, depth_zero=args.depth_zero)
    val_ben = load_all_hidden_states(val_benign_dir, target_layers, depth_zero=args.depth_zero)

    # Keep only layers present (and non-empty) in all four splits.
    valid_layers = [
        layer for layer in sorted(target_layers)
        if all(
            d.get(layer, None) is not None and d[layer].numel() > 0
            for d in (train_harm, train_ben, val_harm, val_ben)
        )
    ]
    if not valid_layers:
        logger.error("No layers with complete data found")
        return

    hidden_size = None
    for layer in valid_layers:
        if train_harm[layer].numel() > 0:
            hidden_size = int(train_harm[layer].shape[-1])
            break
    if hidden_size is None:
        logger.error("Could not determine hidden size")
        return
    logger.info("Hidden size: %d; Num valid layers: %d", hidden_size, len(valid_layers))

    summary: Dict[int, Dict[str, object]] = {}
    name_suffix = "_depth_0" if args.depth_zero else ""

    for layer in valid_layers:
        logger.info("Training LogisticRegression for layer %d...", layer)

        X_train, y_train = build_xy(train_harm[layer], train_ben[layer])
        X_val, y_val = build_xy(val_harm[layer], val_ben[layer])

        clf, train_metrics, val_metrics = train_layer(X_train, y_train, X_val, y_val, args.seed)

        logger.info(
            "Layer %2d - Train Acc: %.4f, Train TPR: %.4f, Train TNR: %.4f, "
            "Val Acc: %.4f, Val TPR: %.4f, Val TNR: %.4f",
            layer, train_metrics["accuracy"], train_metrics["tpr"], train_metrics["tnr"],
            val_metrics["accuracy"], val_metrics["tpr"], val_metrics["tnr"],
        )

        joblib.dump(clf, ckpt_dir / f"layer_{layer}{name_suffix}.joblib")

        history = build_layer_history(clf, train_metrics, val_metrics)
        write_json(ckpt_dir / f"layer_{layer}{name_suffix}.json", history)
        summary[layer] = history

        del X_train, y_train, X_val, y_val

    metadata = {
        "model": args.model,
        "safety_tokens": args.safety_tokens,
        "mask_tokens": args.mask_tokens,
        "hook_position": args.hook_position,
        "gradual_cache": args.gradual_cache,
        "depth_zero": bool(args.depth_zero),
        "layers": valid_layers,
        "seed": args.seed,
        "hidden_size": hidden_size,
        "classifier": "sklearn.LogisticRegression",
        "hyperparameters": {
            "penalty": PENALTY,
            "C": C,
            "solver": SOLVER,
            "max_iter": MAX_ITER,
            "tol": TOL,
            "class_weight": CLASS_WEIGHT,
        },
    }
    write_json(ckpt_dir / "training_metadata.json", metadata)

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE (ADA-LP Logistic Regression)")
    logger.info("=" * 60)
    logger.info("Model: %s", args.model)
    logger.info("Safety tokens: %s", _show_literal_newlines(args.safety_tokens))
    logger.info("Layers trained: %d", len(valid_layers))
    logger.info("Hidden size: %d", hidden_size)
    logger.info("Checkpoints saved to: %s", ckpt_dir)


if __name__ == "__main__":
    main()
