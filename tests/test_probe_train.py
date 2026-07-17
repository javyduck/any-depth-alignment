"""ADA-LP probe training helpers + a tiny end-to-end sklearn fit (no model)."""

from __future__ import annotations

import numpy as np
import torch

from ada.probe.train import (
    build_classifier,
    build_xy,
    compute_metrics,
    train_layer,
    C,
    CLASS_WEIGHT,
    MAX_ITER,
    TOL,
)


def test_build_xy_labels_and_nan_drop():
    harmful = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    benign = torch.tensor([[0.0, 0.0], [float("nan"), 0.0]])
    X, y = build_xy(harmful, benign)
    assert X.shape == (3, 2)              # the NaN benign row is dropped
    assert list(y) == [1.0, 1.0, 0.0]    # harmful=1 then benign=0


def test_compute_metrics_perfect():
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2])
    m = compute_metrics(y_true, y_prob)
    assert m["accuracy"] == 1.0 and m["tpr"] == 1.0 and m["tnr"] == 1.0


def test_classifier_hyperparameters_match_paper():
    clf = build_classifier(seed=42)
    assert clf.max_iter == MAX_ITER == 1000
    assert clf.tol == TOL == 1e-4
    assert clf.C == C == 1.0
    assert clf.class_weight == CLASS_WEIGHT == "balanced"


def test_train_layer_separable_reaches_perfect_val():
    rng = np.random.default_rng(0)
    # linearly separable: harmful centred at +3, benign at -3.
    harm = torch.tensor(rng.normal(3, 0.3, size=(200, 8)), dtype=torch.float32)
    ben = torch.tensor(rng.normal(-3, 0.3, size=(200, 8)), dtype=torch.float32)
    X_tr, y_tr = build_xy(harm[:150], ben[:150])
    X_va, y_va = build_xy(harm[150:], ben[150:])
    _clf, tr, va = train_layer(X_tr, y_tr, X_va, y_va, seed=42)
    assert va["accuracy"] == 1.0
