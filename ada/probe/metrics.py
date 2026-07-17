"""Classification metrics for the ADA-LP probe."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def tpr_tnr(labels: Sequence[int], predictions: Sequence[int]) -> Tuple[float, float]:
    """Return (true-positive rate, true-negative rate) with 1 = harmful.

    TPR is recall on harmful examples; TNR is recall on benign examples. Each
    defaults to 0.0 when its class is absent from the batch.
    """
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)

    pos = labels == 1
    neg = labels == 0
    tpr = float((predictions[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((predictions[neg] == 0).mean()) if neg.any() else 0.0
    return tpr, tnr
