from __future__ import annotations

import math
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float] | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "support": int(len(y_true)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        ),
    }

    if y_score is None:
        metrics["roc_auc"] = math.nan
    else:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = math.nan

    return metrics

