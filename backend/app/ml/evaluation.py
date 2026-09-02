from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_probabilities(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    calibration_bins: int = 10,
) -> dict[str, float | int]:
    """Evaluate discrimination, probability quality, and calibration."""

    labels = np.asarray(y_true, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    predictions = (values >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    majority_accuracy = max(float((labels == 0).mean()), float((labels == 1).mean()))
    observed, predicted = calibration_curve(
        labels,
        values,
        n_bins=calibration_bins,
        strategy="uniform",
    )
    gaps = np.abs(observed - predicted)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
        "log_loss": float(log_loss(labels, values, labels=[0, 1])),
        "brier_score": float(brier_score_loss(labels, values)),
        "calibration_mean_absolute_error": float(gaps.mean()),
        "calibration_max_error": float(gaps.max()),
        "calibration_nonempty_bins": int(len(gaps)),
        "majority_class_accuracy": majority_accuracy,
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
