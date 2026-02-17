"""
Shared metric computation and serialization utilities.

Extracted from run_classification.py and run_tabicl.py to eliminate code
duplication (ISSUE-8/9).
"""

import json
from ast import literal_eval
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


def to_numpy(x: Any) -> np.ndarray:
    """Convert array-like to numpy, handling cupy, pandas, and cuDF."""
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas().to_numpy()
    return np.asarray(x)


def one_hot_proba(y_pred: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Build one-hot probability matrix from hard predictions."""
    y_pred = to_numpy(y_pred).astype(int)
    proba = np.zeros((len(y_pred), len(classes)), dtype=float)
    class_pos = {int(c): i for i, c in enumerate(classes)}
    for i, y in enumerate(y_pred):
        pos = class_pos.get(int(y))
        if pos is not None:
            proba[i, pos] = 1.0
    return proba


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    classes: np.ndarray,
) -> Dict[str, Any]:
    """Compute classification metrics including AUC, F1, recall, and confusion matrix."""
    y_true = to_numpy(y_true).astype(int)
    y_pred = to_numpy(y_pred).astype(int)
    y_prob = to_numpy(y_prob)

    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    y_bin = label_binarize(y_true, classes=classes)

    try:
        metrics["auc_weighted"] = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        metrics["auc_weighted"] = np.nan

    try:
        metrics["auc_macro"] = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    except Exception:
        metrics["auc_macro"] = np.nan

    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=classes).tolist()
    return metrics


def serialize(obj: Any) -> Any:
    """Recursively serialize numpy types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def coerce_confusion_matrix(
    raw_cm: Any,
    expected_classes: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Parse a confusion matrix from various serialized formats.

    Returns an integer ndarray of shape (n, n), or None on failure.
    """
    cm_data = raw_cm
    if isinstance(cm_data, str):
        try:
            cm_data = json.loads(cm_data)
        except Exception:
            cm_data = literal_eval(cm_data)

    cm = np.asarray(cm_data)
    cm = np.squeeze(cm)

    if cm.ndim == 1:
        size = cm.size
        n = int(np.sqrt(size))
        if n * n == size:
            cm = cm.reshape(n, n)
        elif expected_classes and size == expected_classes * expected_classes:
            cm = cm.reshape(expected_classes, expected_classes)
        else:
            return None

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return None

    return cm.astype(int)
