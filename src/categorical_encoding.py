"""Train-only categorical encoding utilities.

These helpers keep categorical encoding leakage-free by fitting mappings on
training data only and applying them to validation/test splits.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _clean_cat(series: pd.Series) -> pd.Series:
    """Normalize categorical values with a stable, whitespace-trimmed string view."""
    return series.astype("string").str.strip()


def fit_train_category_maps(
    X_train: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray]]:
    """Fit per-column category maps using only the training split.

    Returns:
        mappings: column -> {category_string -> encoded_float}
        valid_values: column -> sorted valid encoded values
    """
    mappings: Dict[str, Dict[str, float]] = {}
    valid_values: Dict[str, np.ndarray] = {}

    for col in cat_cols:
        series = _clean_cat(X_train[col])
        categories = sorted(series.dropna().unique().tolist())
        mapping = {cat: float(i) for i, cat in enumerate(categories)}
        mappings[col] = mapping
        valid_values[col] = np.arange(len(categories), dtype=float)

    return mappings, valid_values


def encode_with_category_maps(
    X: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    mappings: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Encode a split using training-fitted category maps.

    Unknown categories are mapped to NaN, allowing imputers to handle them.

    Returns:
        encoded_df: numeric dataframe in [num_cols + cat_cols] order
        unseen_counts: per-column count of non-null values not seen in train
    """
    encoded = X.copy()

    for col in num_cols:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce").astype(float)

    unseen_counts: Dict[str, int] = {}
    for col in cat_cols:
        series = _clean_cat(encoded[col])
        mapped = series.map(mappings.get(col, {})).astype(float)
        unseen_counts[col] = int(series.notna().sum() - mapped.notna().sum())
        encoded[col] = mapped

    return encoded[num_cols + cat_cols], unseen_counts

