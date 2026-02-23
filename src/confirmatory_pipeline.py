"""
ConfirmatoryPipeline — leakage-free inner-fold pipeline.

Wraps encode → impute → scale → classify as a single sklearn estimator so
that _manual_random_search can tune classifier hyperparameters without any of
the pre-processing steps leaking information across inner folds.

Each call to .fit() re-trains the full pipeline from scratch on whatever
subset it receives, exactly as required by a correct nested CV.
"""

from __future__ import annotations

import gc
from typing import Dict, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler

from src.categorical_encoding import encode_with_category_maps, fit_train_category_maps


class ConfirmatoryPipeline(BaseEstimator, ClassifierMixin):
    """Leakage-free pipeline: encode → impute → scale → classify.

    Parameters
    ----------
    clf : sklearn-compatible classifier
    imputer : sklearn-compatible imputer (or None for NoImpute)
    num_cols : list of numeric column names in the input DataFrame
    cat_cols : list of categorical column names in the input DataFrame
    needs_scaling : bool — whether to apply StandardScaler before clf
    cat_cols_for_catboost : list or None — if set, passes cat_features to clf.fit()

    Notes
    -----
    The estimator accepts a raw DataFrame (before any encoding) as X.
    set_params / get_params expose the classifier's hyperparameters directly,
    so _manual_random_search can grid-search over them transparently.
    """

    def __init__(
        self,
        clf,
        imputer=None,
        *,
        num_cols: List[str],
        cat_cols: List[str],
        needs_scaling: bool = False,
        cat_cols_for_catboost: Optional[List[str]] = None,
    ):
        self.clf = clf
        self.imputer = imputer
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.needs_scaling = needs_scaling
        self.cat_cols_for_catboost = cat_cols_for_catboost

        # Fitted state (populated by .fit)
        self._cat_maps: Optional[Dict] = None
        self._valid_values: Optional[Dict] = None
        self._scaler: Optional[StandardScaler] = None
        self._clf_fitted = None

    # ── sklearn API ────────────────────────────────────────────────────

    def get_params(self, deep: bool = True) -> dict:
        """Expose clf hyperparameters at the top level for grid search."""
        params = {
            "clf": self.clf,
            "imputer": self.imputer,
            "num_cols": self.num_cols,
            "cat_cols": self.cat_cols,
            "needs_scaling": self.needs_scaling,
            "cat_cols_for_catboost": self.cat_cols_for_catboost,
        }
        if deep and self.clf is not None:
            clf_params = self.clf.get_params(deep=True)
            params.update({f"clf__{k}": v for k, v in clf_params.items()})
        return params

    def set_params(self, **params) -> "ConfirmatoryPipeline":
        """Route clf__ prefixed params to the inner classifier."""
        clf_params = {}
        own_params = {}
        for k, v in params.items():
            if k.startswith("clf__"):
                clf_params[k[5:]] = v
            else:
                own_params[k] = v
        if own_params:
            super().set_params(**own_params)
        if clf_params and self.clf is not None:
            self.clf.set_params(**clf_params)
        return self

    # ── Internal helpers ───────────────────────────────────────────────

    def _encode(self, X_raw) -> "pd.DataFrame":
        """Encode using fitted category maps. Returns a named DataFrame.

        Returning a DataFrame (not ndarray) is required so that
        ColumnTransformer-based imputers (_build_imputers) can locate
        columns by name.
        """
        import pandas as pd
        if not isinstance(X_raw, pd.DataFrame):
            raise TypeError("ConfirmatoryPipeline expects a pandas DataFrame as X.")
        enc, _ = encode_with_category_maps(X_raw, self.num_cols, self.cat_cols, self._cat_maps)
        # enc is already a DataFrame with columns = num_cols + cat_cols
        return enc

    def _impute(self, X_enc: "pd.DataFrame", fit: bool = False) -> np.ndarray:
        """Apply imputer (or NaN fill if no imputer). Returns float32 ndarray.

        Passes the named DataFrame to the imputer so ColumnTransformer
        column selectors continue to work correctly.
        """
        import pandas as pd
        if self.imputer is None:
            arr = X_enc.values.astype(np.float32) if isinstance(X_enc, pd.DataFrame) else np.asarray(X_enc, np.float32)
            return np.nan_to_num(arr, nan=0.0)
        if fit:
            self._imp_fitted = clone(self.imputer)
            self._imp_fitted.fit(X_enc)   # X_enc is a DataFrame ← fixes ColumnTransformer
        out = self._imp_fitted.transform(X_enc)
        # transform() may return DataFrame or ndarray depending on imputer
        if hasattr(out, "values"):
            out = out.values
        return np.nan_to_num(out.astype(np.float32), nan=0.0)

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if not self.needs_scaling:
            return X
        if fit:
            self._scaler = StandardScaler()
            return self._scaler.fit_transform(X)
        return self._scaler.transform(X)

    # ── Public API ─────────────────────────────────────────────────────

    def fit(self, X_raw, y, sample_weight=None, **fit_kwargs):
        """Fit all stages on the provided training data."""
        # 1. Encoding
        self._cat_maps, self._valid_values = fit_train_category_maps(X_raw, self.cat_cols)
        X_enc = self._encode(X_raw)

        # 2. Imputation
        X_imp = self._impute(X_enc, fit=True)

        # 3. Scaling
        X_scaled = self._scale(X_imp, fit=True)

        # 4. Classifier — optional CatBoost cat_features
        clf = clone(self.clf)
        extra = dict(fit_kwargs)
        if self.cat_cols_for_catboost:
            n_cats = len(self.cat_cols_for_catboost)
            n_feats = X_scaled.shape[1]
            if n_cats <= n_feats:
                extra["cat_features"] = list(range(n_feats - n_cats, n_feats))

        if sample_weight is not None:
            try:
                clf.fit(X_scaled, y, sample_weight=sample_weight, **extra)
            except TypeError:
                clf.fit(X_scaled, y, **extra)
        else:
            clf.fit(X_scaled, y, **extra)

        self._clf_fitted = clf
        return self

    def predict(self, X_raw):
        X_enc = self._encode(X_raw)
        X_imp = self._impute(X_enc, fit=False)
        X_scaled = self._scale(X_imp, fit=False)
        return self._clf_fitted.predict(X_scaled)

    def predict_proba(self, X_raw):
        X_enc = self._encode(X_raw)
        X_imp = self._impute(X_enc, fit=False)
        X_scaled = self._scale(X_imp, fit=False)
        if hasattr(self._clf_fitted, "predict_proba"):
            return self._clf_fitted.predict_proba(X_scaled)
        raise AttributeError(f"{type(self._clf_fitted).__name__} has no predict_proba")

    @property
    def classes_(self):
        return getattr(self._clf_fitted, "classes_", None)
