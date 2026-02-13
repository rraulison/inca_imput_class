"""
Step 2 - Imputation.
Input: data/processed/ artifacts from Step 1.
Output: data/imputed/ (folded train/test parquet files + metadata).

Imputer is fit only on the train split of each fold to avoid data leakage.
"""

import gc
import hashlib
import json
import logging
import pickle
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from xgboost import XGBRegressor

from src.config_loader import load_config

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

try:
    from cuml.ensemble import RandomForestRegressor as CuMLRandomForestRegressor
except Exception:  # pragma: no cover - optional dependency
    CuMLRandomForestRegressor = None

log = logging.getLogger(__name__)

NO_IMPUTER_NAME = "NoImpute"
SPLIT_SIGNATURE_FILE = "split_signature.json"


class CategoricalRounder(BaseEstimator, TransformerMixin):
    """Round imputed categorical values to nearest valid encoded label."""

    def __init__(self, cat_indices, valid_values_list):
        self.cat_indices = cat_indices
        self.valid_values_list = valid_values_list

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    def transform(self, X):
        X = np.asarray(X).copy()
        for idx, valid in zip(self.cat_indices, self.valid_values_list):
            if len(valid) > 0:
                X[:, idx] = np.clip(np.round(X[:, idx]), valid.min(), valid.max())
        return X

    def __sklearn_is_fitted__(self):
        return getattr(self, "_is_fitted", False)


class CuMLRFRegressorAdapter(BaseEstimator, RegressorMixin):
    """Minimal sklearn-compatible adapter for cuML RandomForestRegressor."""

    def __init__(self, n_estimators=100, max_depth=12, min_samples_leaf=5, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.model_ = None

    def fit(self, X, y):
        if cp is None or CuMLRandomForestRegressor is None:
            raise RuntimeError("cuML RandomForestRegressor is unavailable.")
        X_gpu = cp.asarray(np.asarray(X), dtype=cp.float32)
        y_gpu = cp.asarray(np.asarray(y), dtype=cp.float32)
        self.model_ = CuMLRandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.model_.fit(X_gpu, y_gpu)
        return self

    def predict(self, X):
        X_gpu = cp.asarray(np.asarray(X), dtype=cp.float32)
        pred = self.model_.predict(X_gpu)
        return cp.asnumpy(pred)


class SafeColumnTransformer(ColumnTransformer):
    """ColumnTransformer with explicit ndarray output for robust downstream usage."""

    def fit_transform(self, X, y=None):
        out = super().fit_transform(X, y)
        return np.asarray(out)

    def transform(self, X):
        out = super().transform(X)
        return np.asarray(out)


class Float32Caster(BaseEstimator, TransformerMixin):
    """Cast arrays/dataframes to float32 to reduce memory and distance costs."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=np.float32)


class SubsampledKNNImputer(BaseEstimator, TransformerMixin):
    """KNNImputer wrapper that can fit on a random subset for scalability."""

    def __init__(self, n_neighbors=5, weights="uniform", metric="nan_euclidean", max_fit_samples=None, random_state=42):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.max_fit_samples = max_fit_samples
        self.random_state = random_state
        self.imputer_ = None
        self._fit_rows_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X)
        n_rows = X_arr.shape[0]
        X_fit = X_arr

        if self.max_fit_samples is not None and self.max_fit_samples > 0 and n_rows > self.max_fit_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(n_rows, size=self.max_fit_samples, replace=False)
            X_fit = X_arr[idx]
            self._fit_rows_ = int(self.max_fit_samples)
        else:
            self._fit_rows_ = int(n_rows)

        self.imputer_ = KNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
        )
        self.imputer_.fit(X_fit)
        return self

    def transform(self, X):
        return self.imputer_.transform(np.asarray(X))


def _build_missforest_estimator(cfg):
    p = cfg["imputation"]["params"]
    seed = cfg["experiment"]["random_seed"]
    use_gpu = cfg["hardware"]["use_gpu"]

    if use_gpu and cp is not None and CuMLRandomForestRegressor is not None:
        log.info("MissForest estimator: cuML RandomForestRegressor (GPU)")
        return CuMLRFRegressorAdapter(
            n_estimators=p["missforest_n_estimators"],
            max_depth=p["missforest_max_depth"],
            min_samples_leaf=5,
            random_state=seed,
        )

    log.info("MissForest estimator: sklearn RandomForestRegressor (CPU fallback)")
    return RandomForestRegressor(
        n_estimators=p["missforest_n_estimators"],
        max_depth=p["missforest_max_depth"],
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )


def _build_imputers(num_cols, cat_cols, valid_values, cfg):
    seed = cfg["experiment"]["random_seed"]
    p = cfg["imputation"]["params"]
    use_gpu = cfg["hardware"]["use_gpu"]

    all_cols = num_cols + cat_cols
    cat_idx = [all_cols.index(c) for c in cat_cols]
    cat_valid = [valid_values.get(c, np.array([])) for c in cat_cols]
    rounder = CategoricalRounder(cat_idx, cat_valid)

    xgb_device = cfg["hardware"].get("gpu_device", "cuda") if use_gpu else "cpu"
    xgb_jobs = 1 if use_gpu else -1

    imputers = OrderedDict()

    imputers["Media"] = SafeColumnTransformer(
        [
            ("num", SimpleImputer(strategy="mean"), num_cols),
            ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
        ],
        remainder="drop",
    )

    imputers["Mediana"] = SafeColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", SimpleImputer(strategy="most_frequent"), cat_cols),
        ],
        remainder="drop",
    )

    imputers["kNN"] = Pipeline(
        [
            ("float32", Float32Caster()),
            (
                "imputer",
                SubsampledKNNImputer(
                    n_neighbors=p["knn_neighbors"],
                    weights="uniform",
                    metric="nan_euclidean",
                    max_fit_samples=p.get("knn_max_fit_samples"),
                    random_state=seed,
                ),
            ),
            ("rounder", clone(rounder)),
        ]
    )

    imputers["MICE_XGBoost"] = Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    estimator=XGBRegressor(
                        n_estimators=p["xgb_n_estimators"],
                        max_depth=p["xgb_max_depth"],
                        learning_rate=p["xgb_learning_rate"],
                        subsample=0.8,
                        colsample_bytree=0.8,
                        device=xgb_device,
                        tree_method="hist",
                        n_jobs=xgb_jobs,
                        random_state=seed,
                        verbosity=0,
                    ),
                    max_iter=p["xgb_max_iter"],
                    initial_strategy="most_frequent",
                    random_state=seed,
                    verbose=0,
                ),
            ),
            ("rounder", clone(rounder)),
        ]
    )

    imputers["MICE"] = Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    estimator=BayesianRidge(),
                    max_iter=p["mice_max_iter"],
                    initial_strategy="most_frequent",
                    random_state=seed,
                    verbose=0,
                ),
            ),
            ("rounder", clone(rounder)),
        ]
    )

    missforest_est = _build_missforest_estimator(cfg)
    imputers["MissForest"] = Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    estimator=missforest_est,
                    max_iter=p["missforest_max_iter"],
                    initial_strategy="most_frequent",
                    random_state=seed,
                    verbose=0,
                ),
            ),
            ("rounder", clone(rounder)),
        ]
    )

    return imputers


def _fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def _split_signature(y, fold_indices, n_folds):
    y_arr = np.asarray(y, dtype=np.int32)
    y_hash = hashlib.sha1(y_arr.tobytes()).hexdigest()
    payload = {
        "n_rows": int(len(y_arr)),
        "n_folds": int(n_folds),
        "y_sha1": y_hash,
        "fold_indices": fold_indices,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return {
        "hash": hashlib.sha1(canonical.encode("utf-8")).hexdigest(),
        "n_rows": payload["n_rows"],
        "n_folds": payload["n_folds"],
        "y_sha1": payload["y_sha1"],
    }


def run_imputation(config_path="config/config.yaml", filter_imputers=None):
    cfg = load_config(config_path)
    seed = cfg["experiment"]["random_seed"]
    n_folds = cfg["cv"]["n_outer_folds"]

    out_dir = Path(cfg["paths"]["imputed_data"])
    res_dir = Path(cfg["paths"]["results_raw"])
    out_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    proc_dir = Path(cfg["paths"]["processed_data"])
    X = pd.read_parquet(proc_dir / "X_prepared.parquet")
    y = pd.read_parquet(proc_dir / "y_prepared.parquet")["target"]

    with open(proc_dir / "encoders.pkl", "rb") as f:
        enc = pickle.load(f)

    with open(Path(cfg["paths"]["results_tables"]) / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    all_cols = num_cols + cat_cols
    X = X[all_cols]

    log.info("X shape: %s | total missing: %s", X.shape, f"{int(X.isna().sum().sum()):,}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_indices = {}
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        fold_indices[fold] = {"train": tr_idx.tolist(), "test": te_idx.tolist()}
        for split_name, idx in (("train", tr_idx), ("test", te_idx)):
            yp = out_dir / f"y_fold{fold}_{split_name}.parquet"
            y.iloc[idx].to_frame("target").to_parquet(yp, index=False)

    with open(out_dir / "fold_indices.json", "w", encoding="utf-8") as f:
        json.dump(fold_indices, f)

    current_sig = _split_signature(y, fold_indices, n_folds)
    sig_path = out_dir / SPLIT_SIGNATURE_FILE
    force_recompute = False
    if sig_path.exists():
        with open(sig_path, "r", encoding="utf-8") as f:
            previous_sig = json.load(f)
        if previous_sig.get("hash") != current_sig["hash"]:
            force_recompute = True
            log.warning(
                "Detected split/target signature change (%s -> %s). Recomputing all imputer folds.",
                previous_sig.get("hash"),
                current_sig["hash"],
            )
    with open(sig_path, "w", encoding="utf-8") as f:
        json.dump(current_sig, f, indent=2)

    base_imputers = _build_imputers(num_cols, cat_cols, enc["valid_values"], cfg)
    cfg_imputers = cfg.get("imputation", {}).get("imputers") or list(base_imputers.keys()) + [NO_IMPUTER_NAME]

    imputers = OrderedDict()
    seen = set()
    for name in cfg_imputers:
        if name in seen:
            continue
        seen.add(name)
        if name in base_imputers:
            imputers[name] = base_imputers[name]
        elif name == NO_IMPUTER_NAME:
            imputers[name] = None  # No-imputation baseline for native NaN handling models.
        else:
            log.warning("Configured imputer '%s' is not available and will be skipped.", name)

    if filter_imputers:
        imputers = OrderedDict((k, v) for k, v in imputers.items() if k in filter_imputers)

    tempos_path = res_dir / "tempos_imputacao.json"
    tempos = json.load(open(tempos_path, "r", encoding="utf-8")) if tempos_path.exists() else {}

    for imp_name, imputer in imputers.items():
        log.info("%s", "=" * 58)
        log.info("Imputer: %s", imp_name)
        if force_recompute:
            tempos[imp_name] = {}
        elif imp_name not in tempos:
            tempos[imp_name] = {}

        for fold in tqdm(range(n_folds), desc=imp_name):
            ckpt = out_dir / f"{imp_name}_fold{fold}_train.parquet"
            if ckpt.exists() and not force_recompute:
                log.info("Fold %d already done, skipping", fold)
                continue

            tr_idx = fold_indices[fold]["train"]
            te_idx = fold_indices[fold]["test"]

            X_tr = X.iloc[tr_idx].copy()
            X_te = X.iloc[te_idx].copy()
            log.info("Fold %d start | train=%d test=%d", fold, len(tr_idx), len(te_idx))

            if imp_name == NO_IMPUTER_NAME:
                log.info("Imputer is %s - skipping imputation for fold %d", NO_IMPUTER_NAME, fold)
                t_fit, t_tr, t_te = 0.0, 0.0, 0.0
                # Directly save raw splits with NaNs
                pd.DataFrame(X_tr, columns=all_cols).to_parquet(
                    out_dir / f"{imp_name}_fold{fold}_train.parquet",
                    index=False,
                )
                pd.DataFrame(X_te, columns=all_cols).to_parquet(
                    out_dir / f"{imp_name}_fold{fold}_test.parquet",
                    index=False,
                )
                tempos[imp_name][str(fold)] = {
                    "time_fit": float(t_fit),
                    "time_transform_train": float(t_tr),
                    "time_transform_test": float(t_te),
                }
            else:
                try:
                    t0 = time.time()
                    imp_clone = clone(imputer)
                    imp_clone.fit(X_tr)
                    t_fit = time.time() - t0

                    t0 = time.time()
                    X_tr_imp = imp_clone.transform(X_tr)
                    t_tr = time.time() - t0

                    t0 = time.time()
                    X_te_imp = imp_clone.transform(X_te)
                    t_te = time.time() - t0
                except Exception as e:
                    log.error("Fold %d failed for %s: %s", fold, imp_name, e)
                    tempos[imp_name][str(fold)] = {"error": str(e)}
                    continue

                for arr, split_name in ((X_tr_imp, "train"), (X_te_imp, "test")):
                    n_nan = int(np.isnan(arr).sum())
                    if n_nan > 0:
                        log.warning("%s fold %d has %d residual NaN, filling with 0", split_name, fold, n_nan)

                X_tr_imp = np.nan_to_num(X_tr_imp, nan=0.0)
                X_te_imp = np.nan_to_num(X_te_imp, nan=0.0)

                pd.DataFrame(X_tr_imp, columns=all_cols).to_parquet(
                    out_dir / f"{imp_name}_fold{fold}_train.parquet",
                    index=False,
                )
                pd.DataFrame(X_te_imp, columns=all_cols).to_parquet(
                    out_dir / f"{imp_name}_fold{fold}_test.parquet",
                    index=False,
                )
                tempos[imp_name][str(fold)] = {
                    "time_fit": float(t_fit),
                    "time_transform_train": float(t_tr),
                    "time_transform_test": float(t_te),
                }
                del imp_clone, X_tr_imp, X_te_imp
                gc.collect()

        with open(tempos_path, "w", encoding="utf-8") as f:
            json.dump(tempos, f, indent=2)

    log.info("Imputation finished.")
    return tempos
