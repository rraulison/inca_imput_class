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

from src.categorical_encoding import encode_with_category_maps, fit_train_category_maps
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
BASE_IMPUTER_NAMES = ["Media", "Mediana", "kNN", "MICE_XGBoost", "MICE", "MissForest"]
SPLIT_SIGNATURE_FILE = "split_signature.json"
ENCODING_STATS_FILE = "fold_encoding_stats.json"


class CategoricalRounder(BaseEstimator, TransformerMixin):
    """Round imputed categorical values to nearest valid encoded label.

    Uses nearest-valid-value mapping instead of naive clip+round,
    so that only truly valid encoded labels are produced.
    Example: valid = [0, 1, 5], imputed = 2.7 → mapped to 1 (not 3).
    """

    def __init__(self, cat_indices, valid_values_list):
        self.cat_indices = cat_indices
        self.valid_values_list = valid_values_list

    def fit(self, X, y=None):
        self._is_fitted = True
        return self

    @staticmethod
    def _nearest_valid(values, valid):
        """Map each value to the nearest element in *valid*."""
        valid_sorted = np.sort(valid)
        # np.searchsorted finds insertion point in sorted array
        idx = np.searchsorted(valid_sorted, values, side="left")
        idx = np.clip(idx, 0, len(valid_sorted) - 1)
        # Compare with left and right neighbours to find true nearest
        left = np.clip(idx - 1, 0, len(valid_sorted) - 1)
        right = idx  # already clipped above
        dist_left = np.abs(values - valid_sorted[left])
        dist_right = np.abs(values - valid_sorted[right])
        nearest_idx = np.where(dist_left <= dist_right, left, right)
        return valid_sorted[nearest_idx]

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        for idx, valid in zip(self.cat_indices, self.valid_values_list):
            if len(valid) > 0:
                X[:, idx] = self._nearest_valid(X[:, idx], valid)
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


def run_imputation(config_path: str = "config/config.yaml", filter_imputers: list = None, cfg: dict = None) -> None:
    if cfg is None:
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

    cfg_imputers = cfg.get("imputation", {}).get("imputers") or (BASE_IMPUTER_NAMES + [NO_IMPUTER_NAME])

    imputer_names = []
    seen = set()
    for name in cfg_imputers:
        if name in seen:
            continue
        seen.add(name)
        if name in BASE_IMPUTER_NAMES or name == NO_IMPUTER_NAME:
            imputer_names.append(name)
        else:
            log.warning("Configured imputer '%s' is not available and will be skipped.", name)

    if filter_imputers:
        allowed = set(filter_imputers)
        imputer_names = [name for name in imputer_names if name in allowed]

    if not imputer_names:
        raise ValueError("No valid imputers selected.")

    tempos_path = res_dir / "tempos_imputacao.json"
    if tempos_path.exists():
        with open(tempos_path, "r", encoding="utf-8") as f:
            tempos = json.load(f)
    else:
        tempos = {}

    if force_recompute:
        tempos = {name: {} for name in imputer_names}
    else:
        for name in imputer_names:
            tempos.setdefault(name, {})

    encoding_stats_path = out_dir / ENCODING_STATS_FILE
    if encoding_stats_path.exists() and not force_recompute:
        with open(encoding_stats_path, "r", encoding="utf-8") as f:
            encoding_stats = json.load(f)
    else:
        encoding_stats = {}

    for fold in tqdm(range(n_folds), desc="Folds"):
        tr_idx = fold_indices[fold]["train"]
        te_idx = fold_indices[fold]["test"]

        X_tr_raw = X.iloc[tr_idx].copy()
        X_te_raw = X.iloc[te_idx].copy()

        cat_maps, valid_values_fold = fit_train_category_maps(X_tr_raw, cat_cols)
        X_tr_encoded, unseen_train = encode_with_category_maps(X_tr_raw, num_cols, cat_cols, cat_maps)
        X_te_encoded, unseen_test = encode_with_category_maps(X_te_raw, num_cols, cat_cols, cat_maps)

        unseen_total = int(sum(unseen_test.values()))
        if unseen_total > 0:
            log.warning("Fold %d has %d unseen categorical values in test split (mapped to NaN).", fold, unseen_total)

        encoding_stats[str(fold)] = {
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "n_categories_train": {col: int(len(valid_values_fold.get(col, []))) for col in cat_cols},
            "n_unseen_train": {col: int(unseen_train.get(col, 0)) for col in cat_cols},
            "n_unseen_test": {col: int(unseen_test.get(col, 0)) for col in cat_cols},
        }

        fold_imputers = _build_imputers(num_cols, cat_cols, valid_values_fold, cfg)

        for imp_name in imputer_names:
            ckpt = out_dir / f"{imp_name}_fold{fold}_train.parquet"
            if ckpt.exists() and not force_recompute:
                continue

            log.info("Fold %d start | imputer=%s | train=%d test=%d", fold, imp_name, len(tr_idx), len(te_idx))

            if imp_name == NO_IMPUTER_NAME:
                t_fit, t_tr, t_te = 0.0, 0.0, 0.0
                df_tr = X_tr_encoded.copy()
                df_te = X_te_encoded.copy()
                for df_tmp in (df_tr, df_te):
                    for c in df_tmp.columns:
                        df_tmp[c] = pd.to_numeric(df_tmp[c], errors="coerce").astype(np.float32)

                df_tr.to_parquet(out_dir / f"{imp_name}_fold{fold}_train.parquet", index=False)
                df_te.to_parquet(out_dir / f"{imp_name}_fold{fold}_test.parquet", index=False)
                tempos[imp_name][str(fold)] = {
                    "time_fit": float(t_fit),
                    "time_transform_train": float(t_tr),
                    "time_transform_test": float(t_te),
                }
                continue

            if imp_name not in fold_imputers:
                log.error("Imputer '%s' is not available in this run and will be skipped.", imp_name)
                tempos[imp_name][str(fold)] = {"error": f"imputer_not_available:{imp_name}"}
                continue

            try:
                t0 = time.time()
                imp_clone = clone(fold_imputers[imp_name])
                imp_clone.fit(X_tr_encoded)
                t_fit = time.time() - t0

                t0 = time.time()
                X_tr_imp = imp_clone.transform(X_tr_encoded)
                t_tr = time.time() - t0

                t0 = time.time()
                X_te_imp = imp_clone.transform(X_te_encoded)
                t_te = time.time() - t0
            except Exception as e:
                log.error("Fold %d failed for %s: %s", fold, imp_name, e, exc_info=True)
                tempos[imp_name][str(fold)] = {"error": str(e)}
                continue

            for arr, split_name in ((X_tr_imp, "train"), (X_te_imp, "test")):
                n_nan = int(np.isnan(arr).sum())
                if n_nan > 0:
                    log.warning("%s fold %d has %d residual NaN, filling with 0", split_name, fold, n_nan)

            X_tr_imp = np.nan_to_num(X_tr_imp, nan=0.0)
            X_te_imp = np.nan_to_num(X_te_imp, nan=0.0)

            pd.DataFrame(X_tr_imp, columns=all_cols).to_parquet(out_dir / f"{imp_name}_fold{fold}_train.parquet", index=False)
            pd.DataFrame(X_te_imp, columns=all_cols).to_parquet(out_dir / f"{imp_name}_fold{fold}_test.parquet", index=False)
            tempos[imp_name][str(fold)] = {
                "time_fit": float(t_fit),
                "time_transform_train": float(t_tr),
                "time_transform_test": float(t_te),
            }

            del imp_clone, X_tr_imp, X_te_imp
            gc.collect()

        with open(tempos_path, "w", encoding="utf-8") as f:
            json.dump(tempos, f, indent=2)

        with open(encoding_stats_path, "w", encoding="utf-8") as f:
            json.dump(encoding_stats, f, indent=2)

        del X_tr_raw, X_te_raw, X_tr_encoded, X_te_encoded, fold_imputers
        gc.collect()

    log.info("Imputation finished.")
    return tempos
