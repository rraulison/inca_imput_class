"""
Step 3 - Classification.
Input: data/imputed/ (Step 2 output).
Output: results/raw/all_results.csv and all_results_detailed.json.

Scaling is fit only on train splits.
Hyperparameter tuning runs on inner CV splits.
"""

import gc
import inspect
import json
import logging
import time
import traceback
import warnings

# Suppress RAPIDS cuML SVC probability warnings
warnings.filterwarnings("ignore", message="Random state is currently ignored by probabilistic SVC")
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    get_scorer,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm
from xgboost import XGBClassifier

from src.config_loader import load_config

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover - optional dependency
    CatBoostClassifier = None

try:
    import cupy as cp
except Exception:  # pragma: no cover - optional dependency
    cp = None

try:
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
except Exception:  # pragma: no cover - optional dependency
    CuMLRandomForestClassifier = None

try:
    from cuml.neural_network import MLPClassifier as CuMLMLPClassifier
except Exception:  # pragma: no cover - optional dependency
    CuMLMLPClassifier = None

try:
    from cuml.svm import SVC as CuMLSVC
except Exception:  # pragma: no cover - optional dependency
    CuMLSVC = None

log = logging.getLogger(__name__)

MAIN_IMPUTER_NAMES = ["Media", "Mediana", "kNN", "MICE_XGBoost", "MICE", "MissForest"]
NO_IMPUTER_NAME = "NoImpute"
RAW_CATBOOST_IMPUTER = "RawSemEncoding"
CLF_ORDER = ["XGBoost", "CatBoost", "cuML_RF", "cuML_SVM", "cuML_MLP"]
CHECKPOINT_SCHEMA_VERSION = 2

FAST_FIXED_PARAMS = {
    "XGBoost": {
        "colsample_bytree": 0.6,
        "gamma": 0.3,
        "learning_rate": 0.05,
        "max_depth": 10,
        "min_child_weight": 1,
        "n_estimators": 100,
        "reg_alpha": 0,
        "reg_lambda": 1,
        "subsample": 1.0,
    },
    "CatBoost": {
        "depth": 6,
        "iterations": 600,
        "l2_leaf_reg": 3,
        "learning_rate": 0.1,
        "subsample": 0.7,
    },
    "cuML_RF": {
        "max_depth": 8,
        "max_features": "sqrt",
        "n_bins": 128,
        "n_estimators": 400,
    },
    "cuML_SVM": {
        "C": 10,
        "gamma": "auto",
        "kernel": "rbf",
    },
}

HYBRID_PARAM_SPACE = {
    "XGBoost": {
        "n_estimators": [100, 200],
        "max_depth": [7, 10],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.6, 0.8],
        "min_child_weight": [1, 3],
        "gamma": [0.1, 0.3],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [1, 2],
    },
    "CatBoost": {
        "iterations": [400, 600],
        "depth": [6, 8],
        "learning_rate": [0.05, 0.1],
        "l2_leaf_reg": [1, 3],
        "subsample": [0.7, 0.85],
    },
    "cuML_RF": {
        "n_estimators": [200, 400],
        "max_depth": [8, 12],
        "max_features": ["sqrt", "log2"],
        "n_bins": [64, 128],
    },
    "cuML_SVM": {
        "C": [1, 10],
        "kernel": ["rbf"],
        "gamma": ["scale", "auto"],
    },
}

DEFAULT_PARAM_SPACE = {
    **HYBRID_PARAM_SPACE,
    "cuML_MLP": {
        "hidden_layer_sizes": [(256, 128), (128, 64)],
        "alpha": [0.0001, 0.001],
        "learning_rate_init": [0.001, 0.01],
        "batch_size": [256, 512],
        "max_iter": [200, 300],
    },
}


def _resolve_classifier_params(cfg):
    cfg_params = cfg.get("classification", {}).get("params", {}) or {}
    resolved = {}
    for name in CLF_ORDER:
        param_space = cfg_params.get(name)
        if isinstance(param_space, dict):
            resolved[name] = param_space
        else:
            resolved[name] = DEFAULT_PARAM_SPACE.get(name, {})
            log.warning("classification.params.%s missing; using built-in defaults.", name)
    return resolved


def _n_iter_for_space(base_n_iter, param_space):
    return int(base_n_iter) if param_space else 1


def _to_numpy(x):
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas().to_numpy()
    return np.asarray(x)


def _to_gpu(x):
    if cp is None:
        raise RuntimeError("cupy is required for cuML GPU estimators.")
    return cp.asarray(_to_numpy(x), dtype=cp.float32)


def _one_hot_proba(y_pred, classes):
    y_pred = _to_numpy(y_pred).astype(int)
    proba = np.zeros((len(y_pred), len(classes)), dtype=float)
    class_pos = {int(c): i for i, c in enumerate(classes)}
    for i, y in enumerate(y_pred):
        if int(y) in class_pos:
            proba[i, class_pos[int(y)]] = 1.0
    return proba


def _softmax(x):
    x = _to_numpy(x)
    if x.ndim == 1:
        x = np.column_stack([-x, x])
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    den = np.sum(exp_x, axis=1, keepdims=True)
    den[den == 0] = 1.0
    return exp_x / den


def _filter_supported_params(model_cls, params):
    try:
        sig = inspect.signature(model_cls.__init__)
        valid = set(sig.parameters.keys())
        return {k: v for k, v in params.items() if k in valid}
    except Exception:
        return params


class CuMLRFWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=400, max_depth=16, max_features="sqrt", n_bins=128, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_bins = n_bins
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        if CuMLRandomForestClassifier is None or cp is None:
            raise RuntimeError("cuML RandomForestClassifier is unavailable.")
        params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "n_bins": self.n_bins,
            "random_state": self.random_state,
        }
        params = _filter_supported_params(CuMLRandomForestClassifier, params)
        self.model_ = CuMLRandomForestClassifier(**params)
        X_gpu = _to_gpu(X)
        y_gpu = cp.asarray(_to_numpy(y), dtype=cp.int32)
        self.model_.fit(X_gpu, y_gpu)
        self.classes_ = np.sort(np.unique(_to_numpy(y).astype(int)))
        return self

    def predict(self, X):
        pred = self.model_.predict(_to_gpu(X))
        return _to_numpy(pred).astype(int)

    def predict_proba(self, X):
        if hasattr(self.model_, "predict_proba"):
            return _to_numpy(self.model_.predict_proba(_to_gpu(X)))
        return _one_hot_proba(self.predict(X), self.classes_)


class CuMLSVMWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        gamma="scale",
        probability=True,
        cache_size=2000,
        max_iter=-1,
        tol=1e-3,
        random_state=42,
    ):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.probability = probability
        self.cache_size = cache_size
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        if CuMLSVC is None or cp is None:
            raise RuntimeError("cuML SVC is unavailable.")
        params = {
            "C": self.C,
            "kernel": self.kernel,
            "gamma": self.gamma,
            "probability": self.probability,
            "cache_size": self.cache_size,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "random_state": self.random_state,
        }
        params = _filter_supported_params(CuMLSVC, params)
        self.model_ = CuMLSVC(**params)
        X_gpu = _to_gpu(X)
        y_gpu = cp.asarray(_to_numpy(y), dtype=cp.int32)
        self.model_.fit(X_gpu, y_gpu)
        self.classes_ = np.sort(np.unique(_to_numpy(y).astype(int)))
        return self

    def predict(self, X):
        pred = self.model_.predict(_to_gpu(X))
        return _to_numpy(pred).astype(int)

    def predict_proba(self, X):
        X_gpu = _to_gpu(X)
        # 1. Prefer native predict_proba (Platt scaling when probability=True).
        if hasattr(self.model_, "predict_proba"):
            try:
                proba = _to_numpy(self.model_.predict_proba(X_gpu))
                if proba.ndim == 2 and proba.shape[1] == len(self.classes_):
                    return proba
            except Exception:
                pass

        # 2. decision_function fallback: cuML SVC uses OvO, producing
        #    n_classes*(n_classes-1)/2 scores.  Applying softmax directly
        #    to OvO scores is wrong.  Use OvO vote-counting instead.
        if hasattr(self.model_, "decision_function"):
            try:
                scores = _to_numpy(self.model_.decision_function(X_gpu))
                n_classes = len(self.classes_)
                if scores.ndim == 2 and scores.shape[1] == n_classes:
                    # Rare case: output already has n_classes columns
                    return _softmax(scores)
                # OvO vote counting
                proba = self._ovo_vote_proba(scores, n_classes)
                return proba
            except Exception:
                pass

        # 3. Last resort: one-hot from hard predictions.
        return _one_hot_proba(self.predict(X), self.classes_)

    @staticmethod
    def _ovo_vote_proba(scores, n_classes):
        """Convert OvO decision scores to class probabilities via vote counting."""
        scores = np.asarray(scores)
        n_samples = scores.shape[0]
        votes = np.zeros((n_samples, n_classes), dtype=np.float64)
        col = 0
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                if col >= scores.shape[1]:
                    break
                # Positive score → class i wins, negative → class j wins
                win_i = (scores[:, col] > 0).astype(float)
                votes[:, i] += win_i
                votes[:, j] += 1.0 - win_i
                col += 1
        # Normalize to probabilities
        row_sums = votes.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        return votes / row_sums


class CuMLMLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes=(256, 128),
        alpha=0.0001,
        learning_rate_init=0.001,
        batch_size=512,
        max_iter=200,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        if CuMLMLPClassifier is None or cp is None:
            raise RuntimeError("cuML MLPClassifier is unavailable.")

        hidden = tuple(self.hidden_layer_sizes) if isinstance(self.hidden_layer_sizes, list) else self.hidden_layer_sizes
        params = {
            "hidden_layer_sizes": hidden,
            "alpha": self.alpha,
            "learning_rate_init": self.learning_rate_init,
            "batch_size": self.batch_size,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }
        params = _filter_supported_params(CuMLMLPClassifier, params)

        self.model_ = CuMLMLPClassifier(**params)
        X_gpu = _to_gpu(X)
        y_gpu = cp.asarray(_to_numpy(y), dtype=cp.int32)
        self.model_.fit(X_gpu, y_gpu)
        self.classes_ = np.sort(np.unique(_to_numpy(y).astype(int)))
        return self

    def predict(self, X):
        pred = self.model_.predict(_to_gpu(X))
        return _to_numpy(pred).astype(int)

    def predict_proba(self, X):
        if hasattr(self.model_, "predict_proba"):
            try:
                return _to_numpy(self.model_.predict_proba(_to_gpu(X)))
            except Exception:
                pass
        return _one_hot_proba(self.predict(X), self.classes_)


def _dependency_status():
    return {
        "catboost": CatBoostClassifier is not None,
        "cupy": cp is not None,
        "cuml_rf": CuMLRandomForestClassifier is not None,
        "cuml_svm": CuMLSVC is not None,
        "cuml_mlp": CuMLMLPClassifier is not None,
    }


def _build_classifiers(cfg):
    seed = cfg["experiment"]["random_seed"]
    use_gpu = cfg["hardware"]["use_gpu"]
    gpu_device = cfg["hardware"].get("gpu_device", "cuda")
    gpu_id = cfg["hardware"].get("gpu_id", 0)

    params = _resolve_classifier_params(cfg)
    n_iter = cfg["classification"]["tuning"]["n_iter"]
    n_iter_svm = cfg["classification"]["tuning"].get("n_iter_svm", n_iter)
    n_iter_mlp = cfg["classification"]["tuning"].get("n_iter_mlp", n_iter)

    deps = _dependency_status()
    use_cuda = use_gpu and gpu_device.lower() == "cuda"

    cat_model = None
    cat_available = deps["catboost"]
    if cat_available:
        cat_kwargs = {
            "random_seed": seed,
            "verbose": False,
            "loss_function": "MultiClass",
            "allow_writing_files": False,
            "thread_count": -1,
            "bootstrap_type": "Bernoulli",
        }
        if use_cuda:
            cat_kwargs["task_type"] = "GPU"
            cat_kwargs["devices"] = str(gpu_id)
        else:
            cat_kwargs["task_type"] = "CPU"
        cat_model = CatBoostClassifier(**cat_kwargs)

    return OrderedDict(
        {
            "XGBoost": {
                "model": XGBClassifier(
                    random_state=seed,
                    n_jobs=1,
                    device=gpu_device if use_gpu else "cpu",
                    tree_method="hist",
                    eval_metric="mlogloss",
                    verbosity=0,
                ),
                "params": params["XGBoost"],
                "needs_scaling": False,
                "n_iter": _n_iter_for_space(n_iter, params["XGBoost"]),
                "search_n_jobs": 1,
                "engine": "sklearn",
                "available": True,
            },
            "CatBoost": {
                "model": cat_model,
                "params": params["CatBoost"],
                "needs_scaling": False,
                "n_iter": _n_iter_for_space(n_iter, params["CatBoost"]),
                "search_n_jobs": 1,
                "engine": "sklearn",
                "available": cat_available,
                "error": "catboost is not installed." if not cat_available else None,
            },
            "cuML_RF": {
                "model": CuMLRFWrapper(random_state=seed),
                "params": params["cuML_RF"],
                "needs_scaling": False,
                "n_iter": _n_iter_for_space(n_iter, params["cuML_RF"]),
                "search_n_jobs": 1,
                "engine": "cuml",
                "available": deps["cupy"] and deps["cuml_rf"],
                "error": "cupy/cuml RandomForestClassifier unavailable.",
            },
            "cuML_SVM": {
                "model": CuMLSVMWrapper(
                    random_state=seed,
                    probability=True,
                    cache_size=cfg["hardware"]["svm_cache_mb"],
                ),
                "params": params["cuML_SVM"],
                "needs_scaling": True,
                "n_iter": _n_iter_for_space(n_iter_svm, params["cuML_SVM"]),
                "search_n_jobs": 1,
                "engine": "cuml",
                "available": deps["cupy"] and deps["cuml_svm"],
                "error": "cupy/cuml SVC unavailable.",
            },
            "cuML_MLP": {
                "model": CuMLMLPWrapper(random_state=seed),
                "params": params["cuML_MLP"],
                "needs_scaling": True,
                "n_iter": _n_iter_for_space(n_iter_mlp, params["cuML_MLP"]),
                "search_n_jobs": 1,
                "engine": "cuml",
                "available": deps["cupy"] and deps["cuml_mlp"],
                "error": "cupy/cuml MLPClassifier unavailable.",
            },
        }
    )


def _compute_metrics(y_true, y_pred, y_prob, classes):
    y_true = _to_numpy(y_true).astype(int)
    y_pred = _to_numpy(y_pred).astype(int)
    y_prob = _to_numpy(y_prob)

    metrics = {
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


def _serialize(obj):
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


from src.stats_utils import fmt_time as _fmt_time


def _normalize_runtime_mode(mode):
    mode = str(mode or "default").lower()
    return mode if mode in {"default", "hybrid", "fast"} else "default"


def _runtime_setup(cfg, classifiers):
    runtime_cfg = cfg.get("classification", {}).get("runtime", {})
    mode = _normalize_runtime_mode(runtime_cfg.get("mode", "default"))
    tune_max_samples = runtime_cfg.get("tune_max_samples")
    tune_max_samples = int(tune_max_samples) if tune_max_samples else None

    if mode == "hybrid" and not tune_max_samples:
        tune_max_samples = 20000

    base_inner = int(cfg["cv"]["n_inner_folds"])
    if mode in {"hybrid", "fast"}:
        inner_folds = 2
    else:
        inner_folds = max(2, base_inner)

    for name, clf_cfg in classifiers.items():
        clf_cfg["fixed_params"] = {}
        if mode == "hybrid":
            if name in HYBRID_PARAM_SPACE:
                clf_cfg["params"] = HYBRID_PARAM_SPACE[name]
            if name == "XGBoost":
                clf_cfg["n_iter"] = min(int(clf_cfg["n_iter"]), 8)
            elif name == "CatBoost":
                clf_cfg["n_iter"] = min(int(clf_cfg["n_iter"]), 6)
            else:
                clf_cfg["n_iter"] = min(int(clf_cfg["n_iter"]), 4)
        elif mode == "fast":
            clf_cfg["n_iter"] = 1
            clf_cfg["fixed_params"] = FAST_FIXED_PARAMS.get(name, {})

    return mode, tune_max_samples, inner_folds


def _ordered_config_classifiers(cfg):
    configured = cfg.get("classification", {}).get("classifiers", CLF_ORDER)
    front = [name for name in CLF_ORDER if name in configured]
    tail = [name for name in configured if name not in front]
    return front + tail


def _task_key(runtime_mode, imputer, fold, classifier):
    return f"{runtime_mode}__{imputer}__{int(fold)}__{classifier}"


def _planned_task_keys(runtime_mode, tasks):
    return {
        _task_key(runtime_mode, task["imputer"], task["fold"], task["classifier"])
        for task in tasks
    }


def _result_key_from_row(row, runtime_mode):
    if not isinstance(row, dict):
        return None
    imputer = row.get("imputer")
    classifier = row.get("classifier")
    fold = row.get("fold")
    if imputer is None or classifier is None or fold is None:
        return None
    try:
        fold = int(fold)
    except (TypeError, ValueError):
        return None
    mode = _normalize_runtime_mode(row.get("runtime_mode", runtime_mode))
    return _task_key(mode, imputer, fold, classifier)


def _prune_checkpoint_for_tasks(ckpt, runtime_mode, planned_keys):
    mode_prefix = f"{runtime_mode}__"
    raw_completed = {str(key) for key in ckpt.get("completed", []) if str(key).startswith(mode_prefix)}
    raw_results = [
        row
        for row in ckpt.get("results", [])
        if _normalize_runtime_mode(row.get("runtime_mode", runtime_mode)) == runtime_mode
    ]

    results_by_key = OrderedDict()
    for row in raw_results:
        key = _result_key_from_row(row, runtime_mode)
        if key is None or key not in planned_keys:
            continue
        results_by_key[key] = row

    completed = set(results_by_key.keys())
    dropped_completed = len(raw_completed - completed)
    dropped_results = len(raw_results) - len(results_by_key)
    return completed, list(results_by_key.values()), dropped_completed, dropped_results


def _resolve_classes(meta, imp_dir):
    target_map = meta.get("target_mapping", {}) or {}
    classes_from_map = []
    if isinstance(target_map, dict):
        for encoded in target_map.values():
            try:
                classes_from_map.append(int(encoded))
            except (TypeError, ValueError):
                continue
    if classes_from_map:
        classes = np.array(sorted(set(classes_from_map)), dtype=int)
        return classes, "metadata.target_mapping"

    n_classes_raw = meta.get("n_classes")
    try:
        n_classes = int(n_classes_raw)
    except (TypeError, ValueError):
        n_classes = 0
    if n_classes > 0:
        return np.arange(n_classes, dtype=int), "metadata.n_classes"

    y0 = pd.read_parquet(imp_dir / "y_fold0_train.parquet")["target"]
    return np.sort(y0.astype(int).unique()), "y_fold0_train"


def _take_rows(X, idx):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    return X[idx]


def _fit_with_optional_weights(model, X, y, sample_weight=None, fit_kwargs=None):
    kwargs = dict(fit_kwargs or {})
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    model.fit(X, y, **kwargs)


def _safe_inner_splits(y, requested):
    y = _to_numpy(y).astype(int)
    _, counts = np.unique(y, return_counts=True)
    if len(counts) == 0 or int(counts.min()) < 2:
        return None
    return max(2, min(int(requested), int(counts.min())))


def _stratified_tuning_subset(X, y, sample_weight, max_samples, seed):
    if not max_samples or len(y) <= max_samples:
        return X, y, sample_weight, False

    idx = np.arange(len(y))
    try:
        sub_idx, _ = train_test_split(
            idx,
            train_size=max_samples,
            stratify=y,
            random_state=seed,
        )
    except Exception:
        rng = np.random.RandomState(seed)
        sub_idx = rng.choice(idx, size=max_samples, replace=False)

    sub_idx = np.sort(sub_idx)
    X_sub = _take_rows(X, sub_idx)
    y_sub = y[sub_idx]
    w_sub = None if sample_weight is None else np.asarray(sample_weight)[sub_idx]
    return X_sub, y_sub, w_sub, True


def _manual_random_search(estimator, param_distributions, n_iter, cv, scoring, X, y, seed, refit_X=None, refit_y=None, sample_weight=None, fit_kwargs=None):
    fit_kwargs = dict(fit_kwargs or {})
    search_estimator = clone(estimator)
    if hasattr(search_estimator, "probability") and hasattr(search_estimator, "set_params"):
        # Optimization: SVC probability=True is slow (internal CV).
        # If scoring (e.g. f1) doesn't explicitly need proba, disable it during search.
        # inner_cv will call predict(), not predict_proba().
        # We assume 'scoring' string implies need. "roc_auc" needs proba.
        if isinstance(scoring, str) and "auc" not in scoring.lower() and search_estimator.probability:
            try:
                search_estimator.set_params(probability=False)
                # log.info("Disabled probability for tuning (scoring=%s)", scoring)
            except Exception:
                pass

    scorer = get_scorer(scoring)
    sampled = list(ParameterSampler(param_distributions, n_iter=n_iter, random_state=seed))
    if not sampled:
        sampled = [{}]

    best_score = -np.inf
    best_params = {}

    for params in sampled:
        fold_scores = []
        for tr_idx, va_idx in cv.split(X, y):
            est = clone(search_estimator)
            est.set_params(**params)
            
            X_tr_fold = _take_rows(X, tr_idx)
            y_tr_fold = _take_rows(y, tr_idx)
            X_va_fold = _take_rows(X, va_idx)
            y_va_fold = _take_rows(y, va_idx)

            # Build fold fit kwargs: subsample sample_weight and merge fit_kwargs
            fold_fit_kwargs = dict(fit_kwargs)
            if sample_weight is not None:
                fold_fit_kwargs["sample_weight"] = np.asarray(sample_weight)[tr_idx]

            _fit_with_optional_weights(est, X_tr_fold, y_tr_fold, fit_kwargs=fold_fit_kwargs)
            fold_scores.append(float(scorer(est, X_va_fold, y_va_fold)))
            del est
            gc.collect()

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    best_estimator = clone(estimator)
    best_estimator.set_params(**best_params)
    fit_X = X if refit_X is None else refit_X
    fit_y = y if refit_y is None else refit_y

    # Use the full sample_weight for refitting if not using a subset
    refit_kwargs = dict(fit_kwargs)
    if sample_weight is not None and refit_X is None:
        refit_kwargs["sample_weight"] = sample_weight
    _fit_with_optional_weights(best_estimator, fit_X, fit_y, fit_kwargs=refit_kwargs)

    return best_estimator, best_params, best_score


def _save_classification_checkpoint(ckpt_path, completed, all_results):
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "completed": sorted(completed),
        "results": _serialize(all_results),
    }
    with open(ckpt_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def _evaluate_combination(
    imp_name,
    fold,
    clf_name,
    clf_cfg,
    X_tr,
    X_te,
    y_tr,
    y_te,
    classes,
    *,
    seed,
    runtime_mode,
    tune_max_samples,
    n_inner,
    scoring,
    svm_max,
    t_imp_fit,
    t_imp_transform,
    fit_kwargs=None,
):
    fit_kwargs = dict(fit_kwargs or {})
    try:
        if clf_cfg["needs_scaling"]:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_tr)
            Xte = scaler.transform(X_te)
        else:
            if isinstance(X_tr, pd.DataFrame):
                Xtr = X_tr.copy()
                Xte = X_te.copy()
            else:
                Xtr = np.asarray(X_tr)
                Xte = np.asarray(X_te)

        ytr = _to_numpy(y_tr).astype(int)
        yte_np = _to_numpy(y_te).astype(int)

        if clf_name == "cuML_SVM" and svm_max and len(Xtr) > svm_max:
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(Xtr), svm_max, replace=False)
            Xtr = _take_rows(Xtr, idx)
            ytr = ytr[idx]
            log.warning("cuML_SVM subsampled train set: %d -> %d", len(y_tr), svm_max)

        sample_weight = None
        if clf_name in {"XGBoost", "CatBoost"}:
            sample_weight = compute_sample_weight("balanced", ytr)

        if runtime_mode == "fast":
            best_params = dict(clf_cfg.get("fixed_params", {}))
            best_model = clone(clf_cfg["model"])
            if best_params:
                best_model.set_params(**best_params)
            _fit_with_optional_weights(best_model, Xtr, ytr, sample_weight=sample_weight, fit_kwargs=fit_kwargs)
            best_score = np.nan
            t_tuning = 0.0
        else:
            X_tune, y_tune, w_tune, used_subset = Xtr, ytr, sample_weight, False
            if runtime_mode == "hybrid":
                tune_seed = seed + fold * 101 + sum(ord(ch) for ch in clf_name)
                X_tune, y_tune, w_tune, used_subset = _stratified_tuning_subset(
                    Xtr,
                    ytr,
                    sample_weight,
                    tune_max_samples,
                    tune_seed,
                )
                if used_subset:
                    log.info(
                        "%s fold%d %s: tuning subset %d/%d samples",
                        imp_name,
                        fold,
                        clf_name,
                        len(y_tune),
                        len(ytr),
                    )

            n_inner_eff = _safe_inner_splits(y_tune, n_inner)
            if n_inner_eff is None:
                best_params = dict(clf_cfg.get("fixed_params", {}))
                best_model = clone(clf_cfg["model"])
                if best_params:
                    best_model.set_params(**best_params)
                _fit_with_optional_weights(best_model, Xtr, ytr, sample_weight=sample_weight, fit_kwargs=fit_kwargs)
                best_score = np.nan
                t_tuning = 0.0
            else:
                inner_cv = StratifiedKFold(n_splits=n_inner_eff, shuffle=True, random_state=seed + fold)
                t0 = time.time()
                if clf_cfg["engine"] == "cuml":
                    best_model, best_params, best_score = _manual_random_search(
                        estimator=clf_cfg["model"],
                        param_distributions=clf_cfg["params"],
                        n_iter=clf_cfg["n_iter"],
                        cv=inner_cv,
                        scoring=scoring,
                        X=X_tune,
                        y=y_tune,
                        seed=seed + fold,
                        refit_X=Xtr if used_subset else None,
                        refit_y=ytr if used_subset else None,
                        sample_weight=w_tune,
                        fit_kwargs=fit_kwargs,
                    )
                else:
                    search = RandomizedSearchCV(
                        estimator=clone(clf_cfg["model"]),
                        param_distributions=clf_cfg["params"],
                        n_iter=clf_cfg["n_iter"],
                        cv=inner_cv,
                        scoring=scoring,
                        random_state=seed + fold,
                        n_jobs=clf_cfg["search_n_jobs"],
                        verbose=0,
                        error_score="raise",
                        refit=True,
                    )
                    tune_fit_kwargs = dict(fit_kwargs)
                    if w_tune is not None:
                        tune_fit_kwargs["sample_weight"] = w_tune
                    search.fit(X_tune, y_tune, **tune_fit_kwargs)

                    best_params = search.best_params_
                    best_score = search.best_score_
                    if used_subset:
                        best_model = clone(clf_cfg["model"])
                        best_model.set_params(**best_params)
                        _fit_with_optional_weights(
                            best_model,
                            Xtr,
                            ytr,
                            sample_weight=sample_weight,
                            fit_kwargs=fit_kwargs,
                        )
                    else:
                        best_model = search.best_estimator_
                    del search

                t_tuning = time.time() - t0

        t0 = time.time()
        y_pred = _to_numpy(best_model.predict(Xte)).astype(int)
        if hasattr(best_model, "predict_proba"):
            y_prob = _to_numpy(best_model.predict_proba(Xte))
        else:
            y_prob = _one_hot_proba(y_pred, classes)
        t_pred = time.time() - t0

        if y_prob.ndim == 1:
            y_prob = np.column_stack([1 - y_prob, y_prob])
        if y_prob.shape[1] != len(classes):
            y_prob = _one_hot_proba(y_pred, classes)

        metrics = _compute_metrics(yte_np, y_pred, y_prob, classes)

        result = {
            "fold": fold,
            "imputer": imp_name,
            "classifier": clf_name,
            "accuracy": metrics["accuracy"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
            "auc_weighted": metrics["auc_weighted"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "auc_macro": metrics["auc_macro"],
            "time_imputation_fit": t_imp_fit,
            "time_imputation_transform": t_imp_transform,
            "time_tuning": t_tuning,
            "time_prediction": t_pred,
            "time_total": t_imp_fit + t_imp_transform + t_tuning + t_pred,
            "best_params": str(best_params),
            "best_inner_score": float(best_score),
            "classification_report": metrics["classification_report"],
            "confusion_matrix": metrics["confusion_matrix"],
        }

        log.info(
            "%s fold%d %s: F1=%.4f AUC=%.4f (%s)",
            imp_name,
            fold,
            clf_name,
            metrics["f1_weighted"],
            metrics["auc_weighted"] if not np.isnan(metrics["auc_weighted"]) else -1,
            _fmt_time(t_tuning),
        )
        return result

    except Exception as e:
        log.error("%s fold%d %s failed: %s", imp_name, fold, clf_name, e)
        traceback.print_exc()
        return {
            "fold": fold,
            "imputer": imp_name,
            "classifier": clf_name,
            "error": str(e),
        }


def run_classification(
    config_path: str = "config/config.yaml",
    filter_imputers: list = None,
    filter_classifiers: list = None,
    cfg: dict = None,
) -> None:
    if cfg is None:
        cfg = load_config(config_path)
    seed = cfg["experiment"]["random_seed"]
    n_outer = cfg["cv"]["n_outer_folds"]
    scoring = cfg["classification"]["tuning"]["scoring"]
    svm_max_raw = cfg["classification"].get("svm_max_train_samples")
    svm_max = 20000 if svm_max_raw is None else int(svm_max_raw)

    imp_dir = Path(cfg["paths"]["imputed_data"])
    proc_dir = Path(cfg["paths"]["processed_data"])
    res_dir = Path(cfg["paths"]["results_raw"])
    res_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    with open(Path(cfg["paths"]["results_tables"]) / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    _ = meta.get("n_classes", None)
    cat_cols = meta.get("cat_cols", [])

    tempos_imp = {}
    tempos_path = res_dir / "tempos_imputacao.json"
    if tempos_path.exists():
        with open(tempos_path, "r", encoding="utf-8") as f:
            tempos_imp = json.load(f)

    requested_imputers = filter_imputers or MAIN_IMPUTER_NAMES
    requested_main_imputers = [name for name in requested_imputers if name in MAIN_IMPUTER_NAMES]
    available_main_imputers = [
        name
        for name in requested_main_imputers
        if all((imp_dir / f"{name}_fold{f}_train.parquet").exists() for f in range(n_outer))
    ]
    log.info("Main imputers available: %s", available_main_imputers)

    classifiers = _build_classifiers(cfg)
    runtime_mode, tune_max_samples, n_inner = _runtime_setup(cfg, classifiers)
    log.info(
        "Classification mode=%s | inner_folds=%d | tune_max_samples=%s",
        runtime_mode,
        n_inner,
        tune_max_samples,
    )

    requested_clf = filter_classifiers or _ordered_config_classifiers(cfg)
    requested_clf = [name for name in requested_clf if name in classifiers]
    main_clf_names = [name for name in requested_clf if name != "CatBoost"]

    baseline_cfg = cfg.get("classification", {}).get("baselines", {})
    use_xgb_baseline = bool(baseline_cfg.get("xgboost_native_missing", True))
    use_catboost_raw = bool(baseline_cfg.get("catboost_raw_no_encoding", True))

    filter_imputers_set = set(filter_imputers or [])
    filter_classifiers_set = set(filter_classifiers or [])

    noimpute_available = all((imp_dir / f"{NO_IMPUTER_NAME}_fold{f}_train.parquet").exists() for f in range(n_outer))
    run_xgb_baseline = (
        use_xgb_baseline
        and noimpute_available
        and ("XGBoost" in classifiers)
        and (not filter_imputers or NO_IMPUTER_NAME in filter_imputers_set)
        and (not filter_classifiers or "XGBoost" in filter_classifiers_set)
    )
    if use_xgb_baseline and not noimpute_available:
        log.warning(
            "Baseline '%s + XGBoost' requested but NoImpute folds are missing in %s.",
            NO_IMPUTER_NAME,
            imp_dir,
        )

    raw_path = proc_dir / "X_raw_prepared.parquet"
    fold_indices_path = imp_dir / "fold_indices.json"
    run_catboost_raw = (
        use_catboost_raw
        and raw_path.exists()
        and fold_indices_path.exists()
        and (not filter_imputers or RAW_CATBOOST_IMPUTER in filter_imputers_set)
        and (not filter_classifiers or "CatBoost" in filter_classifiers_set)
    )
    if use_catboost_raw and (not raw_path.exists() or not fold_indices_path.exists()):
        log.warning(
            "Baseline '%s + CatBoost' requested but required artifacts are missing: %s, %s",
            RAW_CATBOOST_IMPUTER,
            raw_path,
            fold_indices_path,
        )

    tasks = []
    for imp_name in available_main_imputers:
        for fold in range(n_outer):
            for clf_name in main_clf_names:
                tasks.append({"kind": "main", "imputer": imp_name, "fold": fold, "classifier": clf_name})

    if run_xgb_baseline:
        for fold in range(n_outer):
            tasks.append(
                {
                    "kind": "baseline_xgb_native_missing",
                    "imputer": NO_IMPUTER_NAME,
                    "fold": fold,
                    "classifier": "XGBoost",
                }
            )

    if run_catboost_raw:
        for fold in range(n_outer):
            tasks.append(
                {
                    "kind": "baseline_catboost_raw",
                    "imputer": RAW_CATBOOST_IMPUTER,
                    "fold": fold,
                    "classifier": "CatBoost",
                }
            )

    log.info("Planned tasks: %d", len(tasks))
    planned_keys = _planned_task_keys(runtime_mode, tasks)

    ckpt_suffix = "" if runtime_mode == "default" else f"_{runtime_mode}"
    ckpt_path = res_dir / f"checkpoint_classification{ckpt_suffix}.json"
    out_csv = res_dir / f"all_results{ckpt_suffix}.csv"
    out_json = res_dir / f"all_results_detailed{ckpt_suffix}.json"

    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        if ckpt.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            log.warning(
                "Checkpoint schema mismatch (%s). Ignoring old checkpoint and starting fresh.",
                ckpt.get("schema_version"),
            )
            completed = set()
            all_results = []
        else:
            completed, all_results, dropped_completed, dropped_results = _prune_checkpoint_for_tasks(
                ckpt,
                runtime_mode,
                planned_keys,
            )
            if dropped_completed or dropped_results:
                log.info(
                    "Checkpoint pruned to current plan: dropped %d completed keys and %d result rows.",
                    dropped_completed,
                    dropped_results,
                )
            log.info(
                "Checkpoint loaded (%s): %d completed combinations (%d planned).",
                runtime_mode,
                len(completed),
                len(planned_keys),
            )
    else:
        completed = set()
        all_results = []

    classes, classes_source = _resolve_classes(meta, imp_dir)
    log.info("Classes resolved from %s: %s", classes_source, classes.tolist())

    X_raw_all = None
    y_all = None
    fold_indices = {}
    raw_cat_features = [col for col in cat_cols if isinstance(col, str)]
    raw_missing_token = meta.get("raw_cat_missing_token", "__MISSING__")
    if run_catboost_raw:
        X_raw_all = pd.read_parquet(raw_path)
        raw_cat_features = [col for col in raw_cat_features if col in X_raw_all.columns]
        for col in raw_cat_features:
            X_raw_all[col] = X_raw_all[col].astype("string").fillna(raw_missing_token)
        y_all = pd.read_parquet(proc_dir / "y_prepared.parquet")["target"]
        if len(X_raw_all) != len(y_all):
            raise ValueError(
                f"Raw feature size mismatch: X_raw_prepared has {len(X_raw_all)} rows and y_prepared has {len(y_all)} rows."
            )
        with open(fold_indices_path, "r", encoding="utf-8") as f:
            fold_indices = json.load(f)

    pbar = tqdm(total=len(tasks), desc="Classification")

    for task in tasks:
        imp_name = task["imputer"]
        fold = int(task["fold"])
        clf_name = task["classifier"]
        key = _task_key(runtime_mode, imp_name, fold, clf_name)

        if key in completed:
            pbar.update(1)
            continue

        clf_cfg = classifiers[clf_name]
        result = None

        try:
            if not clf_cfg.get("available", False):
                err_msg = clf_cfg.get("error") or "Classifier dependencies unavailable."
                log.error("%s fold%d %s: %s", imp_name, fold, clf_name, err_msg)
                result = {
                    "runtime_mode": runtime_mode,
                    "fold": fold,
                    "imputer": imp_name,
                    "classifier": clf_name,
                    "error": err_msg,
                }
            elif task["kind"] in {"main", "baseline_xgb_native_missing"}:
                X_tr = pd.read_parquet(imp_dir / f"{imp_name}_fold{fold}_train.parquet")
                X_te = pd.read_parquet(imp_dir / f"{imp_name}_fold{fold}_test.parquet")
                y_tr = pd.read_parquet(imp_dir / f"y_fold{fold}_train.parquet")["target"]
                y_te = pd.read_parquet(imp_dir / f"y_fold{fold}_test.parquet")["target"]

                ti = tempos_imp.get(imp_name, {}).get(str(fold), {})
                t_imp_fit = ti.get("time_fit", 0.0)
                t_imp_transform = ti.get("time_transform_train", 0.0) + ti.get("time_transform_test", 0.0)

                result = _evaluate_combination(
                    imp_name,
                    fold,
                    clf_name,
                    clf_cfg,
                    X_tr,
                    X_te,
                    y_tr,
                    y_te,
                    classes,
                    seed=seed,
                    runtime_mode=runtime_mode,
                    tune_max_samples=tune_max_samples,
                    n_inner=n_inner,
                    scoring=scoring,
                    svm_max=svm_max,
                    t_imp_fit=t_imp_fit,
                    t_imp_transform=t_imp_transform,
                    fit_kwargs=None,
                )
                result["runtime_mode"] = runtime_mode
                del X_tr, X_te, y_tr, y_te
            else:
                fold_idx = fold_indices.get(str(fold), fold_indices.get(fold))
                if fold_idx is None:
                    raise KeyError(f"Fold indices not found for fold={fold}")

                tr_idx = np.asarray(fold_idx["train"], dtype=int)
                te_idx = np.asarray(fold_idx["test"], dtype=int)
                X_tr = X_raw_all.iloc[tr_idx].copy()
                X_te = X_raw_all.iloc[te_idx].copy()
                y_tr = y_all.iloc[tr_idx]
                y_te = y_all.iloc[te_idx]

                result = _evaluate_combination(
                    imp_name,
                    fold,
                    clf_name,
                    clf_cfg,
                    X_tr,
                    X_te,
                    y_tr,
                    y_te,
                    classes,
                    seed=seed,
                    runtime_mode=runtime_mode,
                    tune_max_samples=tune_max_samples,
                    n_inner=n_inner,
                    scoring=scoring,
                    svm_max=svm_max,
                    t_imp_fit=0.0,
                    t_imp_transform=0.0,
                    fit_kwargs={"cat_features": raw_cat_features},
                )
                result["runtime_mode"] = runtime_mode
                del X_tr, X_te, y_tr, y_te

        except Exception as e:
            log.error("%s fold%d %s failed during data assembly: %s", imp_name, fold, clf_name, e, exc_info=True)
            traceback.print_exc()
            result = {
                "runtime_mode": runtime_mode,
                "fold": fold,
                "imputer": imp_name,
                "classifier": clf_name,
                "error": str(e),
            }

        all_results.append(result)
        completed.add(key)
        pbar.update(1)
        _save_classification_checkpoint(ckpt_path, completed, all_results)
        gc.collect()

    pbar.close()

    flat = [
        {k: v for k, v in r.items() if k not in ("classification_report", "confusion_matrix")}
        for r in all_results
    ]
    df = pd.DataFrame(flat)
    df.to_csv(out_csv, index=False)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_serialize(all_results), f, indent=2)

    # Keep canonical outputs for downstream analysis.
    if out_csv.name != "all_results.csv":
        df.to_csv(res_dir / "all_results.csv", index=False)
    if out_json.name != "all_results_detailed.json":
        with open(res_dir / "all_results_detailed.json", "w", encoding="utf-8") as f:
            json.dump(_serialize(all_results), f, indent=2)

    log.info(
        "Classification finished (%s). %d result rows saved to %s.",
        runtime_mode,
        len(df),
        out_csv.name,
    )
    return df
