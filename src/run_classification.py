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
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, StratifiedKFold
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

IMPUTER_NAMES = ["Media", "Mediana", "kNN", "MICE_XGBoost", "MICE", "MissForest"]
CLF_ORDER = ["XGBoost", "CatBoost", "cuML_RF", "cuML_SVM", "cuML_MLP"]


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
        if hasattr(self.model_, "predict_proba"):
            try:
                return _to_numpy(self.model_.predict_proba(X_gpu))
            except Exception:
                pass

        if hasattr(self.model_, "decision_function"):
            try:
                scores = self.model_.decision_function(X_gpu)
                return _softmax(scores)
            except Exception:
                pass

        return _one_hot_proba(self.predict(X), self.classes_)


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

    params = cfg["classification"]["params"]
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
                "n_iter": n_iter,
                "search_n_jobs": 1,
                "engine": "sklearn",
                "available": True,
            },
            "CatBoost": {
                "model": cat_model,
                "params": params["CatBoost"],
                "needs_scaling": False,
                "n_iter": n_iter,
                "search_n_jobs": 1,
                "engine": "sklearn",
                "available": cat_available,
                "error": "catboost is not installed." if not cat_available else None,
            },
            "cuML_RF": {
                "model": CuMLRFWrapper(random_state=seed),
                "params": params["cuML_RF"],
                "needs_scaling": False,
                "n_iter": n_iter,
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
                "n_iter": n_iter_svm,
                "search_n_jobs": 1,
                "engine": "cuml",
                "available": deps["cupy"] and deps["cuml_svm"],
                "error": "cupy/cuml SVC unavailable.",
            },
            "cuML_MLP": {
                "model": CuMLMLPWrapper(random_state=seed),
                "params": params["cuML_MLP"],
                "needs_scaling": True,
                "n_iter": n_iter_mlp,
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


def _fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def _manual_random_search(estimator, param_distributions, n_iter, cv, scoring, X, y, seed):
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
            est.fit(X[tr_idx], y[tr_idx])
            fold_scores.append(float(scorer(est, X[va_idx], y[va_idx])))
            del est
            gc.collect()

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    best_estimator = clone(estimator)
    best_estimator.set_params(**best_params)
    best_estimator.fit(X, y)

    return best_estimator, best_params, best_score


def run_classification(config_path="config/config.yaml", filter_imputers=None, filter_classifiers=None):
    cfg = load_config(config_path)
    seed = cfg["experiment"]["random_seed"]
    n_outer = cfg["cv"]["n_outer_folds"]
    n_inner = cfg["cv"]["n_inner_folds"]
    scoring = cfg["classification"]["tuning"]["scoring"]
    svm_max = cfg["classification"].get("svm_max_train_samples") or 20000

    imp_dir = Path(cfg["paths"]["imputed_data"])
    res_dir = Path(cfg["paths"]["results_raw"])
    res_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    with open(Path(cfg["paths"]["results_tables"]) / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    _ = meta.get("n_classes", None)

    tempos_imp = {}
    tempos_path = res_dir / "tempos_imputacao.json"
    if tempos_path.exists():
        with open(tempos_path, "r", encoding="utf-8") as f:
            tempos_imp = json.load(f)

    imp_names = filter_imputers or IMPUTER_NAMES
    available_imp = [
        name
        for name in imp_names
        if all((imp_dir / f"{name}_fold{f}_train.parquet").exists() for f in range(n_outer))
    ]
    log.info("Available imputers: %s", available_imp)

    classifiers = _build_classifiers(cfg)
    clf_names = filter_classifiers or CLF_ORDER
    clf_names = [name for name in clf_names if name in classifiers]

    ckpt_path = res_dir / "checkpoint_classification.json"
    if ckpt_path.exists():
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
        completed = set(ckpt.get("completed", []))
        all_results = ckpt.get("results", [])
        log.info("Checkpoint loaded: %d completed combinations", len(completed))
    else:
        completed = set()
        all_results = []

    y0 = pd.read_parquet(imp_dir / "y_fold0_train.parquet")["target"]
    classes = np.sort(y0.unique())

    total = len(available_imp) * len(clf_names) * n_outer
    pbar = tqdm(total=total, desc="Classification")
    pbar.update(len(completed))

    for imp_name in available_imp:
        for fold in range(n_outer):
            X_tr = pd.read_parquet(imp_dir / f"{imp_name}_fold{fold}_train.parquet")
            X_te = pd.read_parquet(imp_dir / f"{imp_name}_fold{fold}_test.parquet")
            y_tr = pd.read_parquet(imp_dir / f"y_fold{fold}_train.parquet")["target"]
            y_te = pd.read_parquet(imp_dir / f"y_fold{fold}_test.parquet")["target"]

            ti = tempos_imp.get(imp_name, {}).get(str(fold), {})
            t_imp_fit = ti.get("time_fit", 0)
            t_imp_transform = ti.get("time_transform_train", 0) + ti.get("time_transform_test", 0)

            for clf_name in clf_names:
                key = f"{imp_name}__{fold}__{clf_name}"
                if key in completed:
                    pbar.update(1)
                    continue

                clf_cfg = classifiers[clf_name]
                if not clf_cfg.get("available", False):
                    err_msg = clf_cfg.get("error") or "Classifier dependencies unavailable."
                    log.error("%s fold%d %s: %s", imp_name, fold, clf_name, err_msg)
                    result = {
                        "fold": fold,
                        "imputer": imp_name,
                        "classifier": clf_name,
                        "error": err_msg,
                    }
                    all_results.append(result)
                    completed.add(key)
                    pbar.update(1)
                    with open(ckpt_path, "w", encoding="utf-8") as f:
                        json.dump({"completed": list(completed), "results": _serialize(all_results)}, f)
                    continue

                if clf_cfg["needs_scaling"]:
                    scaler = StandardScaler()
                    Xtr = scaler.fit_transform(X_tr)
                    Xte = scaler.transform(X_te)
                else:
                    Xtr = X_tr.values
                    Xte = X_te.values

                ytr = y_tr.values

                if clf_name == "cuML_SVM" and svm_max and len(Xtr) > svm_max:
                    rng = np.random.RandomState(seed)
                    idx = rng.choice(len(Xtr), svm_max, replace=False)
                    Xtr, ytr = Xtr[idx], ytr[idx]
                    log.warning("cuML_SVM subsampled train set: %d -> %d", len(X_tr), svm_max)

                sample_weight = None
                if clf_name in {"XGBoost", "CatBoost"}:
                    sample_weight = compute_sample_weight("balanced", ytr)

                inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)

                try:
                    t0 = time.time()
                    if clf_cfg["engine"] == "cuml":
                        best_model, best_params, best_score = _manual_random_search(
                            estimator=clf_cfg["model"],
                            param_distributions=clf_cfg["params"],
                            n_iter=clf_cfg["n_iter"],
                            cv=inner_cv,
                            scoring=scoring,
                            X=Xtr,
                            y=ytr,
                            seed=seed,
                        )
                    else:
                        search = RandomizedSearchCV(
                            estimator=clone(clf_cfg["model"]),
                            param_distributions=clf_cfg["params"],
                            n_iter=clf_cfg["n_iter"],
                            cv=inner_cv,
                            scoring=scoring,
                            random_state=seed,
                            n_jobs=clf_cfg["search_n_jobs"],
                            verbose=0,
                            error_score="raise",
                            refit=True,
                        )
                        if sample_weight is not None:
                            search.fit(Xtr, ytr, sample_weight=sample_weight)
                        else:
                            search.fit(Xtr, ytr)
                        best_model = search.best_estimator_
                        best_params = search.best_params_
                        best_score = search.best_score_
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

                    metrics = _compute_metrics(y_te.values, y_pred, y_prob, classes)

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

                except Exception as e:
                    log.error("%s fold%d %s failed: %s", imp_name, fold, clf_name, e)
                    traceback.print_exc()
                    result = {
                        "fold": fold,
                        "imputer": imp_name,
                        "classifier": clf_name,
                        "error": str(e),
                    }

                all_results.append(result)
                completed.add(key)
                pbar.update(1)

                with open(ckpt_path, "w", encoding="utf-8") as f:
                    json.dump({"completed": list(completed), "results": _serialize(all_results)}, f)

                gc.collect()

            del X_tr, X_te, y_tr, y_te
            gc.collect()

    pbar.close()

    flat = [
        {k: v for k, v in r.items() if k not in ("classification_report", "confusion_matrix")}
        for r in all_results
    ]
    df = pd.DataFrame(flat)
    df.to_csv(res_dir / "all_results.csv", index=False)

    with open(res_dir / "all_results_detailed.json", "w", encoding="utf-8") as f:
        json.dump(_serialize(all_results), f, indent=2)

    log.info("Classification finished. %d result rows saved.", len(df))
    return df
