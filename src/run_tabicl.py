"""
Benchmark script for TabICL on prepared SisRHC/INCA artifacts.

Default behavior:
- Uses `data/processed/X_raw_prepared.parquet` and `y_prepared.parquet`
- Reuses folds from `data/imputed/fold_indices.json` when available
- Saves metrics to `results/raw/tabicl_results.csv`
"""

import argparse
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize

try:
    from src.config_loader import load_config
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config_loader import load_config

log = logging.getLogger(__name__)


def _setup_logging():
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _to_numpy(x):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return x.to_numpy()
    return np.asarray(x)


def _serialize(obj):
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _filter_supported_kwargs(fn, kwargs):
    """Filter kwargs based on callable signature for API compatibility."""
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return dict(kwargs), {}

    supported = {}
    dropped = {}
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in params:
            supported[key] = value
        else:
            dropped[key] = value
    return supported, dropped


def _is_repetition_histogram_error(exc):
    return "Repetition level histogram size mismatch" in str(exc)


def _read_parquet_via_helper_python(path, columns=None):
    helper_python = os.environ.get("TABICL_PARQUET_HELPER_PYTHON", sys.executable)
    helper_python_path = Path(helper_python)
    if not helper_python_path.exists():
        raise RuntimeError(f"Helper Python not found: {helper_python_path}")

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        out_path = Path(tmp.name)

    helper_code = r"""
import json
import pandas as pd
import sys

parquet_path = sys.argv[1]
columns_json = sys.argv[2]
output_pickle = sys.argv[3]

cols = json.loads(columns_json)
if not cols:
    cols = None

df = pd.read_parquet(parquet_path, columns=cols)
df.to_pickle(output_pickle)
"""

    try:
        cmd = [
            str(helper_python_path),
            "-c",
            helper_code,
            str(path),
            json.dumps(columns or []),
            str(out_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return pd.read_pickle(out_path)
    except subprocess.CalledProcessError as sub_exc:
        stderr = (sub_exc.stderr or "").strip()
        raise RuntimeError(f"Helper Python parquet read failed: {stderr}") from sub_exc
    finally:
        try:
            out_path.unlink(missing_ok=True)
        except Exception:
            pass


def _read_parquet_safe(path, columns=None):
    """Read parquet with a single pyarrow low-level fallback.

    The previous implementation had 5 nested fallback levels (Polars,
    pyarrow.parquet, row-group recovery, helper subprocess) which masked
    corrupted data. Now we try pandas → pyarrow low-level and fail fast.
    """
    path = Path(path)
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception as exc:
        if not _is_repetition_histogram_error(exc):
            raise

        log.warning(
            "Parquet read failed for %s with '%s'. Retrying with low-level pyarrow.",
            path,
            exc,
        )

    try:
        import pyarrow.parquet as pq

        table = pq.read_table(path, columns=columns, use_threads=False)
        return table.to_pandas()
    except Exception as final_exc:
        raise RuntimeError(
            f"Failed to read parquet file {path}. Original: {exc}. Fallback: {final_exc}"
        ) from final_exc


def _take_rows(X, start, end):
    if isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[start:end]
    return X[start:end]


def _is_oom_error(exc):
    msg = str(exc).lower()
    oom_markers = (
        "out of memory",
        "can't allocate memory",
        "cannot allocate memory",
        "cuda out of memory",
        "cublas status alloc failed",
    )
    return any(marker in msg for marker in oom_markers)


def _one_hot_proba(y_pred, classes):
    y_pred = _to_numpy(y_pred).astype(int)
    proba = np.zeros((len(y_pred), len(classes)), dtype=float)
    class_pos = {int(c): i for i, c in enumerate(classes)}
    for i, y in enumerate(y_pred):
        pos = class_pos.get(int(y))
        if pos is not None:
            proba[i, pos] = 1.0
    return proba


def _align_proba(y_prob, model_classes, full_classes, y_pred):
    y_prob = _to_numpy(y_prob)
    if y_prob.ndim == 1:
        y_prob = np.column_stack([1 - y_prob, y_prob])

    if y_prob.shape[1] == len(full_classes):
        return y_prob

    if model_classes is None:
        return _one_hot_proba(y_pred, full_classes)

    aligned = np.zeros((len(y_prob), len(full_classes)), dtype=float)
    model_classes = [int(c) for c in _to_numpy(model_classes).tolist()]
    pos = {int(c): i for i, c in enumerate(full_classes)}
    for src_idx, cls in enumerate(model_classes):
        dst_idx = pos.get(cls)
        if dst_idx is not None and src_idx < y_prob.shape[1]:
            aligned[:, dst_idx] = y_prob[:, src_idx]

    row_sum = aligned.sum(axis=1, keepdims=True)
    missing = np.where(row_sum.squeeze() == 0)[0]
    if len(missing):
        aligned[missing] = _one_hot_proba(np.asarray(y_pred)[missing], full_classes)
        row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return aligned / row_sum


def _predict_with_chunking(model, X, classes, chunk_size):
    n_rows = len(X)
    if chunk_size is None or int(chunk_size) <= 0 or n_rows <= int(chunk_size):
        chunk_size = n_rows
    else:
        chunk_size = int(chunk_size)

    while True:
        try:
            y_pred_parts = []
            y_prob_parts = []
            model_classes = getattr(model, "classes_", None)
            cls_arr = None if model_classes is None else _to_numpy(model_classes).astype(int)

            for start in range(0, n_rows, chunk_size):
                end = min(start + chunk_size, n_rows)
                X_batch = _take_rows(X, start, end)

                if hasattr(model, "predict_proba"):
                    prob = _to_numpy(model.predict_proba(X_batch))
                    y_prob_parts.append(prob)

                    if cls_arr is not None and prob.ndim == 2 and prob.shape[1] == len(cls_arr):
                        pred = cls_arr[np.argmax(prob, axis=1)]
                    else:
                        pred = _to_numpy(model.predict(X_batch)).astype(int)
                    y_pred_parts.append(pred)
                else:
                    pred = _to_numpy(model.predict(X_batch)).astype(int)
                    y_pred_parts.append(pred)

            y_pred = np.concatenate([_to_numpy(p).reshape(-1) for p in y_pred_parts]).astype(int)
            if y_prob_parts:
                y_prob = np.vstack([_to_numpy(p) for p in y_prob_parts])
            else:
                y_prob = _one_hot_proba(y_pred, classes)
            return y_pred, y_prob
        except RuntimeError as exc:
            if _is_oom_error(exc) and chunk_size > 1:
                new_chunk = max(1, chunk_size // 2)
                log.warning(
                    "OOM during TabICL inference with chunk_size=%d. Retrying with chunk_size=%d.",
                    chunk_size,
                    new_chunk,
                )
                chunk_size = new_chunk
                continue
            raise


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


def _warn_python_compat():
    if sys.version_info >= (3, 13):
        log.warning(
            "tabicl currently declares Python support for >=3.10,<3.13. "
            "If import fails, use a Python 3.10-3.12 environment."
        )


def _torch_cuda_status():
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on runtime
        return False, 0, str(exc)

    try:
        available = bool(torch.cuda.is_available())
        n_devices = int(torch.cuda.device_count()) if available else 0
        return available, n_devices, None
    except Exception as exc:  # pragma: no cover - depends on runtime
        return False, 0, str(exc)


def _configure_numba_cache():
    """Ensure numba cache goes to a writable location (important in managed envs)."""
    if os.environ.get("NUMBA_CACHE_DIR"):
        return

    base_tmp = Path(tempfile.gettempdir())
    cache_dir = Path(os.environ.get("TABICL_NUMBA_CACHE_DIR", str(base_tmp / "tabicl_numba_cache")))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
        log.info("NUMBA_CACHE_DIR set to %s", cache_dir)
    except Exception as exc:
        log.warning("Could not set NUMBA_CACHE_DIR (%s). Import may fail with numba cache errors.", exc)


def _load_tabicl_classifier():
    _warn_python_compat()
    _configure_numba_cache()
    try:
        from tabicl import TabICLClassifier
    except Exception as exc:
        raise RuntimeError(
            "Could not import tabicl. Install with: `pip install tabicl` "
            "(or `pip install git+https://github.com/soda-inria/tabicl.git`). "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    return TabICLClassifier


def _resolve_paths(cfg):
    proc_dir = Path(cfg["paths"]["processed_data"])
    imp_dir = Path(cfg["paths"]["imputed_data"])
    tbl_dir = Path(cfg["paths"]["results_tables"])
    return proc_dir, imp_dir, tbl_dir


def _subsample_if_needed(X, y, max_rows, seed):
    if max_rows is None or len(y) <= max_rows:
        return X.reset_index(drop=True), y.reset_index(drop=True)

    idx = np.arange(len(y))
    sub_idx, _ = train_test_split(idx, train_size=max_rows, stratify=y, random_state=seed)
    sub_idx = np.sort(sub_idx)
    X = X.iloc[sub_idx].reset_index(drop=True)
    y = y.iloc[sub_idx].reset_index(drop=True)
    return X, y


def _load_data(cfg, input_source, max_rows, seed):
    proc_dir, _, tbl_dir = _resolve_paths(cfg)
    x_name = "X_raw_prepared.parquet" if input_source == "raw_prepared" else "X_prepared.parquet"
    X_path = proc_dir / x_name
    X_prepared_path = proc_dir / "X_prepared.parquet"
    y_path = proc_dir / "y_prepared.parquet"
    meta_path = tbl_dir / "metadata.json"

    if not X_path.exists():
        raise FileNotFoundError(f"Missing input features: {X_path}")
    if not y_path.exists():
        raise FileNotFoundError(f"Missing labels: {y_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")

    try:
        X = _read_parquet_safe(X_path)
    except Exception as exc:
        if input_source == "raw_prepared" and X_prepared_path.exists():
            log.warning(
                "Could not read %s (%s). Falling back to %s.",
                X_path,
                exc,
                X_prepared_path,
            )
            X = _read_parquet_safe(X_prepared_path)
        else:
            raise

    y_df = _read_parquet_safe(y_path, columns=["target"])
    if "target" not in y_df.columns:
        raise ValueError(f"Expected column 'target' in {y_path}. Found: {list(y_df.columns)}")
    y = y_df["target"]
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if len(X) != len(y):
        raise ValueError(f"Row mismatch: X has {len(X)} rows and y has {len(y)} rows.")

    X, y = _subsample_if_needed(X, y, max_rows=max_rows, seed=seed)
    return X, y, meta


def _load_existing_folds(fold_indices_path):
    with open(fold_indices_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    folds = []
    for key in sorted(raw.keys(), key=lambda v: int(v)):
        fold = int(key)
        tr_idx = np.asarray(raw[key]["train"], dtype=int)
        te_idx = np.asarray(raw[key]["test"], dtype=int)
        folds.append((fold, tr_idx, te_idx))
    return folds


def _generate_folds(y, n_splits, seed):
    y_np = _to_numpy(y).astype(int)
    _, counts = np.unique(y_np, return_counts=True)
    min_count = int(counts.min())
    if min_count < n_splits:
        n_splits = min_count
        log.warning("Reducing n_splits to %d due to small class support.", n_splits)
    if n_splits < 2:
        raise ValueError("Need at least 2 folds for cross-validation.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(fold, tr_idx, te_idx) for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y_np)), y_np))]


def _resolve_folds(cfg, y, split_source, n_folds, seed):
    _, imp_dir, _ = _resolve_paths(cfg)
    fold_indices_path = imp_dir / "fold_indices.json"

    if split_source == "existing":
        if not fold_indices_path.exists():
            raise FileNotFoundError(
                f"{fold_indices_path} not found. Run `python main.py --step impute` "
                "or use `--split-source new`."
            )
        folds = _load_existing_folds(fold_indices_path)
        max_idx = max(int(idx) for _, tr, te in folds for idx in np.concatenate([tr, te]))
        if max_idx >= len(y):
            raise ValueError(
                "Existing fold indices are incompatible with current dataset rows. "
                "Use `--split-source new`."
            )
        return folds

    resolved = n_folds if n_folds else int(cfg["cv"]["n_outer_folds"])
    return _generate_folds(y=y, n_splits=resolved, seed=seed)


def _resolve_categorical_features(mode, meta, X):
    if mode == "none":
        return []
    if mode == "auto":
        return "auto"

    cat_cols = meta.get("cat_cols", [])
    cat_cols = [c for c in cat_cols if isinstance(c, str) and c in X.columns]
    if not cat_cols:
        return "auto"
    return cat_cols


def _build_inference_config(args):
    cfg = {}
    if args.inference_batch_size is not None:
        cfg["batch_size"] = int(args.inference_batch_size)
    if args.inference_num_estimators is not None:
        cfg["num_estimators"] = int(args.inference_num_estimators)
    return cfg or None


def _resolve_kv_cache(mode):
    if mode == "off":
        return False
    if mode in {"kv", "repr"}:
        return mode
    raise ValueError(f"Unsupported kv cache mode: {mode}")


def _resolve_auto_bool(mode):
    if mode == "auto":
        return "auto"
    if mode == "on":
        return True
    if mode == "off":
        return False
    raise ValueError(f"Unsupported mode: {mode}")


def run_tabicl(args):
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg["experiment"]["random_seed"])
    np.random.seed(seed)
    kv_cache = _resolve_kv_cache(args.kv_cache)
    use_amp = _resolve_auto_bool(args.use_amp)
    use_fa3 = _resolve_auto_bool(args.use_fa3)

    if args.split_source == "existing" and args.max_rows is not None:
        raise ValueError(
            "`--max-rows` cannot be used with `--split-source existing` because fold indices "
            "were generated for the full prepared dataset. Use `--split-source new`."
        )

    X, y, meta = _load_data(
        cfg=cfg,
        input_source=args.input_source,
        max_rows=args.max_rows,
        seed=seed,
    )
    folds = _resolve_folds(
        cfg=cfg,
        y=y,
        split_source=args.split_source,
        n_folds=args.n_folds,
        seed=seed,
    )
    classes = np.sort(np.unique(_to_numpy(y).astype(int).reshape(-1)))
    categorical_features = _resolve_categorical_features(args.categorical_features, meta, X)
    TabICLClassifier = _load_tabicl_classifier()

    cuda_available, n_devices, cuda_err = _torch_cuda_status()
    if args.require_gpu and not cuda_available:
        detail = f" ({cuda_err})" if cuda_err else ""
        raise RuntimeError(
            "GPU required (`--require-gpu`) but CUDA is not available in this environment."
            f"{detail}"
        )
    if cuda_available:
        log.info("CUDA available: %d device(s).", n_devices)
    else:
        log.warning("CUDA not available; TabICL may run on CPU.")

    results = []
    detailed = []
    for fold, tr_idx, te_idx in folds:
        X_tr = X.iloc[tr_idx].copy()
        X_te = X.iloc[te_idx].copy()
        y_tr = y.iloc[tr_idx].astype(int)
        y_te = y.iloc[te_idx].astype(int)

        init_kwargs = {
            "n_estimators": args.n_estimators,
            "class_shift": args.class_shift,
            "outlier_threshold": args.outlier_threshold,
            "n_jobs": args.n_jobs,
            "verbose": args.tabicl_verbose,
            "checkpoint_version": args.checkpoint_version,
            "inference_config": _build_inference_config(args),
            "random_state": seed,
            "device": "cuda" if args.require_gpu and args.device is None else args.device,
            "use_amp": use_amp,
            "use_fa3": use_fa3,
            "offload_mode": args.offload_mode,
            "disk_offload_dir": args.disk_offload_dir,
        }
        init_kwargs, dropped_init = _filter_supported_kwargs(TabICLClassifier.__init__, init_kwargs)
        if dropped_init and fold == 0:
            log.info("Ignoring unsupported TabICL __init__ args: %s", sorted(dropped_init.keys()))
        model = TabICLClassifier(**init_kwargs)

        t0 = time.time()
        fit_kwargs = {
            "max_steps": args.max_steps,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "kv_cache": kv_cache,
            "eval_set": None,
            "deterministic_eval": not args.non_deterministic_eval,
            "score_eval": not args.disable_score_eval,
            "use_early_stopping": not args.disable_early_stopping,
            "categorical_features": categorical_features,
        }
        fit_kwargs, dropped_fit = _filter_supported_kwargs(TabICLClassifier.fit, fit_kwargs)
        if dropped_fit and fold == 0:
            log.info("Ignoring unsupported TabICL fit args: %s", sorted(dropped_fit.keys()))
        model.fit(X_tr, y_tr, **fit_kwargs)
        t_fit = time.time() - t0

        t0 = time.time()
        y_pred, y_prob = _predict_with_chunking(
            model=model,
            X=X_te,
            classes=classes,
            chunk_size=args.predict_chunk_size,
        )
        t_pred = time.time() - t0

        model_classes = getattr(model, "classes_", None)
        y_prob = _align_proba(y_prob, model_classes=model_classes, full_classes=classes, y_pred=y_pred)
        metrics = _compute_metrics(y_te, y_pred, y_prob, classes)

        row = {
            "runtime_mode": "tabicl",
            "fold": int(fold),
            "imputer": args.label,
            "classifier": "TabICL",
            "accuracy": metrics["accuracy"],
            "recall_weighted": metrics["recall_weighted"],
            "f1_weighted": metrics["f1_weighted"],
            "auc_weighted": metrics["auc_weighted"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
            "auc_macro": metrics["auc_macro"],
            "time_imputation_fit": 0.0,
            "time_imputation_transform": 0.0,
            "time_tuning": 0.0,
            "time_fit": t_fit,
            "time_prediction": t_pred,
            "time_total": t_fit + t_pred,
            "best_params": str(
                {
                    "n_estimators": args.n_estimators,
                    "max_steps": args.max_steps,
                    "patience": args.patience,
                    "batch_size": args.batch_size,
                    "kv_cache": args.kv_cache,
                    "inference_config": _build_inference_config(args),
                    "categorical_features": args.categorical_features,
                    "checkpoint_version": args.checkpoint_version,
                    "use_amp": args.use_amp,
                    "use_fa3": args.use_fa3,
                    "offload_mode": args.offload_mode,
                    "disk_offload_dir": args.disk_offload_dir,
                }
            ),
            "best_inner_score": np.nan,
        }
        results.append(row)

        drow = dict(row)
        drow["classification_report"] = metrics["classification_report"]
        drow["confusion_matrix"] = metrics["confusion_matrix"]
        detailed.append(drow)

        log.info(
            "Fold %d | F1w=%.4f | AUCw=%s | fit=%.1fs pred=%.2fs",
            fold,
            metrics["f1_weighted"],
            "nan" if np.isnan(metrics["auc_weighted"]) else f"{metrics['auc_weighted']:.4f}",
            t_fit,
            t_pred,
        )

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(_serialize(detailed), f, indent=2)

    summary = df[["accuracy", "recall_weighted", "f1_weighted", "auc_weighted", "time_total"]].mean(numeric_only=True)
    log.info("Saved: %s", out_csv)
    log.info("Saved: %s", out_json)
    log.info(
        "Mean metrics | acc=%.4f recall_w=%.4f f1_w=%.4f auc_w=%s time_total=%.2fs",
        summary["accuracy"],
        summary["recall_weighted"],
        summary["f1_weighted"],
        "nan" if np.isnan(summary["auc_weighted"]) else f"{summary['auc_weighted']:.4f}",
        summary["time_total"],
    )
    return df


def build_parser():
    parser = argparse.ArgumentParser(description="Run TabICL benchmark on prepared dataset artifacts.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to pipeline config.")
    parser.add_argument(
        "--input-source",
        default="raw_prepared",
        choices=["raw_prepared", "prepared"],
        help="`raw_prepared` uses categorical raw features; `prepared` uses encoded numeric features.",
    )
    parser.add_argument(
        "--split-source",
        default="existing",
        choices=["existing", "new"],
        help="Reuse folds from data/imputed/fold_indices.json or generate new StratifiedKFold splits.",
    )
    parser.add_argument("--n-folds", type=int, default=None, help="Used only with --split-source new.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.random_seed.")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional stratified downsampling before CV.")

    parser.add_argument("--label", default="TabICL_native", help="Label stored in result column `imputer`.")
    parser.add_argument("--n-estimators", type=int, default=8, help="TabICL ensemble size.")
    parser.add_argument("--max-steps", type=int, default=100, help="Max optimization steps for TabICL fit.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size for TabICL fit.")
    parser.add_argument(
        "--kv-cache",
        default="off",
        choices=["off", "kv", "repr"],
        help="Cache mode for TabICL fit: off (default), kv (faster predict, high memory), repr (middle ground).",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs inside TabICL.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail fast if CUDA/GPU is unavailable.")
    parser.add_argument("--device", default=None, help="Optional TabICL device override (e.g., cuda, cpu).")
    parser.add_argument(
        "--offload-mode",
        default="auto",
        choices=["auto", "gpu", "cpu", "disk"],
        help="Memory offload strategy for column embeddings (auto/gpu/cpu/disk).",
    )
    parser.add_argument(
        "--disk-offload-dir",
        default=None,
        help="Directory used when --offload-mode=disk.",
    )
    parser.add_argument(
        "--use-amp",
        default="auto",
        choices=["auto", "on", "off"],
        help="Automatic mixed precision policy (auto/on/off).",
    )
    parser.add_argument(
        "--use-fa3",
        default="auto",
        choices=["auto", "on", "off"],
        help="FlashAttention-3 policy (auto/on/off, effective only if installed).",
    )
    parser.add_argument("--class-shift", type=float, default=0.01, help="TabICL class_shift parameter.")
    parser.add_argument("--outlier-threshold", type=float, default=6.0, help="TabICL outlier_threshold parameter.")
    parser.add_argument("--checkpoint-version", default=None, help="TabICL checkpoint version override.")
    parser.add_argument(
        "--categorical-features",
        default="metadata",
        choices=["metadata", "auto", "none"],
        help="How categorical features are passed to TabICL fit.",
    )
    parser.add_argument(
        "--inference-num-estimators",
        type=int,
        default=None,
        help="Override TabICL inference_config.num_estimators.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=64,
        help="Override TabICL inference_config.batch_size.",
    )
    parser.add_argument(
        "--predict-chunk-size",
        type=int,
        default=512,
        help="Chunk size for inference over test rows (OOM-safe).",
    )
    parser.add_argument(
        "--disable-early-stopping",
        action="store_true",
        help="Disable TabICL early stopping during fit.",
    )
    parser.add_argument(
        "--non-deterministic-eval",
        action="store_true",
        help="Disable deterministic evaluation inside TabICL fit.",
    )
    parser.add_argument(
        "--disable-score-eval",
        action="store_true",
        help="Disable score_eval inside TabICL fit.",
    )
    parser.add_argument("--tabicl-verbose", action="store_true", help="Enable verbose logs from TabICL.")

    parser.add_argument("--output-csv", default="results/raw/tabicl_results.csv", help="Output CSV path.")
    parser.add_argument(
        "--output-json",
        default="results/raw/tabicl_results_detailed.json",
        help="Output detailed JSON path.",
    )
    return parser


def main():
    _setup_logging()
    parser = build_parser()
    args = parser.parse_args()
    run_tabicl(args)


if __name__ == "__main__":
    main()
