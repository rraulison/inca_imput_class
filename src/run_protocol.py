#!/usr/bin/env python3
"""
Confirmatory Protocol — Leakage-Free Repeated Nested CV.

This script orchestrates the confirmatory experiment described in
docs/protocol_imputation_downstream.md.

Key guarantees:
  - Imputer is fit ONLY on X_train_outer (no leakage).
  - All (imputer, classifier) combinations share EXACTLY the same
    outer splits within each repetition (strict pairing).
  - Repeated CV with configurable seed schedule.
  - Checkpointing per (repeat, fold, imputer, classifier) for safe resume.

Usage:
    python src/run_protocol.py
    python src/run_protocol.py --dry-run --n-sample 2000 --repeats 1 \
        --imputer Media,NoImpute --classifier XGBoost
"""

import argparse
import gc
import hashlib
import json
import logging
import pickle
import subprocess
import sys
import time
import traceback
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── project imports ───────────────────────────────────────────────────
try:
    from src.config_loader import load_config
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config_loader import load_config

from src.categorical_encoding import encode_with_category_maps, fit_train_category_maps
from src.confirmatory_pipeline import ConfirmatoryPipeline
from src.metrics_utils import compute_metrics, one_hot_proba, serialize, to_numpy
from src.run_classification import (
    _build_classifiers,
    _manual_random_search,
    _n_iter_for_space,
    _resolve_classifier_params,
    _safe_inner_splits,
    _take_rows,
    _fit_with_optional_weights,
)
from src.run_imputation import NO_IMPUTER_NAME, _build_imputers
from src.stats_utils import fmt_time

try:
    from sklearn.utils.class_weight import compute_sample_weight
except ImportError:
    compute_sample_weight = None

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Protocol-specific metrics
# ─────────────────────────────────────────────────────────────────────
# NOTE: CLASS_88 is resolved dynamically from metadata.target_mapping in run_protocol().
# Do NOT set it here as a hardcoded constant — use the class_88 argument threaded through.
ORDINAL_CLASSES = list(range(5))  # 0, 1, 2, 3, 4


def _qwk_from_confusion(cm):
    """Quadratic weighted kappa from a confusion matrix."""
    total = cm.sum()
    k = cm.shape[0]
    if total <= 0 or k < 2:
        return np.nan
    idx = np.arange(k)
    dist_sq = (idx[:, None] - idx[None, :]) ** 2
    weights = dist_sq / float((k - 1) ** 2)
    row_marg = cm.sum(axis=1)
    col_marg = cm.sum(axis=0)
    expected = np.outer(row_marg, col_marg) / total
    num = float((weights * cm).sum())
    den = float((weights * expected).sum())
    if den <= 0:
        return np.nan
    return float(1.0 - (num / den))


def _protocol_metrics(y_true, y_pred, y_prob, classes, class_88=None):
    """Compute full metrics including protocol-specific ordinal/class-88 ones."""
    y_true = to_numpy(y_true).astype(int)
    y_pred = to_numpy(y_pred).astype(int)
    y_prob = to_numpy(y_prob)

    base = compute_metrics(y_true, y_pred, y_prob, classes)

    # ── QWK on ordinal subset (0–4 only) ──
    from sklearn.metrics import confusion_matrix as sk_cm, f1_score, recall_score

    mask_ord = np.isin(y_true, ORDINAL_CLASSES) & np.isin(y_pred, ORDINAL_CLASSES)
    if mask_ord.sum() >= 2:
        cm_ord = sk_cm(y_true[mask_ord], y_pred[mask_ord], labels=ORDINAL_CLASSES)
        base["qwk_0_4"] = _qwk_from_confusion(cm_ord)
    else:
        base["qwk_0_4"] = np.nan

    # ── Class 88 guardrails (skipped gracefully if class_88 is None) ──
    if class_88 is not None:
        has_88_true = (y_true == class_88).sum()
        if has_88_true > 0:
            base["f1_88"] = float(f1_score(y_true == class_88, y_pred == class_88, zero_division=0))
            base["recall_88"] = float(recall_score(y_true == class_88, y_pred == class_88, zero_division=0))
            y_bin_true = (y_true == class_88).astype(int)
            y_bin_pred = (y_pred == class_88).astype(int)
            base["f1_estadiavel_bin"] = float(f1_score(y_bin_true, y_bin_pred, average="weighted", zero_division=0))
        else:
            base["f1_88"] = np.nan
            base["recall_88"] = np.nan
            base["f1_estadiavel_bin"] = np.nan
    else:
        base["f1_88"] = np.nan
        base["recall_88"] = np.nan
        base["f1_estadiavel_bin"] = np.nan

    return base


# ─────────────────────────────────────────────────────────────────────
# Task key and checkpointing
# ─────────────────────────────────────────────────────────────────────
CHECKPOINT_FILE = "protocol_checkpoint.json"


def _git_commit() -> str:
    """Return the current git HEAD hash (or 'unknown')."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
    except Exception:
        return "unknown"


def _checkpoint_signature(cfg: dict, X, y, imp_names: list, clf_names: list) -> str:
    """Build a comprehensive hash covering data, config, and full experiment plan."""
    proto = cfg.get("protocol", {})
    fields = {
        "data_X": hashlib.sha256(
            pd.util.hash_pandas_object(X).values.tobytes()
        ).hexdigest()[:16],
        "data_y": hashlib.sha256(
            np.asarray(y).tobytes()
        ).hexdigest()[:16],
        "proto_cfg": hashlib.sha256(
            json.dumps(proto, sort_keys=True).encode()
        ).hexdigest()[:16],
        "imputers": "|".join(sorted(imp_names)),
        "classifiers": "|".join(sorted(clf_names)),
        "git_commit": _git_commit(),
    }
    return hashlib.sha256(
        json.dumps(fields, sort_keys=True).encode()
    ).hexdigest()[:16]


def _task_key(repeat, fold, imputer, classifier):
    return f"{repeat}__{fold}__{imputer}__{classifier}"


def _load_checkpoint(path, current_sig: str | None = None):
    if not path.exists():
        return {"completed": set(), "results": [], "signature": current_sig}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stored_sig = data.get("signature")
    strict = True  # default; caller may override based on config
    if current_sig is not None and stored_sig != current_sig:
        log.warning(
            "Checkpoint signature mismatch (stored=%s current=%s) — "
            "discarding checkpoint and starting from scratch.",
            stored_sig, current_sig,
        )
        return {"completed": set(), "results": [], "signature": current_sig}
    return {
        "completed": set(data.get("completed", [])),
        "results": data.get("results", []),
        "signature": stored_sig,
    }


def _save_checkpoint(path, completed, results, signature: str | None = None):
    payload = {
        "signature": signature,
        "completed": sorted(completed),
        "results": serialize(results),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


# ─────────────────────────────────────────────────────────────────────
# Manifesto
# ─────────────────────────────────────────────────────────────────────
def _create_manifest(cfg, X, y, out_dir):
    """Create protocol manifest with hashes for reproducibility."""
    X_hash = hashlib.sha256(pd.util.hash_pandas_object(X).values.tobytes()).hexdigest()[:16]
    y_hash = hashlib.sha256(np.asarray(y).tobytes()).hexdigest()[:16]

    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    config_str = json.dumps(cfg.get("protocol", {}), sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

    manifest = {
        "dataset_X_hash": X_hash,
        "dataset_y_hash": y_hash,
        "git_commit": git_hash,
        "config_protocol_hash": config_hash,
        "seed_schedule": cfg["protocol"]["seed_schedule"],
        "n_samples": len(X),
        "n_features": X.shape[1],
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "manifest_protocol.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    log.info("Manifest saved: %s", out_dir / "manifest_protocol.json")
    return manifest


# ─────────────────────────────────────────────────────────────────────
# Main Protocol Runner
# ─────────────────────────────────────────────────────────────────────
def run_protocol(
    config_path: str = "config/config.yaml",
    cfg: dict = None,
    dry_run: bool = False,
    n_sample: int = None,
    repeats: int = None,
    filter_imputers: list = None,
    filter_classifiers: list = None,
) -> pd.DataFrame:
    """Run the confirmatory repeated nested CV protocol.

    Returns a DataFrame with one row per (repeat, fold, imputer, classifier).
    """
    if cfg is None:
        cfg = load_config(config_path)

    proto = cfg.get("protocol", {})
    seed_schedule = proto.get("seed_schedule", [42, 123, 456, 789, 1024])
    n_repeats = repeats or proto.get("repeats", 5)
    n_outer = proto.get("outer_folds", 5)
    n_inner = proto.get("inner_folds", 5)
    tuning_budget = proto.get("tuning_budget", 30)
    scoring = proto.get("primary_metric", "f1_weighted")

    # Ensure we have enough seeds
    if n_repeats > len(seed_schedule):
        rng = np.random.default_rng(seed_schedule[0])
        extra = rng.integers(0, 100000, size=n_repeats - len(seed_schedule)).tolist()
        seed_schedule = seed_schedule + extra
    seed_schedule = seed_schedule[:n_repeats]

    # ── Load data ──
    proc_dir = Path(cfg["paths"]["processed_data"])
    X = pd.read_parquet(proc_dir / "X_prepared.parquet")
    y = pd.read_parquet(proc_dir / "y_prepared.parquet")["target"]

    if n_sample and n_sample < len(X):
        rng = np.random.default_rng(seed_schedule[0])
        idx = rng.choice(len(X), size=n_sample, replace=False)
        idx = np.sort(idx)
        X = X.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        log.info("Subsampled to %d rows", n_sample)

    with open(Path(cfg["paths"]["results_tables"]) / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    all_cols = num_cols + cat_cols
    X = X[all_cols]

    # ── Resolve Class 88 from metadata (P4: no silent fallback) ──
    _target_mapping = meta.get("target_mapping", {})
    # target_mapping may store int keys as strings
    class_88 = _target_mapping.get("88", _target_mapping.get(88, None))
    if class_88 is not None:
        class_88 = int(class_88)
    else:
        log.warning(
            "Class '88' not found in metadata.target_mapping; "
            "class-88 guardrail metrics will be omitted."
        )
    classes = np.array(sorted(y.astype(int).unique()), dtype=int)
    log.info("Protocol: X=%s | classes=%s | repeats=%d | outer=%d | inner=%d",
             X.shape, classes, n_repeats, n_outer, n_inner)

    # ── Manifest ──
    out_dir = Path(cfg["paths"]["results_tables"]) / "protocol"
    res_dir = Path(cfg["paths"]["results_raw"])
    res_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    _create_manifest(cfg, X, y, out_dir)

    # ── Resolve imputers & classifiers ──
    cfg_imputers = cfg.get("imputation", {}).get("imputers", [])
    imp_names = [n for n in cfg_imputers if n in (
        ["Media", "Mediana", "kNN", "MICE_XGBoost", "MICE", "MissForest", NO_IMPUTER_NAME]
    )]
    if filter_imputers:
        imp_names = [n for n in imp_names if n in filter_imputers]

    classifiers = _build_classifiers(cfg)
    clf_names = list(classifiers.keys())
    if filter_classifiers:
        clf_names = [n for n in clf_names if n in filter_classifiers]
    clf_names = [n for n in clf_names if classifiers[n].get("available", True)]

    # ── inner_refit_imputer flag (A4: block fast mode in confirmatory) ──
    inner_refit = proto.get("inner_refit_imputer", True)
    # Override tuning budget
    is_fast_mode = cfg.get("classification", {}).get("runtime", {}).get("mode", "") == "fast"
    
    # We now allow inner_refit=False as a documented feature for huge datasets.
    # We only raise if fast mode was forced, and the config says inner_refit_imputer: True. 
    # But if inner_refit_imputer is already False, we just accept it.
    if is_fast_mode and inner_refit and not dry_run:
        # dry_run is always allowed; fast+confirmatory requires explicit override
        raise ValueError(
            "inner_refit_imputer=true is conceptually opposed to fast runtime mode. "
            "To allow fast mode, set inner_refit_imputer: false explicitly in config.yaml."
        )

    if not inner_refit and not dry_run:
        log.warning("Protocol is running with inner_refit_imputer=False. This saves compute but introduces inner-loop imputation leakage.")

    if is_fast_mode or dry_run:
        inner_refit = False  # in dry_run or fast-mode we allow leaky shortcut
        n_inner = 2

    for name in clf_names:
        if is_fast_mode or dry_run:
            classifiers[name]["n_iter"] = 1
            for k, v in classifiers[name].get("params", {}).items():
                if isinstance(v, list) and len(v) > 0:
                    classifiers[name]["params"][k] = [v[0]]
            continue
            
        params = classifiers[name].get("params", {})
        if params:
            total_combos = 1
            for vals in params.values():
                if isinstance(vals, list):
                    total_combos *= len(vals)
            classifiers[name]["n_iter"] = min(tuning_budget, total_combos)
        else:
            classifiers[name]["n_iter"] = 1

    svm_max_raw = cfg["classification"].get("svm_max_train_samples")
    svm_max = 20000 if svm_max_raw is None else (int(svm_max_raw) if svm_max_raw else None)

    log.info("Imputers: %s", imp_names)
    log.info("Classifiers: %s", clf_names)
    log.info("inner_refit_imputer: %s", inner_refit)

    # ── Checkpoint with full signature (P2/A2) ──
    ckpt_path = res_dir / CHECKPOINT_FILE
    current_sig = _checkpoint_signature(cfg, X, y, imp_names, clf_names)
    ckpt = _load_checkpoint(ckpt_path, current_sig)
    completed = ckpt["completed"]
    all_results = ckpt["results"]

    # ── Persist splits ──
    all_splits = {}

    # ── Main loop ──
    total_combos = n_repeats * n_outer * len(imp_names) * len(clf_names)
    done_count = len(completed)
    log.info("Total combinations: %d | Already completed: %d", total_combos, done_count)

    for rep_idx, seed in enumerate(seed_schedule):
        log.info("═══ Repeat %d/%d (seed=%d) ═══", rep_idx + 1, n_repeats, seed)
        skf = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
        folds = list(skf.split(X, y))

        # Persist split indices
        all_splits[rep_idx] = {
            "seed": seed,
            "folds": {
                fold_idx: {"train": tr_idx.tolist(), "test": te_idx.tolist()}
                for fold_idx, (tr_idx, te_idx) in enumerate(folds)
            },
        }

        for fold_idx, (tr_idx, te_idx) in enumerate(folds):
            X_tr_raw = X.iloc[tr_idx].copy()
            X_te_raw = X.iloc[te_idx].copy()
            y_tr = y.iloc[tr_idx].values.astype(int)
            y_te = y.iloc[te_idx].values.astype(int)

            # ── Encoding for outer test (always fit on outer train) ──
            cat_maps, valid_values_fold = fit_train_category_maps(X_tr_raw, cat_cols)
            X_te_enc, _ = encode_with_category_maps(X_te_raw, num_cols, cat_cols, cat_maps)

            # ── Build imputers for this fold ──
            fold_imputers = _build_imputers(num_cols, cat_cols, valid_values_fold, cfg)

            for imp_name in imp_names:
                # ── Outer imputation (for final test evaluation) ──
                t_imp_start = time.time()
                if imp_name == NO_IMPUTER_NAME:
                    # Encode outer train for final model fit
                    X_tr_enc_outer, _ = encode_with_category_maps(X_tr_raw, num_cols, cat_cols, cat_maps)
                    X_tr_imp = X_tr_enc_outer.values.astype(np.float32)
                    X_te_imp = X_te_enc.values.astype(np.float32)
                    t_imp_fit = 0.0
                    t_imp_transform = 0.0
                else:
                    try:
                        # Encode outer train
                        X_tr_enc_outer, _ = encode_with_category_maps(X_tr_raw, num_cols, cat_cols, cat_maps)
                        imp_clone = clone(fold_imputers[imp_name])
                        t0 = time.time()
                        imp_clone.fit(X_tr_enc_outer)
                        t_imp_fit = time.time() - t0

                        t0 = time.time()
                        X_tr_imp = imp_clone.transform(X_tr_enc_outer)
                        X_te_imp = imp_clone.transform(X_te_enc)
                        t_imp_transform = time.time() - t0

                        X_tr_imp = np.nan_to_num(X_tr_imp, nan=0.0)
                        X_te_imp = np.nan_to_num(X_te_imp, nan=0.0)

                        del imp_clone
                    except Exception as e:
                        log.error("Imputation failed: rep=%d fold=%d imp=%s: %s",
                                  rep_idx, fold_idx, imp_name, e)
                        traceback.print_exc()
                        for clf_name in clf_names:
                            key = _task_key(rep_idx, fold_idx, imp_name, clf_name)
                            if key not in completed:
                                all_results.append({
                                    "repeat": rep_idx, "outer_fold": fold_idx,
                                    "imputer": imp_name, "classifier": clf_name,
                                    "seed": seed, "error": str(e),
                                })
                                completed.add(key)
                        _save_checkpoint(ckpt_path, completed, all_results, current_sig)
                        continue

                for clf_name in clf_names:
                    key = _task_key(rep_idx, fold_idx, imp_name, clf_name)
                    if key in completed:
                        continue

                    clf_cfg = classifiers[clf_name]
                    log.info("  rep=%d fold=%d %s × %s (n_train=%d n_test=%d refit=%s)",
                             rep_idx, fold_idx, imp_name, clf_name, len(y_tr), len(y_te), inner_refit)

                    try:
                        result = _run_single_combo(
                            X_tr_raw=X_tr_raw,
                            X_te_imp=X_te_imp,
                            X_tr_imp=X_tr_imp,
                            y_tr=y_tr,
                            y_te=y_te,
                            classes=classes,
                            clf_name=clf_name,
                            clf_cfg=clf_cfg,
                            seed=seed,
                            fold=fold_idx,
                            n_inner=n_inner,
                            scoring=scoring,
                            svm_max=svm_max,
                            cat_cols=cat_cols,
                            num_cols=num_cols,
                            imp_name=imp_name,
                            imputer=fold_imputers.get(imp_name) if imp_name != NO_IMPUTER_NAME else None,
                            inner_refit=inner_refit,
                            class_88=class_88,
                        )
                        result.update({
                            "repeat": rep_idx,
                            "outer_fold": fold_idx,
                            "imputer": imp_name,
                            "classifier": clf_name,
                            "seed": seed,
                            "runtime_mode": "confirmatory",
                            "n_train": len(y_tr),
                            "n_test": len(y_te),
                            "time_imputation_fit": t_imp_fit,
                            "time_imputation_transform": t_imp_transform,
                        })
                        all_results.append(result)
                        completed.add(key)

                        log.info("    → F1_w=%.4f F1_m=%.4f QWK=%.4f (imp %.1fs + clf %.1fs)",
                                 result.get("f1_weighted", 0),
                                 result.get("f1_macro", 0),
                                 result.get("qwk_0_4", 0),
                                 t_imp_fit + t_imp_transform,
                                 result.get("time_tuning", 0) + result.get("time_prediction", 0))

                    except Exception as e:
                        log.error("Classification failed: rep=%d fold=%d %s × %s: %s",
                                  rep_idx, fold_idx, imp_name, clf_name, e)
                        traceback.print_exc()
                        all_results.append({
                            "repeat": rep_idx, "outer_fold": fold_idx,
                            "imputer": imp_name, "classifier": clf_name,
                            "seed": seed, "error": str(e),
                        })
                        completed.add(key)

                    # Save checkpoint after each combination
                    _save_checkpoint(ckpt_path, completed, all_results, current_sig)
                    gc.collect()

                del X_tr_imp, X_te_imp
                gc.collect()

            del X_tr_raw, X_te_raw, X_te_enc, fold_imputers
            gc.collect()

    # ── Persist splits ──
    splits_path = res_dir / "protocol_splits.pkl"
    with open(splits_path, "wb") as f:
        pickle.dump(all_splits, f)
    log.info("Splits saved: %s", splits_path)

    # ── Save results CSV ──
    results_df = pd.DataFrame([r for r in all_results if "error" not in r])
    csv_cols = [
        "repeat", "outer_fold", "classifier", "imputer", "seed",
        "accuracy", "f1_weighted", "f1_macro", "auc_weighted",
        "qwk_0_4", "f1_88", "recall_88", "f1_estadiavel_bin",
        "time_imputation_fit", "time_imputation_transform",
        "time_tuning", "time_prediction", "time_total",
        "best_params", "runtime_mode", "n_train", "n_test",
    ]
    available_cols = [c for c in csv_cols if c in results_df.columns]
    results_path = res_dir / "protocol_results.csv"
    results_df[available_cols].to_csv(results_path, index=False)
    log.info("Results saved: %s (%d rows)", results_path, len(results_df))

    # Clean checkpoint on full completion
    if len(completed) >= total_combos:
        ckpt_path.unlink(missing_ok=True)
        log.info("Protocol completed — checkpoint removed.")

    return results_df


# ───────────────────────────────────────────────────────────────────
# Single Combination Runner
# ───────────────────────────────────────────────────────────────────
def _run_single_combo(
    *,
    X_tr_raw,          # raw pd.DataFrame (outer train, before any encoding)
    X_te_imp,          # np.ndarray (outer test, already encoded+imputed)
    X_tr_imp,          # np.ndarray (outer train, already encoded+imputed) — used in fast/legacy path
    y_tr, y_te, classes,
    clf_name, clf_cfg, seed, fold, n_inner, scoring, svm_max,
    cat_cols, num_cols, imp_name, imputer,
    inner_refit: bool = True,
    class_88=None,
):
    """Train classifier with inner CV tuning and evaluate on outer test.

    When inner_refit=True, the tuning uses ConfirmatoryPipeline so that
    encoding+imputation are re-fit on each inner fold (no leakage).
    The final model is then fit on the full outer train with the best
    hyper-parameters found.

    When inner_refit=False (fast/dry-run mode), the pre-imputed arrays
    are used directly (faster, but with inner-fold leakage — documented).
    """
    ytr = to_numpy(y_tr).astype(int)
    yte = to_numpy(y_te).astype(int)

    is_no_impute = (imp_name == NO_IMPUTER_NAME)
    cat_cols_for_catboost = cat_cols if is_no_impute else None

    # ── SVM subsampling (applied to raw indices before any path) ──
    svm_subsample_idx = None
    if clf_name == "cuML_SVM" and svm_max and len(ytr) > svm_max:
        try:
            from sklearn.model_selection import train_test_split as _tts
            _, _, svm_idx, _ = _tts(
                ytr, ytr, train_size=svm_max, stratify=ytr, random_state=seed
            )
            svm_subsample_idx = svm_idx
        except Exception:
            rng_svm = np.random.RandomState(seed)
            svm_subsample_idx = rng_svm.choice(len(ytr), svm_max, replace=False)

    # ── Sample weights ──
    def _make_sample_weight(y_labels):
        if clf_name in {"XGBoost", "CatBoost"} and compute_sample_weight is not None:
            return compute_sample_weight("balanced", y_labels)
        return None

    t_tuning = 0.0
    best_params = {}
    best_score = np.nan

    if inner_refit:
        # ──────────────────────────────────────────────────
        # LEAKAGE-FREE PATH: ConfirmatoryPipeline
        # Each inner fold re-fits encode → impute → scale → clf
        # ──────────────────────────────────────────────────
        X_raw_tr = X_tr_raw
        y_raw_tr = ytr
        if svm_subsample_idx is not None:
            X_raw_tr = X_raw_tr.iloc[svm_subsample_idx].reset_index(drop=True)
            y_raw_tr = y_raw_tr[svm_subsample_idx]

        sw = _make_sample_weight(y_raw_tr)

        pipeline = ConfirmatoryPipeline(
            clf=clone(clf_cfg["model"]),
            imputer=clone(imputer) if imputer is not None else None,
            num_cols=num_cols,
            cat_cols=cat_cols,
            needs_scaling=clf_cfg["needs_scaling"],
            cat_cols_for_catboost=cat_cols_for_catboost,
        )

        n_inner_eff = _safe_inner_splits(y_raw_tr, n_inner)
        if n_inner_eff is None:
            pipeline.fit(X_raw_tr, y_raw_tr, sample_weight=sw)
        else:
            inner_cv = StratifiedKFold(n_splits=n_inner_eff, shuffle=True, random_state=seed + fold)
            # Translate clf param_distributions to pipeline's clf__ namespace
            clf_params = clf_cfg.get("params", {})
            pipe_params = {f"clf__{k}": v for k, v in clf_params.items()}
            t0 = time.time()
            pipeline, best_params_prefixed, best_score = _manual_random_search(
                estimator=pipeline,
                param_distributions=pipe_params,
                n_iter=clf_cfg["n_iter"],
                cv=inner_cv,
                scoring=scoring,
                X=X_raw_tr,
                y=y_raw_tr,
                seed=seed + fold,
                sample_weight=sw,
            )
            t_tuning = time.time() - t0
            # Strip clf__ prefix for storage
            best_params = {k[5:]: v for k, v in best_params_prefixed.items()}


        # Final prediction on outer test.
        # The pipeline was fit on the outer train; we apply its fitted scaler
        # (if any) to X_te_imp (already encoded+imputed with outer cat_maps/imputer).
        if clf_cfg["needs_scaling"] and pipeline._scaler is not None:
            Xte_final = pipeline._scaler.transform(X_te_imp)
        else:
            Xte_final = X_te_imp.astype(np.float32)
        t0 = time.time()
        y_pred = to_numpy(pipeline._clf_fitted.predict(Xte_final)).astype(int)
        if hasattr(pipeline._clf_fitted, "predict_proba"):
            y_prob = to_numpy(pipeline._clf_fitted.predict_proba(Xte_final))
        else:
            y_prob = one_hot_proba(y_pred, classes)
        t_pred = time.time() - t0

        best_model_ref = pipeline

    else:
        # ──────────────────────────────────────────────────
        # FAST / DRY-RUN PATH (inner-fold leakage documented)
        # Uses pre-imputed arrays passed from caller
        # ──────────────────────────────────────────────────
        Xtr = X_tr_imp.astype(np.float32)
        Xte_final = X_te_imp.astype(np.float32)

        if clf_cfg["needs_scaling"]:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte_final = scaler.transform(Xte_final)

        if svm_subsample_idx is not None:
            Xtr = _take_rows(Xtr, svm_subsample_idx)
            ytr = ytr[svm_subsample_idx]

        sw = _make_sample_weight(ytr)

        fit_kwargs = {}
        if clf_name == "CatBoost" and cat_cols_for_catboost:
            n_feats = Xtr.shape[1]
            n_cats = len(cat_cols_for_catboost)
            if n_cats > 0 and n_cats <= n_feats:
                fit_kwargs["cat_features"] = list(range(n_feats - n_cats, n_feats))

        n_inner_eff = _safe_inner_splits(ytr, n_inner)
        if n_inner_eff is None:
            best_model = clone(clf_cfg["model"])
            _fit_with_optional_weights(best_model, Xtr, ytr, sample_weight=sw, fit_kwargs=fit_kwargs)
            t_tuning = 0.0
        else:
            inner_cv = StratifiedKFold(n_splits=n_inner_eff, shuffle=True, random_state=seed + fold)
            t0 = time.time()
            best_model, best_params, best_score = _manual_random_search(
                estimator=clf_cfg["model"],
                param_distributions=clf_cfg.get("params", {}),
                n_iter=clf_cfg["n_iter"],
                cv=inner_cv,
                scoring=scoring,
                X=Xtr,
                y=ytr,
                seed=seed + fold,
                sample_weight=sw,
                fit_kwargs=fit_kwargs,
            )
            t_tuning = time.time() - t0

        t0 = time.time()
        y_pred = to_numpy(best_model.predict(Xte_final)).astype(int)
        if hasattr(best_model, "predict_proba"):
            y_prob = to_numpy(best_model.predict_proba(Xte_final))
        else:
            y_prob = one_hot_proba(y_pred, classes)
        t_pred = time.time() - t0
        best_model_ref = best_model

    # ── Post-process proba ──
    if y_prob.ndim == 1:
        y_prob = np.column_stack([1 - y_prob, y_prob])
    if y_prob.shape[1] != len(classes):
        y_prob = one_hot_proba(y_pred, classes)

    # ── Metrics ──
    metrics = _protocol_metrics(yte, y_pred, y_prob, classes, class_88=class_88)

    result = {
        "accuracy": metrics["accuracy"],
        "f1_weighted": metrics["f1_weighted"],
        "f1_macro": metrics["f1_macro"],
        "auc_weighted": metrics["auc_weighted"],
        "qwk_0_4": metrics["qwk_0_4"],
        "f1_88": metrics["f1_88"],
        "recall_88": metrics["recall_88"],
        "f1_estadiavel_bin": metrics["f1_estadiavel_bin"],
        "time_tuning": t_tuning,
        "time_prediction": t_pred,
        "time_total": t_tuning + t_pred,
        "best_params": str(best_params),
        "best_inner_score": float(best_score) if not np.isnan(best_score) else None,
        "inner_refit": inner_refit,
    }

    del best_model_ref
    gc.collect()
    return result


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────
def _setup_logging():
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("results/raw/protocol.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Confirmatory Protocol — Repeated Nested CV")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick validation run with small sample")
    parser.add_argument("--n-sample", type=int, default=None,
                        help="Subsample size (e.g. 2000 for dry-run)")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Override number of repetitions")
    parser.add_argument("--imputer", default=None,
                        help="Comma-separated imputer filter (e.g. Media,NoImpute)")
    parser.add_argument("--classifier", default=None,
                        help="Comma-separated classifier filter (e.g. XGBoost)")
    args = parser.parse_args()

    _setup_logging()

    cfg = load_config(args.config)

    n_sample = args.n_sample
    repeats = args.repeats
    if args.dry_run:
        n_sample = n_sample or 2000
        repeats = repeats or 1
        log.info("DRY-RUN mode: n_sample=%d, repeats=%d", n_sample, repeats)

    filter_imputers = args.imputer.split(",") if args.imputer else None
    filter_classifiers = args.classifier.split(",") if args.classifier else None

    run_protocol(
        config_path=args.config,
        cfg=cfg,
        dry_run=args.dry_run,
        n_sample=n_sample,
        repeats=repeats,
        filter_imputers=filter_imputers,
        filter_classifiers=filter_classifiers,
    )


if __name__ == "__main__":
    main()
