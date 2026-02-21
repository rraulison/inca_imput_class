"""Temporal external validation sensitivity analysis.

Train on earlier years and test on later years using prepared artifacts, with
the same imputation/classification stack used in the main pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from src.categorical_encoding import encode_with_category_maps, fit_train_category_maps
    from src.config_loader import load_config
    from src.metrics_utils import serialize
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.categorical_encoding import encode_with_category_maps, fit_train_category_maps
    from src.config_loader import load_config
    from src.metrics_utils import serialize

log = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _split_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    return items or None


def _resolve_imputers(
    cfg: Dict,
    filter_imputers: Optional[List[str]],
    base_imputer_names: List[str],
    no_imputer_name: str,
) -> List[str]:
    cfg_imputers = cfg.get("imputation", {}).get("imputers") or (base_imputer_names + [no_imputer_name])
    allowed = set(base_imputer_names + [no_imputer_name])

    out: List[str] = []
    seen = set()
    for name in cfg_imputers:
        if name in seen:
            continue
        seen.add(name)
        if name in allowed:
            out.append(name)
        else:
            log.warning("Configured imputer '%s' is not available and will be skipped.", name)

    if filter_imputers:
        allow = set(filter_imputers)
        out = [name for name in out if name in allow]

    if not out:
        raise ValueError("No valid imputers selected for temporal sensitivity analysis.")
    return out


def run_temporal_sensitivity(
    config_path: str = "config/config.yaml",
    cfg: Optional[Dict] = None,
    train_end_year: Optional[int] = None,
    test_start_year: Optional[int] = None,
    test_end_year: Optional[int] = None,
    filter_imputers: Optional[List[str]] = None,
    filter_classifiers: Optional[List[str]] = None,
    runtime_mode: Optional[str] = None,
) -> pd.DataFrame:
    cfg = cfg or load_config(config_path)
    temporal_cfg = cfg.get("temporal_validation", {})
    train_end_year = int(train_end_year if train_end_year is not None else temporal_cfg.get("train_end_year", 2020))
    test_start_year = int(test_start_year if test_start_year is not None else temporal_cfg.get("test_start_year", 2021))
    test_end_year = int(test_end_year if test_end_year is not None else temporal_cfg.get("test_end_year", 2023))

    from src.run_classification import (
        _build_classifiers,
        _evaluate_combination,
        _ordered_config_classifiers,
        _runtime_setup,
    )
    from src.run_imputation import BASE_IMPUTER_NAMES, NO_IMPUTER_NAME, _build_imputers

    if runtime_mode is not None:
        cfg.setdefault("classification", {}).setdefault("runtime", {})["mode"] = runtime_mode

    proc_dir = Path(cfg["paths"]["processed_data"])
    raw_dir = Path(cfg["paths"]["results_raw"])
    table_dir = Path(cfg["paths"]["results_tables"]) / "temporal_sensitivity"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    X = pd.read_parquet(proc_dir / "X_prepared.parquet")
    y = pd.read_parquet(proc_dir / "y_prepared.parquet")["target"].astype(int)
    temporal_ref_path = proc_dir / "temporal_reference.parquet"
    if not temporal_ref_path.exists():
        raise FileNotFoundError(
            f"Missing temporal reference file: {temporal_ref_path}. "
            "Run Step 1 (prepare) with the updated pipeline first."
        )

    years = pd.read_parquet(temporal_ref_path)["year"]
    if len(X) != len(y) or len(X) != len(years):
        raise ValueError("X, y and temporal reference must have the same number of rows.")

    with open(Path(cfg["paths"]["results_tables"]) / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    num_cols = meta["num_cols"]
    cat_cols = meta["cat_cols"]
    all_cols = num_cols + cat_cols
    X = X[all_cols]

    years_num = pd.to_numeric(years, errors="coerce")
    train_mask = years_num <= int(train_end_year)
    test_mask = (years_num >= int(test_start_year)) & (years_num <= int(test_end_year))

    if int(train_mask.sum()) == 0:
        raise ValueError("Temporal split produced zero training rows.")
    if int(test_mask.sum()) == 0:
        raise ValueError("Temporal split produced zero test rows.")

    X_train_raw = X.loc[train_mask].reset_index(drop=True)
    X_test_raw = X.loc[test_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_test = y.loc[test_mask].reset_index(drop=True)

    log.info(
        "Temporal split | train<=%d: %d rows | test=%d-%d: %d rows",
        train_end_year,
        len(X_train_raw),
        test_start_year,
        test_end_year,
        len(X_test_raw),
    )

    cat_maps, valid_values = fit_train_category_maps(X_train_raw, cat_cols)
    X_train_encoded, _ = encode_with_category_maps(X_train_raw, num_cols, cat_cols, cat_maps)
    X_test_encoded, unseen_test = encode_with_category_maps(X_test_raw, num_cols, cat_cols, cat_maps)
    unseen_total = int(sum(unseen_test.values()))
    if unseen_total > 0:
        log.warning("Temporal test split has %d unseen categorical values mapped to NaN.", unseen_total)

    imputers = _resolve_imputers(cfg, filter_imputers, BASE_IMPUTER_NAMES, NO_IMPUTER_NAME)

    classifiers = _build_classifiers(cfg)
    runtime_mode_eff, tune_max_samples, n_inner = _runtime_setup(cfg, classifiers)
    requested_clf = filter_classifiers or _ordered_config_classifiers(cfg)
    classifier_names = [name for name in requested_clf if name in classifiers]
    if not classifier_names:
        raise ValueError("No valid classifiers selected for temporal sensitivity analysis.")

    seed = int(cfg["experiment"]["random_seed"])
    scoring = cfg["classification"]["tuning"]["scoring"]
    svm_max_raw = cfg["classification"].get("svm_max_train_samples")
    svm_max = 20000 if svm_max_raw is None else int(svm_max_raw)
    classes = np.array(sorted(set(y_train.unique()).union(set(y_test.unique()))), dtype=int)

    results = []
    for imp_name in imputers:
        if imp_name == NO_IMPUTER_NAME:
            X_tr = X_train_encoded.copy()
            X_te = X_test_encoded.copy()
            t_imp_fit, t_imp_transform = 0.0, 0.0
        else:
            if imp_name not in BASE_IMPUTER_NAMES:
                for clf_name in classifier_names:
                    results.append(
                        {
                            "fold": 0,
                            "imputer": imp_name,
                            "classifier": clf_name,
                            "runtime_mode": runtime_mode_eff,
                            "error": f"imputer_not_available:{imp_name}",
                        }
                    )
                continue

            fold_imputers = _build_imputers(num_cols, cat_cols, valid_values, cfg)
            imputer = fold_imputers[imp_name]

            try:
                t0 = time.time()
                imputer.fit(X_train_encoded)
                t_fit = time.time() - t0

                t0 = time.time()
                X_tr_arr = imputer.transform(X_train_encoded)
                t_tr = time.time() - t0

                t0 = time.time()
                X_te_arr = imputer.transform(X_test_encoded)
                t_te = time.time() - t0

                X_tr = pd.DataFrame(np.nan_to_num(X_tr_arr, nan=0.0), columns=all_cols)
                X_te = pd.DataFrame(np.nan_to_num(X_te_arr, nan=0.0), columns=all_cols)
                t_imp_fit = float(t_fit)
                t_imp_transform = float(t_tr + t_te)
            except Exception as exc:
                for clf_name in classifier_names:
                    results.append(
                        {
                            "fold": 0,
                            "imputer": imp_name,
                            "classifier": clf_name,
                            "runtime_mode": runtime_mode_eff,
                            "error": f"imputation_failed:{exc}",
                        }
                    )
                continue

        for clf_name in classifier_names:
            clf_cfg = classifiers[clf_name]
            if not clf_cfg.get("available", False):
                results.append(
                    {
                        "fold": 0,
                        "imputer": imp_name,
                        "classifier": clf_name,
                        "runtime_mode": runtime_mode_eff,
                        "error": clf_cfg.get("error") or "classifier_unavailable",
                    }
                )
                continue

            row = _evaluate_combination(
                imp_name,
                0,
                clf_name,
                clf_cfg,
                X_tr,
                X_te,
                y_train,
                y_test,
                classes,
                seed=seed,
                runtime_mode=runtime_mode_eff,
                tune_max_samples=tune_max_samples,
                n_inner=n_inner,
                scoring=scoring,
                svm_max=svm_max,
                t_imp_fit=t_imp_fit,
                t_imp_transform=t_imp_transform,
                fit_kwargs=None,
            )
            row["runtime_mode"] = runtime_mode_eff
            results.append(row)

    for row in results:
        row["split"] = "temporal_holdout"
        row["train_end_year"] = int(train_end_year)
        row["test_start_year"] = int(test_start_year)
        row["test_end_year"] = int(test_end_year)
        row["n_train"] = int(len(X_train_raw))
        row["n_test"] = int(len(X_test_raw))

    out_csv = raw_dir / "temporal_sensitivity_results.csv"
    out_json = raw_dir / "temporal_sensitivity_results_detailed.json"
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("classification_report", "confusion_matrix")} for r in results])
    df.to_csv(out_csv, index=False)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(serialize(results), f, indent=2)

    valid_df = df[df["error"].isna()].copy() if "error" in df.columns else df.copy()
    if not valid_df.empty:
        summary = (
            valid_df.groupby(["imputer", "classifier"], dropna=False)
            .agg(
                accuracy_mean=("accuracy", "mean"),
                f1_weighted_mean=("f1_weighted", "mean"),
                auc_weighted_mean=("auc_weighted", "mean"),
                time_total_mean=("time_total", "mean"),
            )
            .reset_index()
            .sort_values(["f1_weighted_mean", "auc_weighted_mean"], ascending=[False, False])
        )
    else:
        summary = pd.DataFrame(columns=["imputer", "classifier", "accuracy_mean", "f1_weighted_mean", "auc_weighted_mean", "time_total_mean"])

    summary_path = table_dir / "summary_temporal.csv"
    summary.to_csv(summary_path, index=False)

    manifest = {
        "train_end_year": int(train_end_year),
        "test_start_year": int(test_start_year),
        "test_end_year": int(test_end_year),
        "n_train": int(len(X_train_raw)),
        "n_test": int(len(X_test_raw)),
        "runtime_mode": runtime_mode_eff,
        "imputers": imputers,
        "classifiers": classifier_names,
        "outputs": {
            "results_csv": str(out_csv),
            "results_detailed_json": str(out_json),
            "summary_csv": str(summary_path),
        },
    }
    (table_dir / "manifest_temporal.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("Temporal sensitivity finished. Rows saved: %d", len(df))
    return df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Temporal external validation sensitivity analysis")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--train-end-year", type=int, default=None)
    parser.add_argument("--test-start-year", type=int, default=None)
    parser.add_argument("--test-end-year", type=int, default=None)
    parser.add_argument("--runtime-mode", choices=["default", "hybrid", "fast"], default=None)
    parser.add_argument("--imputers", default=None, help="Comma-separated allowlist")
    parser.add_argument("--classifiers", default=None, help="Comma-separated allowlist")
    return parser


def main() -> None:
    _setup_logging()
    args = _build_parser().parse_args()
    run_temporal_sensitivity(
        config_path=args.config,
        train_end_year=args.train_end_year,
        test_start_year=args.test_start_year,
        test_end_year=args.test_end_year,
        filter_imputers=_split_csv_list(args.imputers),
        filter_classifiers=_split_csv_list(args.classifiers),
        runtime_mode=args.runtime_mode,
    )


if __name__ == "__main__":
    main()
