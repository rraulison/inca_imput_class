#!/usr/bin/env python3
"""
Pipeline: Imputation x Classification for oncology staging.
Data source: SisRHC/INCA.

Usage:
    python main.py
    python main.py --step prepare
    python main.py --step impute
    python main.py --step classify
    python main.py --step analyze
    python main.py --step temporal
    python main.py --step protocol
    python main.py --step protocol --dry-run --protocol-imputer Media,NoImpute --protocol-classifier XGBoost
    python main.py --runtime-mode hybrid --n-sample 100000
    python main.py --runtime-mode fast --n-sample 100000
    python main.py --step classify --classifier XGBoost
    python main.py --step impute --imputer MICE_XGBoost
"""

import argparse
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import warnings

# Suppress RAPIDS cuML SVC probability warnings
warnings.filterwarnings("ignore", message="Random state is currently ignored by probabilistic SVC")


def setup_logging():
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("results/raw/experiment.log", mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def set_seed(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)  # kept for backward compat with libs using global state
    os.environ["PYTHONHASHSEED"] = str(seed)


def save_environment():
    Path("results/raw").mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, timeout=60)
        (Path("results/raw") / "pip_freeze.txt").write_text(result.stdout, encoding="utf-8")
    except Exception:
        pass

    import pandas as pd
    import scipy
    import sklearn
    import xgboost

    env = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "scipy": scipy.__version__,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        import catboost

        env["catboost"] = catboost.__version__
    except Exception:
        env["catboost"] = "not_installed"

    try:
        import cupy

        env["cupy"] = cupy.__version__
    except Exception:
        env["cupy"] = "not_installed"

    try:
        import cuml

        env["cuml"] = cuml.__version__
    except Exception:
        env["cuml"] = "not_installed"

    return env


def apply_runtime_overrides(cfg, args, log):
    mode = args.runtime_mode.lower()
    runtime_cfg = cfg.setdefault("classification", {}).setdefault("runtime", {})
    runtime_cfg["mode"] = mode

    if args.tune_max_samples is not None:
        runtime_cfg["tune_max_samples"] = int(args.tune_max_samples)
    elif mode == "hybrid":
        runtime_cfg["tune_max_samples"] = 20000
    else:
        runtime_cfg["tune_max_samples"] = None

    if args.n_sample is not None:
        cfg["experiment"]["n_sample"] = int(args.n_sample)
    elif mode in {"hybrid", "fast"}:
        n_sample_raw = cfg["experiment"].get("n_sample")
        if n_sample_raw is None or int(n_sample_raw) < 100000:
            cfg["experiment"]["n_sample"] = 100000

    log.info(
        "Runtime mode: %s | n_sample=%s | tune_max_samples=%s",
        mode,
        cfg["experiment"]["n_sample"],
        runtime_cfg.get("tune_max_samples"),
    )


def main():
    parser = argparse.ArgumentParser(description="Pipeline Imputation x Classification")
    parser.add_argument("--step", default="all", help="prepare, impute, classify, analyze, temporal, protocol, all")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--data", default=None, help="Raw CSV path")
    parser.add_argument("--imputer", default=None, help="Filter to one imputer")
    parser.add_argument("--classifier", default=None, help="Filter to one classifier")
    parser.add_argument(
        "--runtime-mode",
        default="default",
        choices=["default", "hybrid", "fast"],
        help="default=full tuning, hybrid=light tuning on subset, fast=fixed params without tuning",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=None,
        help="Override experiment.n_sample (e.g., 100000)",
    )
    parser.add_argument(
        "--tune-max-samples",
        type=int,
        default=None,
        help="Max train samples for tuning in hybrid mode (default: 20000)",
    )
    # Protocol flags
    parser.add_argument("--dry-run", action="store_true", help="Quick protocol validation run")
    parser.add_argument("--repeats", type=int, default=None, help="Override protocol repeats")
    parser.add_argument("--protocol-imputer", default=None, help="Comma-separated imputer filter for protocol")
    parser.add_argument("--protocol-classifier", default=None, help="Comma-separated classifier filter for protocol")
    args = parser.parse_args()

    log = setup_logging()
    steps = args.step.lower().split(",")
    run_all = "all" in steps

    from src.config_loader import load_config

    cfg = load_config(args.config)
    apply_runtime_overrides(cfg, args, log)
    set_seed(cfg["experiment"]["random_seed"])

    log.info("=" * 58)
    log.info(" Oncology staging pipeline (imputation x classification) ")
    log.info("=" * 58)

    env = save_environment()
    for key, value in env.items():
        log.info("%s: %s", key, value)

    if run_all or "prepare" in steps:
        log.info("STEP 1 - PREPARE")
        import pandas as pd

        from src.data_preparation import prepare_data

        data_path = args.data or cfg["data"]["filepath"]
        data_path_obj = Path(data_path)
        suffix = data_path_obj.suffix.lower()

        if suffix == ".parquet":
            df = pd.read_parquet(data_path_obj)
        elif suffix in {".csv", ".txt"}:
            try:
                df = pd.read_csv(data_path_obj, low_memory=False, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(data_path_obj, low_memory=False, encoding="latin1")
        else:
            raise ValueError(
                f"Unsupported input format for '{data_path_obj}'. Use .parquet, .csv or .txt."
            )
        prepare_data(df, args.config, cfg=cfg)
        del df

    if run_all or "impute" in steps:
        log.info("STEP 2 - IMPUTE")
        from src.run_imputation import run_imputation

        filter_imputers = [args.imputer] if args.imputer else None
        run_imputation(args.config, filter_imputers=filter_imputers, cfg=cfg)

    if run_all or "classify" in steps:
        log.info("STEP 3 - CLASSIFY")
        from src.run_classification import run_classification

        filter_imputers = [args.imputer] if args.imputer else None
        filter_classifiers = [args.classifier] if args.classifier else None
        run_classification(
            args.config,
            filter_imputers=filter_imputers,
            filter_classifiers=filter_classifiers,
            cfg=cfg,
        )

    if run_all or "temporal" in steps:
        log.info("STEP 5 - TEMPORAL SENSITIVITY")
        from src.run_temporal_sensitivity import run_temporal_sensitivity

        filter_imputers = [args.imputer] if args.imputer else None
        filter_classifiers = [args.classifier] if args.classifier else None
        run_temporal_sensitivity(
            config_path=args.config,
            cfg=cfg,
            filter_imputers=filter_imputers,
            filter_classifiers=filter_classifiers,
            runtime_mode=args.runtime_mode,
        )

    if run_all or "analyze" in steps:
        log.info("STEP 4 - ANALYZE")
        from src.run_analysis import run_analysis

        run_analysis(args.config, cfg=cfg)

    if "protocol" in steps:
        log.info("STEP 6 - CONFIRMATORY PROTOCOL")
        from src.run_protocol import run_protocol
        from src.run_protocol_stats import run_protocol_stats

        n_sample_proto = args.n_sample
        repeats_proto = args.repeats
        if args.dry_run:
            n_sample_proto = n_sample_proto or 2000
            repeats_proto = repeats_proto or 1
            log.info("Protocol DRY-RUN: n_sample=%d, repeats=%d", n_sample_proto, repeats_proto)

        proto_imputers = args.protocol_imputer.split(",") if args.protocol_imputer else None
        proto_classifiers = args.protocol_classifier.split(",") if args.protocol_classifier else None

        run_protocol(
            config_path=args.config,
            cfg=cfg,
            dry_run=args.dry_run,
            n_sample=n_sample_proto,
            repeats=repeats_proto,
            filter_imputers=proto_imputers,
            filter_classifiers=proto_classifiers,
        )

        log.info("Protocol experiment complete — running statistical analysis...")
        best_path = run_protocol_stats(config_path=args.config, cfg=cfg)

        # P5/A3: chain temporal holdout using the structured best_method.json
        if best_path and best_path.exists():
            import json as _json
            best = _json.loads(best_path.read_text(encoding="utf-8"))
            best_imputer = best.get("imputer")
            if best_imputer:
                log.info("Best imputer selected: %s — running temporal holdout.", best_imputer)
                from src.run_temporal_sensitivity import run_temporal_sensitivity
                run_temporal_sensitivity(
                    config_path=args.config,
                    cfg=cfg,
                    filter_imputers=[best_imputer],
                    runtime_mode=args.runtime_mode,
                )
            else:
                log.info(
                    "No imputer passed the confirmatory criteria — temporal holdout skipped. "
                    "See %s for details.", best_path
                )

    log.info("Completed at: %s", datetime.now().isoformat())


if __name__ == "__main__":
    main()
