"""
Step 1 - Data preparation.
Input: SisRHC raw DataFrame.
Output: X and y prepared in data/processed/.
"""

import gc
import json
import logging
import pickle
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config_loader import load_config

log = logging.getLogger(__name__)


EXCLUDE_REASONS = {
    "TNM": "LEAKAGE",
    "OUTROESTA": "LEAKAGE",
    "ESTADIAG": "LEAKAGE",
    "ESTDFIMT": "LEAKAGE_POST_DIAG",
    "ANTRI": "date",
    "DATAINITRT": "date",
    "DATAOBITO": "date",
    "DATAPRICON": "date",
    "DTDIAGNO": "date",
    "DTTRIAGE": "date",
    "ANOPRIDI": "date",
    "DTPRICON": "date",
    "DTINITRT": "date",
    "CNES": "identifier",
    "MUUH": "identifier",
    "UFUH": "redundant",
    "VALOR_TOT": "irrelevant",
    "TPCASO": "administrative",
    "OCUPACAO": "high_cardinality",
    "PROCEDEN": "high_cardinality",
    "LOCTUPRO": "high_cardinality_or_nan",
    "CLIATEN": "administrative",
    "CLITRAT": "administrative",
}


def _missing_report(df):
    return pd.DataFrame(
        {
            "n_missing": df.isnull().sum(),
            "pct_missing": df.isnull().mean(),
            "n_unique": df.nunique(),
            "dtype": df.dtypes.astype(str),
        }
    ).sort_values("pct_missing", ascending=False)


def _reduce_cardinality(series, min_freq=100):
    counts = series.value_counts(dropna=True)
    rare = counts[counts < min_freq].index
    result = series.copy()
    result[result.isin(rare)] = "OUTROS"
    return result


def _normalize_text(value):
    txt = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return " ".join(txt.lower().split())


def _load_sem_info_codes(dict_path):
    """Load per-column codes labeled as 'sem informacao' from data dictionary."""
    if not dict_path.exists():
        log.warning("Data dictionary not found: %s", dict_path)
        return {}

    with open(dict_path, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    sem_info_codes = {}
    for col, spec in dictionary.items():
        vals = spec.get("valores") if isinstance(spec, dict) else None
        if not isinstance(vals, dict):
            continue

        codes = [str(code).strip() for code, label in vals.items() if "sem informacao" in _normalize_text(label)]
        if codes:
            sem_info_codes[col] = codes

    return sem_info_codes


def _replace_sem_info_with_nan(df, sem_info_codes):
    """Replace known 'sem informacao' category codes by NaN for each available column."""
    replaced = {}
    for col, codes in sem_info_codes.items():
        if col not in df.columns:
            continue

        code_set = {str(c).strip() for c in codes}
        mask = df[col].astype(str).str.strip().isin(code_set)
        n_replaced = int(mask.sum())
        if n_replaced > 0:
            df.loc[mask, col] = np.nan
            replaced[col] = n_replaced

    return replaced


def prepare_data(df, config_path="config/config.yaml"):
    cfg = load_config(config_path)
    seed = cfg["experiment"]["random_seed"]
    data_cfg = cfg["data"]
    paths = cfg["paths"]

    out_proc = Path(paths["processed_data"])
    out_tables = Path(paths["results_tables"])
    out_proc.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    meta = {"shape_original": list(df.shape)}
    log.info("Original shape: %s", df.shape)

    date_col = data_cfg["date_filter_col"]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    n0 = len(df)
    df = df[
        (df[date_col].dt.year >= data_cfg["year_min"])
        & (df[date_col].dt.year <= data_cfg["year_max"])
    ].copy()
    log.info(
        "Date filter %s-%s: %s -> %s",
        data_cfg["year_min"],
        data_cfg["year_max"],
        f"{n0:,}",
        f"{len(df):,}",
    )
    meta["n_after_date_filter"] = len(df)

    dict_path = Path(config_path).resolve().parent / "dicionario_valores_validos.json"
    sem_info_codes = _load_sem_info_codes(dict_path)
    sem_info_replaced = _replace_sem_info_with_nan(df, sem_info_codes)
    if sem_info_replaced:
        log.info("Replaced 'sem informacao' with NaN in %d columns", len(sem_info_replaced))
    if "BASDIAGSP" in sem_info_codes:
        log.info("BASDIAGSP codes treated as NaN: %s", sem_info_codes["BASDIAGSP"])
    meta["sem_info_replaced_by_column"] = sem_info_replaced

    target_col = data_cfg["target_col"]
    log.info(
        "Target distribution before filter (%s):\n%s",
        target_col,
        df[target_col].value_counts(dropna=False).sort_index(),
    )
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    n0 = len(df)
    df = df[df[target_col].isin(data_cfg["valid_classes"])].copy()
    log.info("Target filter: %s -> %s", f"{n0:,}", f"{len(df):,}")
    meta["n_after_target_filter"] = len(df)

    # Keep insertion order while preventing duplicated columns when a feature is
    # listed in both safe and high-cardinality config groups.
    features = list(dict.fromkeys(data_cfg["features_safe"] + data_cfg["features_high_cardinality"]))
    features = [f for f in features if f in df.columns]
    df = df[features + [target_col]].copy()
    log.info("Selected features: %d", len(features))

    mr_pre = _missing_report(df[features])
    mr_pre.to_csv(out_tables / "missing_report_pre_filter.csv")
    threshold = data_cfg["missing_threshold"]
    high_missing = mr_pre[mr_pre["pct_missing"] > threshold].index.tolist()
    if high_missing:
        log.warning(
            "Features removed by missing threshold > %.0f%%: %s",
            threshold * 100,
            high_missing,
        )
        features = [f for f in features if f not in high_missing]
        df = df[features + [target_col]].copy()

    mr_post = _missing_report(df[features])
    mr_post.to_csv(out_tables / "missing_report_post_filter.csv")
    meta["features_excluded_missing"] = high_missing

    min_freq = data_cfg["high_cardinality_min_freq"]
    for col in data_cfg["features_high_cardinality"]:
        if col in df.columns:
            n_before = df[col].nunique(dropna=True)
            df[col] = _reduce_cardinality(df[col], min_freq)
            log.info("Cardinality %s: %d -> %d", col, n_before, df[col].nunique(dropna=True))

    num_cols = [c for c in data_cfg["num_cols"] if c in features]
    cat_cols = [c for c in features if c not in num_cols]

    if "IDADE" in df.columns:
        df["IDADE"] = pd.to_numeric(df["IDADE"], errors="coerce")
        df.loc[(df["IDADE"] < 0) | (df["IDADE"] > 120), "IDADE"] = np.nan

    encoders = {}
    valid_values = {}
    for col in cat_cols:
        le = LabelEncoder()
        mask = df[col].notna()
        encoded = le.fit_transform(df.loc[mask, col].astype(str).str.strip())
        df.loc[mask, col] = encoded
        df[col] = df[col].astype(float)
        encoders[col] = le
        valid_values[col] = np.arange(len(le.classes_), dtype=float)
        log.info("Encoded %s with %d categories", col, len(le.classes_))

    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col].astype(int))
    target_map = dict(zip(le_target.classes_.astype(int), range(len(le_target.classes_))))
    log.info("Target mapping: %s", target_map)

    n_sample = cfg["experiment"]["n_sample"]
    if len(df) > n_sample:
        df, _ = train_test_split(
            df,
            train_size=n_sample,
            stratify=df[target_col],
            random_state=seed,
        )
        log.info("Stratified sample used: %s", f"{len(df):,}")

    ordered_features = num_cols + cat_cols
    X = df[ordered_features].reset_index(drop=True)
    y = df[target_col].reset_index(drop=True)

    assert y.isna().sum() == 0, "Target has NaN values"
    log.info("Final X: %s | y: %s | missing in X: %s", X.shape, y.shape, f"{int(X.isna().sum().sum()):,}")

    X.to_parquet(out_proc / "X_prepared.parquet", index=False)
    y.to_frame("target").to_parquet(out_proc / "y_prepared.parquet", index=False)

    meta.update(
        {
            "features_final": ordered_features,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "n_classes": len(le_target.classes_),
            "target_mapping": {str(k): int(v) for k, v in target_map.items()},
            "shape_final": list(X.shape),
            "n_sample": len(X),
        }
    )
    with open(out_tables / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    with open(out_proc / "encoders.pkl", "wb") as f:
        pickle.dump(
            {
                "feature_encoders": encoders,
                "target_encoder": le_target,
                "valid_values": valid_values,
            },
            f,
        )

    log.info("Saved preparation artifacts in %s", out_proc)

    del df
    gc.collect()
    return X, y, meta
