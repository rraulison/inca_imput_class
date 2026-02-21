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
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.config_loader import load_config

log = logging.getLogger(__name__)

NON_INFO_LABEL_MARKERS = (
    "sem informacao",
    "nao avaliado",
    "nao se aplica",
    "nao informado",
    "ignorado",
    "ignorada",
)

# ======= Utility Functions =======

def _normalize_text(value: Any) -> str:
    """Normalize text by removing accents and lowercasing."""
    txt = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return " ".join(txt.lower().split())

def _is_non_informative_label(label: str) -> bool:
    """Check if a label matches any 'non-informative' marker."""
    return any(marker in _normalize_text(label) for marker in NON_INFO_LABEL_MARKERS)

def _reduce_cardinality(series: pd.Series, min_freq: int = 100) -> pd.Series:
    """Replace rare categorical values with 'OUTROS'."""
    counts = series.value_counts(dropna=True)
    rare = counts[counts < min_freq].index
    result = series.copy()
    result[result.isin(rare)] = "OUTROS"
    return result

def _missing_report(df: pd.DataFrame, missing_rules: Optional[Dict] = None) -> pd.DataFrame:
    """Generate a report of missing values and unique counts per column."""
    if not missing_rules:
        return pd.DataFrame({
            "n_missing": df.isnull().sum(),
            "pct_missing": df.isnull().mean(),
            "n_unique": df.nunique(),
            "dtype": df.dtypes.astype(str),
        }).sort_values("pct_missing", ascending=False)

    rows = {}
    for col in df.columns:
        series = df[col]
        mask = series.isna()
        if rules := missing_rules.get(col, {}):
            mask |= _non_informative_mask(series, rules)

        rows[col] = {
            "n_missing": int(mask.sum()),
            "pct_missing": float(mask.mean()),
            "n_unique": int(series.mask(mask).nunique(dropna=True)),
            "dtype": str(series.dtype),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("pct_missing", ascending=False)

# ======= Dictionary Processing =======

def _load_missing_rules(dict_path: Path) -> Dict:
    """Load per-column dictionary rules for codes that should be treated as missing."""
    if not dict_path.exists():
        raise FileNotFoundError(
            f"Data dictionary not found: {dict_path}. "
            f"This file is required to properly mask non-informative codes as NaN."
        )

    with open(dict_path, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    missing_rules = {}
    for col, spec in dictionary.items():
        if not isinstance(spec, dict):
            continue

        vals = spec.get("valores")
        codes, regex_patterns = [], []

        if isinstance(vals, dict):
            codes = [str(k).strip() for k, lbl in vals.items() if _is_non_informative_label(lbl)]
        elif isinstance(vals, str):
            vals_norm = _normalize_text(vals)
            if "mais de tres 9" in vals_norm and "ocupacao ignorad" in vals_norm:
                regex_patterns.append(r"9{4,}")

        if codes or regex_patterns:
            missing_rules[col] = {"codes": sorted(set(codes)), "regex": regex_patterns}

    return missing_rules


def _non_informative_mask(series: pd.Series, rules: Dict) -> pd.Series:
    """Create a boolean mask for non-informative values in a series."""
    series_str = series.astype("string").str.strip()
    mask = pd.Series(False, index=series.index, dtype=bool)

    if codes := [str(c).strip() for c in rules.get("codes", []) if str(c).strip()]:
        code_set = set(codes)
        mask |= series_str.isin(code_set).fillna(False)
        
        numeric_codes = []
        for code in code_set:
            try:
                numeric_codes.append(float(code))
            except (TypeError, ValueError):
                pass
                
        if numeric_codes:
            series_num = pd.to_numeric(series_str, errors="coerce")
            mask |= series_num.isin(numeric_codes).fillna(False)

    for pattern in rules.get("regex", []):
        mask |= series_str.str.fullmatch(pattern, na=False)

    return mask


def _replace_missing_with_nan(df: pd.DataFrame, missing_rules: Dict) -> Dict:
    """Apply dictionary rules and replace non-informative values by NaN."""
    replaced = {}
    for col, rules in missing_rules.items():
        if col in df.columns:
            mask = _non_informative_mask(df[col], rules)
            if (n_replaced := int(mask.sum())) > 0:
                df.loc[mask, col] = np.nan
                replaced[col] = n_replaced
    return replaced


def _enforce_dictionary_domain(df: pd.DataFrame, dict_path: Path) -> Dict:
    """
    Overwrites values not present in the dictionary with the 'Sem informação' code.
    If no such code exists, values are set to NaN.
    """
    with open(dict_path, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    replaced_info = {}
    for col, spec in dictionary.items():
        if col not in df.columns or not isinstance(spec.get("valores"), dict):
            continue
            
        vals = spec["valores"]
        valid_keys = {str(k).strip() for k in vals.keys()}
        
        non_info_code = next(
            (str(k).strip() for k, lbl in vals.items() if _normalize_text(lbl) == "sem informacao"),
            next((str(k).strip() for k, lbl in vals.items() if _is_non_informative_label(lbl)), None)
        )

        valid_nums = []
        for k in valid_keys:
            try:
                valid_nums.append(float(k))
            except ValueError:
                pass

        series_str = df[col].astype("string").str.strip()
        series_num = pd.to_numeric(series_str, errors="coerce")
        
        mask_valid_str = series_str.isin(valid_keys)
        mask_valid_num = series_num.isin(valid_nums) & series_str.notna() & (series_str != "")
        
        is_invalid = df[col].notna() & ~(mask_valid_str | mask_valid_num)
        
        if (n_invalid := is_invalid.sum()) > 0:
            if non_info_code is not None:
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df.loc[is_invalid, col] = float(non_info_code)
                    except ValueError:
                        df.loc[is_invalid, col] = non_info_code
                else:
                    df.loc[is_invalid, col] = non_info_code
            else:
                df.loc[is_invalid, col] = np.nan
            replaced_info[col] = int(n_invalid)

    return replaced_info


# ======= Data Preparation Steps =======

def _filter_by_date(df: pd.DataFrame, date_col: str, year_min: int, year_max: int) -> pd.DataFrame:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    filtered_df = df[(df[date_col].dt.year >= year_min) & (df[date_col].dt.year <= year_max)].copy()
    log.info("Date filter %s-%s: %s -> %s", year_min, year_max, f"{len(df):,}", f"{len(filtered_df):,}")
    return filtered_df

def _clean_target_column(df: pd.DataFrame, target_col: str, valid_classes: list) -> pd.DataFrame:
    """Group granular staging (e.g. 1A -> 1, 01 -> 1) and filter invalid classes."""
    log.info("Target distribution before filter (%s):\n%s", target_col, df[target_col].value_counts(dropna=False).sort_index())
    
    target_str = df[target_col].astype("string").str.strip().str.upper()
    grouped_target = target_str.copy()
    
    is_multidigit_num = target_str.str.match(r"^\d{2}$", na=False)
    has_first_digit = target_str.str.match(r"^\d.*", na=False)
    is_zero_padded = target_str.str.match(r"^0\d$", na=False)
    
    mask_to_group = has_first_digit & ~is_multidigit_num
    grouped_target.loc[mask_to_group] = target_str.loc[mask_to_group].str[0]
    grouped_target.loc[is_zero_padded] = target_str.loc[is_zero_padded].str[1]
    
    df[target_col] = pd.to_numeric(grouped_target, errors="coerce")
    
    n0 = len(df)
    filtered_df = df[df[target_col].isin(valid_classes)].copy()
    log.info("Target filter (after grouping granular staging): %s -> %s", f"{n0:,}", f"{len(filtered_df):,}")
    return filtered_df

def _get_relevant_features(df: pd.DataFrame, data_cfg: Dict) -> list:
    """Consolidate safe list and high cardinality list, excluding 'features_exclude'."""
    all_features = list(dict.fromkeys(data_cfg["features_safe"] + data_cfg["features_high_cardinality"]))
    if excludes := set(data_cfg.get("features_exclude", [])):
        excluded = [f for f in all_features if f in excludes and f in df.columns]
        if excluded:
            log.info("Excluding features from config (data.features_exclude): %s", excluded)
        all_features = [f for f in all_features if f not in excludes]
    return [f for f in all_features if f in df.columns]

def _clean_age(df: pd.DataFrame) -> None:
    """Remove age anomalies."""
    if "IDADE" in df.columns:
        df["IDADE"] = pd.to_numeric(df["IDADE"], errors="coerce")
        df.loc[(df["IDADE"] < 0) | (df["IDADE"] > 120), "IDADE"] = np.nan

# ======= Main Preparation Function =======

def prepare_data(df: pd.DataFrame, config_path: str = "config/config.yaml", cfg: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.Series, Dict]:
    # 1. Initialization and Config
    cfg = cfg or load_config(config_path)
    data_cfg, exp_cfg, paths = cfg["data"], cfg["experiment"], cfg["paths"]

    out_proc, out_tables = Path(paths["processed_data"]), Path(paths["results_tables"])
    out_proc.mkdir(parents=True, exist_ok=True)
    out_tables.mkdir(parents=True, exist_ok=True)

    log.info("Original shape: %s", df.shape)
    meta = {"shape_original": list(df.shape)}

    # 2. Identify Features
    all_features = _get_relevant_features(df, data_cfg)
    dict_path = Path(config_path).resolve().parent / "dicionario_valores_validos.json"
    missing_rules = _load_missing_rules(dict_path)

    mr_raw = _missing_report(df[all_features], missing_rules=missing_rules)
    mr_raw.to_csv(out_tables / "missing_report_raw.csv")
    meta["missing_report_raw_rows"] = len(df)
    log.info("Raw missing report generated for %d features", len(all_features))

    # 3. Filtering and Dictionary Enforcement
    df = _filter_by_date(df, data_cfg["date_filter_col"], data_cfg["year_min"], data_cfg["year_max"])
    meta["n_after_date_filter"] = len(df)

    if replaced_oov := _enforce_dictionary_domain(df, dict_path):
        log.info("Enforced dictionary domain (set OOV to non-informative code) in %d columns", len(replaced_oov))
    meta["out_of_domain_replaced"] = replaced_oov

    if sem_info_replaced := _replace_missing_with_nan(df, missing_rules):
        log.info("Replaced non-informative dictionary values with NaN in %d columns", len(sem_info_replaced))
    
    if "BASDIAGSP" in missing_rules:
        log.info("BASDIAGSP rules treated as NaN: %s", missing_rules["BASDIAGSP"])
    if "OCUPACAO" in missing_rules:
        log.info("OCUPACAO rules treated as NaN: %s", missing_rules["OCUPACAO"])
    
    meta["sem_info_replaced_by_column"] = sem_info_replaced

    df = _clean_target_column(df, data_cfg["target_col"], data_cfg["valid_classes"])
    meta["n_after_target_filter"] = len(df)

    # Keep temporal reference aligned with rows for external temporal validation.
    temporal_year = pd.to_datetime(df[data_cfg["date_filter_col"]], errors="coerce", dayfirst=True).dt.year

    # 4. Feature Selection by Missing Threshold
    features = [f for f in all_features if f in df.columns]
    target_col = data_cfg["target_col"]
    df = df[features + [target_col]].copy()

    mr_pre = _missing_report(df[features])
    mr_pre.to_csv(out_tables / "missing_report_pre_filter.csv")
    
    threshold = data_cfg["missing_threshold"]
    high_missing = mr_pre[mr_pre["pct_missing"] > threshold].index.tolist()
    if high_missing:
        log.warning("Features removed by missing threshold > %.0f%%: %s", threshold * 100, high_missing)
        features = [f for f in features if f not in high_missing]
        df = df[features + [target_col]].copy()

    _missing_report(df[features]).to_csv(out_tables / "missing_report_post_filter.csv")
    meta["features_excluded_missing"] = high_missing

    # 5. Data Transformation (Cardinality, Age)
    min_freq = data_cfg["high_cardinality_min_freq"]
    for col in data_cfg["features_high_cardinality"]:
        if col in df.columns:
            n_before = df[col].nunique(dropna=True)
            df[col] = _reduce_cardinality(df[col], min_freq)
            log.info("Cardinality %s: %d -> %d", col, n_before, df[col].nunique(dropna=True))

    _clean_age(df)

    # 6. Feature Categorization (categorical encoding is done train-only per fold in Step 2)
    num_cols = [c for c in data_cfg["num_cols"] if c in features]
    cat_cols = [c for c in features if c not in num_cols]
    ordered_features = num_cols + cat_cols

    for col in cat_cols:
        df[col] = df[col].astype("string").str.strip()

    df_raw = df[ordered_features].copy()

    le_target = LabelEncoder()
    df[target_col] = le_target.fit_transform(df[target_col].astype(int))
    target_map = {int(k): int(v) for k, v in zip(le_target.classes_, range(len(le_target.classes_)))}
    log.info("Target mapping: %s", target_map)

    # 7. Sampling
    n_sample = exp_cfg["n_sample"]
    if len(df) > n_sample:
        idx_sample, _ = train_test_split(np.arange(len(df)), train_size=n_sample, stratify=df[target_col], random_state=exp_cfg["random_seed"])
        idx_sample.sort()
        df = df.iloc[idx_sample].copy()
        df_raw = df_raw.iloc[idx_sample].copy()
        temporal_year = temporal_year.iloc[idx_sample].copy()
        log.info("Stratified sample used: %s", f"{len(df):,}")

    # 8. Formatting X and Y
    X = df[ordered_features].reset_index(drop=True)
    X_raw = df_raw.reset_index(drop=True)
    temporal_year = temporal_year.reset_index(drop=True)

    for col in num_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in cat_cols:
        X_raw[col] = X_raw[col].astype("string").fillna("__MISSING__")
    
    y = df[target_col].reset_index(drop=True)

    assert y.isna().sum() == 0, "Target has NaN values"
    log.info("Final X: %s | y: %s | missing in X: %s", X.shape, y.shape, f"{int(X.isna().sum().sum()):,}")

    # 9. Save Artifacts
    X.to_parquet(out_proc / "X_prepared.parquet", index=False)
    X_raw.to_parquet(out_proc / "X_raw_prepared.parquet", index=False)
    y.to_frame("target").to_parquet(out_proc / "y_prepared.parquet", index=False)
    temporal_year.to_frame("year").to_parquet(out_proc / "temporal_reference.parquet", index=False)

    meta.update({
        "features_final": ordered_features,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "n_classes": len(le_target.classes_),
        "target_mapping": {str(k): int(v) for k, v in target_map.items()},
        "shape_final": list(X.shape),
        "n_sample": len(X),
        "raw_cat_missing_token": "__MISSING__",
        "categorical_encoding_strategy": "train_only_per_fold",
        "temporal_reference_file": "temporal_reference.parquet",
    })
    
    with open(out_tables / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    with open(out_proc / "encoders.pkl", "wb") as f:
        pickle.dump(
            {
                "feature_encoders": {},
                "target_encoder": le_target,
                "categorical_encoding_strategy": "train_only_per_fold",
            },
            f,
        )

    log.info("Saved preparation artifacts in %s", out_proc)

    del df
    gc.collect()
    return X, y, meta
