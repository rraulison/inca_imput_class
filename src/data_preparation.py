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


NON_INFO_LABEL_MARKERS = (
    "sem informacao",
    "nao avaliado",
    "nao se aplica",
    "nao informado",
    "ignorado",
    "ignorada",
)


# NOTE: Feature exclusion is now driven by data.features_exclude in config.yaml.
# The former EXCLUDE_REASONS dict was removed because it was dead code
# (never referenced by any function) and created a false sense of safety.


def _missing_report(df, missing_rules=None):
    if not missing_rules:
        return pd.DataFrame(
            {
                "n_missing": df.isnull().sum(),
                "pct_missing": df.isnull().mean(),
                "n_unique": df.nunique(),
                "dtype": df.dtypes.astype(str),
            }
        ).sort_values("pct_missing", ascending=False)

    rows = {}
    for col in df.columns:
        series = df[col]
        mask = series.isna()
        rules = missing_rules.get(col, {})
        if rules:
            mask = mask | _non_informative_mask(series, rules)

        rows[col] = {
            "n_missing": int(mask.sum()),
            "pct_missing": float(mask.mean()),
            "n_unique": int(series.mask(mask).nunique(dropna=True)),
            "dtype": str(series.dtype),
        }

    return pd.DataFrame.from_dict(rows, orient="index").sort_values("pct_missing", ascending=False)


def _reduce_cardinality(series, min_freq=100):
    counts = series.value_counts(dropna=True)
    rare = counts[counts < min_freq].index
    result = series.copy()
    result[result.isin(rare)] = "OUTROS"
    return result


def _normalize_text(value):
    txt = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return " ".join(txt.lower().split())


def _is_non_informative_label(label):
    return any(marker in _normalize_text(label) for marker in NON_INFO_LABEL_MARKERS)


def _load_missing_rules(dict_path):
    """Load per-column dictionary rules for codes that should be treated as missing.

    Raises FileNotFoundError if the dictionary JSON is absent, since
    the pipeline depends on it for correct NaN handling of non-informative codes.
    """
    if not dict_path.exists():
        raise FileNotFoundError(
            f"Data dictionary not found: {dict_path}. "
            f"This file is required to properly mask non-informative codes as NaN. "
            f"Please provide '''config/dicionario_valores_validos.json'''."
        )

    with open(dict_path, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    missing_rules = {}
    for col, spec in dictionary.items():
        if not isinstance(spec, dict):
            continue

        vals = spec.get("valores") if isinstance(spec, dict) else None
        codes = []
        regex_patterns = []

        if isinstance(vals, dict):
            codes = [str(code).strip() for code, label in vals.items() if _is_non_informative_label(label)]
        elif isinstance(vals, str):
            vals_norm = _normalize_text(vals)
            # From dictionary text: "mais de tres 9 representa ocupacao ignorada".
            if "mais de tres 9" in vals_norm and "ocupacao ignorad" in vals_norm:
                regex_patterns.append(r"9{4,}")

        if codes or regex_patterns:
            missing_rules[col] = {
                "codes": sorted(set(codes)),
                "regex": regex_patterns,
            }

    return missing_rules


def _non_informative_mask(series, rules):
    series_str = series.astype("string").str.strip()
    mask = pd.Series(False, index=series.index, dtype=bool)

    codes = [str(c).strip() for c in rules.get("codes", []) if str(c).strip()]
    if codes:
        code_set = set(codes)
        mask |= series_str.isin(code_set).fillna(False)

        numeric_codes = []
        for code in code_set:
            try:
                numeric_codes.append(float(code))
            except (TypeError, ValueError):
                continue

        if numeric_codes:
            series_num = pd.to_numeric(series_str, errors="coerce")
            mask |= series_num.isin(numeric_codes).fillna(False)

    for pattern in rules.get("regex", []):
        mask |= series_str.str.fullmatch(pattern, na=False)

    return mask


def _replace_missing_with_nan(df, missing_rules):
    """Apply dictionary rules and replace non-informative values by NaN."""
    replaced = {}
    for col, rules in missing_rules.items():
        if col not in df.columns:
            continue

        mask = _non_informative_mask(df[col], rules)
        n_replaced = int(mask.sum())
        if n_replaced > 0:
            df.loc[mask, col] = np.nan
            replaced[col] = n_replaced

    return replaced


def _enforce_dictionary_domain(df, dict_path):
    """
    Overwrites any value that is not present in the variable's dictionary
    with its designated 'Sem informação' code (like 9 or 4).
    If a variable does not have a 'Sem informação' code, out-of-domain values
    are set directly to NaN.
    """
    with open(dict_path, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    replaced_info = {}
    for col, spec in dictionary.items():
        if col not in df.columns:
            continue
        vals = spec.get("valores")
        if not isinstance(vals, dict):
            continue

        valid_keys = set(str(k).strip() for k in vals.keys())

        non_info_code = None
        fallback_code = None
        for k, label in vals.items():
            lbl_norm = _normalize_text(label)
            if lbl_norm == "sem informacao":
                non_info_code = str(k).strip()
                break
            if _is_non_informative_label(label):
                if fallback_code is None:
                    fallback_code = str(k).strip()

        if non_info_code is None:
            non_info_code = fallback_code

        valid_nums = []
        for k in valid_keys:
            try:
                valid_nums.append(float(k))
            except ValueError:
                pass

        series_str = df[col].astype("string").str.strip()
        series_num = pd.to_numeric(series_str, errors="coerce")
        
        mask_valid_str = series_str.isin(valid_keys)
        # Avoid treating empty strings as matching if valid_keys doesn't explicitly have it
        mask_valid_num = series_num.isin(valid_nums) & series_str.notna() & (series_str != "")
        
        is_invalid = df[col].notna() & ~(mask_valid_str | mask_valid_num)
        
        n_invalid = is_invalid.sum()
        if n_invalid > 0:
            if non_info_code is not None:
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        df.loc[is_invalid, col] = float(non_info_code)
                    except ValueError:
                        df.loc[is_invalid, col] = non_info_code
                else:
                    # Assign as string
                    # If column implies integer (e.g., object types that might mix str and int)
                    df.loc[is_invalid, col] = non_info_code
            else:
                df.loc[is_invalid, col] = np.nan
            replaced_info[col] = int(n_invalid)

    return replaced_info


def prepare_data(df: "pd.DataFrame", config_path: str = "config/config.yaml", cfg: dict = None) -> None:
    if cfg is None:
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

    # Keep insertion order while preventing duplicated columns when a feature is
    # listed in both safe and high-cardinality config groups.
    all_features = list(dict.fromkeys(data_cfg["features_safe"] + data_cfg["features_high_cardinality"]))
    all_features = [f for f in all_features if f in df.columns]

    # BUG-5 fix: apply features_exclude from config (previously this key was
    # defined in config.yaml but never read by the code).
    features_exclude = set(data_cfg.get("features_exclude", []))
    if features_exclude:
        excluded_here = [f for f in all_features if f in features_exclude]
        if excluded_here:
            log.info("Excluding features from config (data.features_exclude): %s", excluded_here)
        all_features = [f for f in all_features if f not in features_exclude]

    dict_path = Path(config_path).resolve().parent / "dicionario_valores_validos.json"
    missing_rules = _load_missing_rules(dict_path)

    mr_raw = _missing_report(df[all_features], missing_rules=missing_rules)
    mr_raw.to_csv(out_tables / "missing_report_raw.csv")
    meta["missing_report_raw_rows"] = int(len(df))
    log.info("Raw missing report generated for %d features", len(all_features))

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

    out_of_domain_replaced = _enforce_dictionary_domain(df, dict_path)
    if out_of_domain_replaced:
        log.info("Enforced dictionary domain (set OOV to non-informative code) in %d columns", len(out_of_domain_replaced))
    meta["out_of_domain_replaced"] = out_of_domain_replaced

    sem_info_replaced = _replace_missing_with_nan(df, missing_rules)
    if sem_info_replaced:
        log.info("Replaced non-informative dictionary values with NaN in %d columns", len(sem_info_replaced))
    if "BASDIAGSP" in missing_rules:
        log.info("BASDIAGSP rules treated as NaN: %s", missing_rules["BASDIAGSP"])
    if "OCUPACAO" in missing_rules:
        log.info("OCUPACAO rules treated as NaN: %s", missing_rules["OCUPACAO"])
    meta["sem_info_replaced_by_column"] = sem_info_replaced

    target_col = data_cfg["target_col"]
    log.info(
        "Target distribution before filter (%s):\n%s",
        target_col,
        df[target_col].value_counts(dropna=False).sort_index(),
    )
    
    # 1. Clean the target column to group granular staging (e.g. 1A -> 1, 2B -> 2)
    # The logic keeps the first digit if the value starts with 0-9.
    # We must explicitly protect special codes like '88' and '99' and '00'-'09'.
    target_str = df[target_col].astype("string").str.strip().str.upper()
    grouped_target = target_str.copy()
    
    # Extract the first character for values that start with a digit (but only length > 1 if not special)
    # We map '1A', '1B', etc. to '1', '2A' to '2', etc.
    # Preserve double-digit codes like '88', '99', '00', '01'
    is_multidigit_num = target_str.str.match(r"^\d{2}$", na=False)
    has_first_digit = target_str.str.match(r"^\d.*", na=False)
    
    # Apply grouping: if it starts with a digit but is NOT exactly 2 digits
    # (or if we want to be safe, length > 1 but not matching exactly two digits 88, 99)
    mask_to_group = has_first_digit & ~is_multidigit_num
    grouped_target.loc[mask_to_group] = target_str.loc[mask_to_group].str[0]
    
    # Convert '00', '01', '02', '03', '04' to '0', '1', '2', '3', '4'
    is_zero_padded = target_str.str.match(r"^0\d$", na=False)
    grouped_target.loc[is_zero_padded] = target_str.loc[is_zero_padded].str[1]
    
    # Reassign the grouped column
    df[target_col] = grouped_target
    
    # 2. Convert to numeric, letting non-numeric stuff become NaN
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    
    # 3. Filter only valid clinical staging classes
    n0 = len(df)
    df = df[df[target_col].isin(data_cfg["valid_classes"])].copy()
    log.info("Target filter (after grouping granular staging): %s -> %s", f"{n0:,}", f"{len(df):,}")
    meta["n_after_target_filter"] = len(df)

    features = [f for f in all_features if f in df.columns]
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

    ordered_features = num_cols + cat_cols
    df_raw = df[ordered_features].copy()

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
        idx = np.arange(len(df))
        idx_sample, _ = train_test_split(
            idx,
            train_size=n_sample,
            stratify=df[target_col],
            random_state=seed,
        )
        idx_sample = np.sort(idx_sample)
        df = df.iloc[idx_sample].copy()
        df_raw = df_raw.iloc[idx_sample].copy()
        log.info("Stratified sample used: %s", f"{len(df):,}")

    X = df[ordered_features].reset_index(drop=True)
    X_raw = df_raw[ordered_features].reset_index(drop=True)
    for col in cat_cols:
        # CatBoost baseline without encoding requires string/integer categories.
        # Represent missing categorical values explicitly instead of numerical encoding.
        X_raw[col] = X_raw[col].astype("string").fillna("__MISSING__")
    y = df[target_col].reset_index(drop=True)

    assert y.isna().sum() == 0, "Target has NaN values"
    log.info("Final X: %s | y: %s | missing in X: %s", X.shape, y.shape, f"{int(X.isna().sum().sum()):,}")

    X.to_parquet(out_proc / "X_prepared.parquet", index=False)
    X_raw.to_parquet(out_proc / "X_raw_prepared.parquet", index=False)
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
            "raw_cat_missing_token": "__MISSING__",
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
