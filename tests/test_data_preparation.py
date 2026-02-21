import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_preparation import (
    _clean_target_column,
    _replace_missing_with_nan,
    prepare_data,
)


def _write_dict_file(config_path):
    dictionary = {
        "SEXO": {
            "valores": {
                "1": "Masculino",
                "2": "Feminino",
                "9": "Sem informacao",
            }
        },
        "OCUPACAO": {"valores": "Mais de tres 9 ocupacao ignorada"},
    }
    dict_path = config_path.parent / "dicionario_valores_validos.json"
    dict_path.write_text(json.dumps(dictionary), encoding="utf-8")


def _base_cfg(tmp_path):
    return {
        "experiment": {"random_seed": 42, "n_sample": 999_999},
        "data": {
            "target_col": "ESTADIAM",
            "valid_classes": [1, 2, 3, 4, 88],
            "missing_threshold": 0.99,
            "date_filter_col": "DATAPRICON",
            "year_min": 2013,
            "year_max": 2023,
            "high_cardinality_min_freq": 1,
            "features_safe": ["SEXO", "IDADE", "OCUPACAO"],
            "features_high_cardinality": ["OCUPACAO"],
            "num_cols": ["IDADE"],
            "features_exclude": [],
        },
        "paths": {
            "processed_data": str(tmp_path / "data" / "processed"),
            "results_tables": str(tmp_path / "results" / "tables"),
            "imputed_data": str(tmp_path / "data" / "imputed"),
            "results_raw": str(tmp_path / "results" / "raw"),
            "results_figures": str(tmp_path / "results" / "figures"),
            "raw_data": str(tmp_path / "data" / "raw"),
        },
    }


def test_replace_missing_with_nan_applies_code_and_regex_rules():
    df = pd.DataFrame(
        {
            "OCUPACAO": ["1111", "9999", "2", np.nan, "8888"],
        }
    )
    rules = {"OCUPACAO": {"codes": ["2"], "regex": [r"9{4,}"]}}

    replaced = _replace_missing_with_nan(df, rules)

    assert replaced == {"OCUPACAO": 2}
    assert int(df["OCUPACAO"].isna().sum()) == 3


def test_clean_target_column_groups_granular_staging_and_filters_invalid():
    df = pd.DataFrame({"ESTADIAM": ["1A", "02", "3", "4B", "88", "XX"]})

    out = _clean_target_column(df, target_col="ESTADIAM", valid_classes=[1, 2, 3, 4, 88])

    assert sorted(out["ESTADIAM"].astype(int).tolist()) == [1, 2, 3, 4, 88]


def test_prepare_data_keeps_categorical_raw_and_saves_temporal_reference(tmp_path, monkeypatch):
    def _fake_to_parquet(self, path, index=False):
        _ = index
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.to_pickle(out)

    monkeypatch.setattr(pd.DataFrame, "to_parquet", _fake_to_parquet)

    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment: {}\n", encoding="utf-8")
    _write_dict_file(config_path)

    cfg = _base_cfg(tmp_path)
    df = pd.DataFrame(
        {
            "DATAPRICON": [
                "01/01/2019",
                "01/01/2020",
                "01/01/2021",
                "01/01/2022",
                "01/01/2018",
            ],
            "ESTADIAM": [1, 2, 1, 2, 1],
            "SEXO": ["1", "2", "1", "2", "1"],
            "IDADE": [30, 40, 50, 60, 70],
            "OCUPACAO": ["1111", "9999", "1111", "2222", "3333"],
        }
    )

    X, y, meta = prepare_data(df, config_path=str(config_path), cfg=cfg)

    assert list(X.columns) == ["IDADE", "SEXO", "OCUPACAO"]
    assert X["SEXO"].dtype.name in {"object", "string"}
    assert set(X["SEXO"].dropna().unique().tolist()) <= {"1", "2"}

    temporal_ref_path = tmp_path / "data" / "processed" / "temporal_reference.parquet"
    assert temporal_ref_path.exists()

    years = pd.read_pickle(temporal_ref_path)["year"]
    assert len(years) == len(X) == len(y)
    assert years.between(2013, 2023, inclusive="both").all()

    assert meta["categorical_encoding_strategy"] == "train_only_per_fold"
    assert meta["temporal_reference_file"] == "temporal_reference.parquet"

    with open(tmp_path / "data" / "processed" / "encoders.pkl", "rb") as f:
        enc = pickle.load(f)
    assert enc["feature_encoders"] == {}
    assert enc["categorical_encoding_strategy"] == "train_only_per_fold"
