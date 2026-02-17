import numpy as np
import importlib
import sys
import types
from pathlib import Path


def _load_run_classification():
    if "xgboost" not in sys.modules:
        xgb = types.ModuleType("xgboost")

        class _DummyXGBClassifier:  # pragma: no cover - only used when xgboost is missing
            def __init__(self, *args, **kwargs):
                pass

        xgb.XGBClassifier = _DummyXGBClassifier
        sys.modules["xgboost"] = xgb

    return importlib.import_module("src.run_classification")


def test_prune_checkpoint_filters_to_planned_tasks_and_deduplicates():
    mod = _load_run_classification()
    ckpt = {
        "completed": [
            "default__Media__0__XGBoost",
            "default__Media__1__XGBoost",
            "default__MICE__0__XGBoost",
            "hybrid__Media__0__XGBoost",
        ],
        "results": [
            {"runtime_mode": "default", "imputer": "Media", "fold": 0, "classifier": "XGBoost", "score": 0.10},
            {"runtime_mode": "default", "imputer": "Media", "fold": 0, "classifier": "XGBoost", "score": 0.20},
            {"runtime_mode": "default", "imputer": "MICE", "fold": 0, "classifier": "XGBoost", "score": 0.30},
            {"runtime_mode": "hybrid", "imputer": "Media", "fold": 0, "classifier": "XGBoost", "score": 0.40},
        ],
    }
    planned = {"default__Media__0__XGBoost"}

    completed, rows, dropped_completed, dropped_results = mod._prune_checkpoint_for_tasks(
        ckpt=ckpt,
        runtime_mode="default",
        planned_keys=planned,
    )

    assert completed == planned
    assert len(rows) == 1
    assert rows[0]["score"] == 0.20
    assert dropped_completed == 2
    assert dropped_results == 2


def test_resolve_classes_prefers_metadata_target_mapping():
    mod = _load_run_classification()
    meta = {"target_mapping": {"0": 0, "1": "1", "88": 2}, "n_classes": 99}
    classes, source = mod._resolve_classes(meta=meta, imp_dir=Path("."))

    assert source == "metadata.target_mapping"
    assert np.array_equal(classes, np.array([0, 1, 2], dtype=int))


def test_resolve_classes_falls_back_to_n_classes():
    mod = _load_run_classification()
    meta = {"n_classes": "4"}
    classes, source = mod._resolve_classes(meta=meta, imp_dir=Path("."))

    assert source == "metadata.n_classes"
    assert np.array_equal(classes, np.array([0, 1, 2, 3], dtype=int))
