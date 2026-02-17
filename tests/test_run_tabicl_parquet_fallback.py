import pandas as pd
import pytest

import src.run_tabicl as run_tabicl


def test_read_parquet_safe_reports_original_error_without_helper(monkeypatch):
    def _fail_pandas(*args, **kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(run_tabicl.pd, "read_parquet", _fail_pandas)

    import pyarrow.parquet as pq

    def _fail_pyarrow(*args, **kwargs):
        raise RuntimeError("fallback failed")

    monkeypatch.setattr(pq, "read_table", _fail_pyarrow)
    monkeypatch.delenv("TABICL_PARQUET_HELPER_PYTHON", raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        run_tabicl._read_parquet_safe("dummy.parquet")

    message = str(exc_info.value)
    assert "Original: Repetition level histogram size mismatch" in message
    assert "PyArrow fallback: fallback failed" in message


def test_read_parquet_safe_uses_helper_python_fallback(monkeypatch):
    def _fail_pandas(*args, **kwargs):
        raise OSError("Repetition level histogram size mismatch")

    monkeypatch.setattr(run_tabicl.pd, "read_parquet", _fail_pandas)

    import pyarrow.parquet as pq

    def _fail_pyarrow(*args, **kwargs):
        raise RuntimeError("fallback failed")

    monkeypatch.setattr(pq, "read_table", _fail_pyarrow)
    monkeypatch.setenv("TABICL_PARQUET_HELPER_PYTHON", "/opt/rapids-25.08/bin/python")

    calls = {}

    def _helper(path, columns=None):
        calls["path"] = str(path)
        calls["columns"] = columns
        return pd.DataFrame({"target": [1]})

    monkeypatch.setattr(run_tabicl, "_read_parquet_via_helper_python", _helper)

    df = run_tabicl._read_parquet_safe("dummy.parquet", columns=["target"])
    assert list(df.columns) == ["target"]
    assert calls["path"].endswith("dummy.parquet")
    assert calls["columns"] == ["target"]
