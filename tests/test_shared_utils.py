"""Basic smoke tests for shared utility modules."""

import numpy as np
import pytest

from src.stats_utils import (
    bootstrap_mean_ci,
    cohen_dz,
    equivalence_and_noninferiority,
    fmt_time,
    holm_adjust,
    rank_biserial,
    wilcoxon_two_sided,
)
from src.metrics_utils import (
    coerce_confusion_matrix,
    compute_metrics,
    one_hot_proba,
    serialize,
    to_numpy,
)


# ─── stats_utils ───

class TestFmtTime:
    def test_seconds(self):
        assert fmt_time(30.0) == "30.0s"

    def test_minutes(self):
        assert fmt_time(120.0) == "2.0min"

    def test_hours(self):
        assert fmt_time(7200.0) == "2.0h"


class TestBootstrapMeanCI:
    def test_basic(self):
        rng = np.random.default_rng(42)
        diff = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        lo, hi = bootstrap_mean_ci(diff, n_boot=1000, ci_level=0.95, rng=rng)
        assert lo < hi
        assert lo < np.mean(diff) < hi

    def test_empty(self):
        rng = np.random.default_rng(42)
        lo, hi = bootstrap_mean_ci(np.array([]), n_boot=100, ci_level=0.95, rng=rng)
        assert np.isnan(lo) and np.isnan(hi)


class TestWilcoxon:
    def test_significant(self):
        stat, pval = wilcoxon_two_sided(np.array([1, 2, 3, 4, 5], dtype=float))
        assert pval < 0.10

    def test_zeros(self):
        stat, pval = wilcoxon_two_sided(np.array([0.0, 0.0, 0.0]))
        assert pval == 1.0


class TestCohenDz:
    def test_basic(self):
        d = cohen_dz(np.array([1.0, 1.0, 1.0, 1.0]))
        assert np.isinf(d)  # sd=0, mean>0

    def test_small(self):
        d = cohen_dz(np.array([0.1, 0.2, 0.0, 0.1]))
        assert 0 < d < 10


class TestHolmAdjust:
    def test_basic(self):
        pvals = np.array([0.01, 0.04, 0.03])
        adj = holm_adjust(pvals)
        assert adj[0] == pytest.approx(0.03, abs=1e-6)
        assert all(adj <= 1.0)


class TestEquivalence:
    def test_equivalent(self):
        diff = np.array([0.001, 0.002, -0.001, 0.0, 0.001])
        result = equivalence_and_noninferiority(diff, margin=0.05, alpha=0.05)
        assert result["equivalent"] is True


# ─── metrics_utils ───

class TestToNumpy:
    def test_list(self):
        r = to_numpy([1, 2, 3])
        assert isinstance(r, np.ndarray)


class TestOneHotProba:
    def test_basic(self):
        pred = np.array([0, 1, 2])
        classes = np.array([0, 1, 2])
        p = one_hot_proba(pred, classes)
        assert p.shape == (3, 3)
        assert np.all(p.sum(axis=1) == 1.0)


class TestComputeMetrics:
    def test_basic(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 2])
        classes = np.array([0, 1, 2])
        y_prob = one_hot_proba(y_pred, classes)
        m = compute_metrics(y_true, y_pred, y_prob, classes)
        assert 0 <= m["accuracy"] <= 1.0
        assert "confusion_matrix" in m


class TestSerialize:
    def test_numpy_types(self):
        obj = {"a": np.int64(1), "b": np.float64(2.5), "c": np.array([1, 2])}
        s = serialize(obj)
        assert s["a"] == 1
        assert s["b"] == 2.5
        assert s["c"] == [1, 2]


class TestCoerceConfusionMatrix:
    def test_list(self):
        cm = coerce_confusion_matrix([[10, 2], [3, 15]])
        assert cm.shape == (2, 2)

    def test_string(self):
        cm = coerce_confusion_matrix("[[10, 2], [3, 15]]")
        assert cm.shape == (2, 2)

    def test_flat(self):
        cm = coerce_confusion_matrix([10, 2, 3, 15])
        assert cm.shape == (2, 2)

    def test_invalid(self):
        cm = coerce_confusion_matrix([1, 2, 3])  # not a perfect square
        assert cm is None
