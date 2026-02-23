"""Tests for the confirmatory protocol pipeline."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.stats_utils import permutation_test_paired


# ─── Permutation Test ───


class TestPermutationTestPaired:
    def test_known_large_effect(self):
        """Large positive effect should yield p < 0.05."""
        rng = np.random.default_rng(42)
        a = rng.normal(1.0, 0.1, size=25)  # treatment
        b = rng.normal(0.0, 0.1, size=25)  # control
        mean_diff, pval = permutation_test_paired(a, b, n_perms=5000, seed=42)
        assert mean_diff > 0.5
        assert pval < 0.05

    def test_no_effect(self):
        """No effect should yield p > 0.10."""
        rng = np.random.default_rng(123)
        a = rng.normal(0.0, 0.1, size=25)
        b = rng.normal(0.0, 0.1, size=25)
        # Use same seed to make test deterministic
        mean_diff, pval = permutation_test_paired(a, b, n_perms=5000, seed=99)
        assert abs(mean_diff) < 0.15
        assert pval > 0.10

    def test_empty_input(self):
        mean_diff, pval = permutation_test_paired(np.array([]), np.array([]))
        assert np.isnan(mean_diff)
        assert np.isnan(pval)

    def test_mismatched_length(self):
        mean_diff, pval = permutation_test_paired(np.array([1, 2]), np.array([1]))
        assert np.isnan(mean_diff)

    def test_reproducibility(self):
        """Same seed should produce same p-value."""
        a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        b = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
        _, p1 = permutation_test_paired(a, b, n_perms=1000, seed=42)
        _, p2 = permutation_test_paired(a, b, n_perms=1000, seed=42)
        assert p1 == p2


# ─── Splits Reproducibility ───


class TestSplitsReproducibility:
    def test_same_seed_same_splits(self):
        """Same seed should produce identical stratified splits."""
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2] * 5)
        seed = 42

        skf1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        skf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

        splits1 = list(skf1.split(y, y))
        splits2 = list(skf2.split(y, y))

        for (tr1, te1), (tr2, te2) in zip(splits1, splits2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)

    def test_different_seed_different_splits(self):
        """Different seeds should produce different splits."""
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2] * 5)

        skf1 = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        skf2 = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)

        splits1 = list(skf1.split(y, y))
        splits2 = list(skf2.split(y, y))

        # At least one fold should differ
        any_different = False
        for (tr1, _), (tr2, _) in zip(splits1, splits2):
            if not np.array_equal(tr1, tr2):
                any_different = True
                break
        assert any_different


# ─── No-Leakage Validation ───


class TestNoLeakage:
    def test_imputer_fit_on_train_only(self):
        """Verify that imputer fitted on train does not see test data.

        We create a dataset where test has a unique marker value.
        If imputer leaks test data, the marker would influence train imputation.
        """
        from sklearn.impute import SimpleImputer

        rng = np.random.default_rng(42)
        n_train, n_test, n_feat = 100, 50, 5

        X_train = rng.normal(0, 1, size=(n_train, n_feat))
        X_test = rng.normal(0, 1, size=(n_test, n_feat))

        # Inject NaN in train
        X_train[0, 0] = np.nan
        # Set test col 0 to extreme value (1000) — should NOT affect train imputation
        X_test[:, 0] = 1000.0

        # Fit imputer on train only (correct protocol)
        imp = SimpleImputer(strategy="mean")
        imp.fit(X_train)
        X_train_imputed = imp.transform(X_train)

        # The imputed value should be the train mean, NOT influenced by test's 1000
        train_mean_col0 = np.nanmean(X_train[:, 0])
        assert abs(X_train_imputed[0, 0] - train_mean_col0) < 0.01
        assert X_train_imputed[0, 0] < 10  # Far from 1000


# ─── T1: ConfirmatoryPipeline inner-fold leakage test ───


class TestConfirmatoryPipelineLeakage:
    """T1: Verify ConfirmatoryPipeline re-fits all stages on each inner fold."""

    def _make_dataset(self, n=200, n_cat=2, n_num=3, seed=42):
        rng = np.random.default_rng(seed)
        num_data = {f"num_{i}": rng.normal(0, 1, n) for i in range(n_num)}
        cat_data = {f"cat_{i}": rng.choice(["a", "b", "c"], n).astype(str) for i in range(n_cat)}
        y = rng.integers(0, 3, n)
        # Introduce missings in numeric cols
        for col in list(num_data)[:2]:
            idx = rng.choice(n, n // 5, replace=False)
            num_data[col] = num_data[col].astype(float)
            num_data[col][idx] = np.nan
        df = pd.DataFrame({**num_data, **cat_data})
        return df, y.astype(int), list(num_data), list(cat_data)

    def test_imputer_refit_per_fold(self):
        """ConfirmatoryPipeline.fit() on different subsets learns different imputer statistics."""
        from sklearn.impute import SimpleImputer
        from sklearn.tree import DecisionTreeClassifier
        from src.confirmatory_pipeline import ConfirmatoryPipeline

        df, y, num_cols, cat_cols = self._make_dataset(n=300)

        fold1_idx = np.arange(0, 150)
        fold2_idx = np.arange(150, 300)

        pipe1 = ConfirmatoryPipeline(
            clf=DecisionTreeClassifier(random_state=0),
            imputer=SimpleImputer(strategy="mean"),
            num_cols=num_cols, cat_cols=cat_cols,
        )
        pipe1.fit(df.iloc[fold1_idx].reset_index(drop=True), y[fold1_idx])

        pipe2 = ConfirmatoryPipeline(
            clf=DecisionTreeClassifier(random_state=0),
            imputer=SimpleImputer(strategy="mean"),
            num_cols=num_cols, cat_cols=cat_cols,
        )
        pipe2.fit(df.iloc[fold2_idx].reset_index(drop=True), y[fold2_idx])

        # Imputer statistics must differ (different subsets have different nan-masked means)
        stats1 = pipe1._imp_fitted.statistics_
        stats2 = pipe2._imp_fitted.statistics_
        # At least one feature should have different learned mean
        assert not np.allclose(stats1, stats2, atol=1e-6), (
            "Imputer statistics identical across folds — re-fit may not be working."
        )

    def test_pipeline_predict_proba_shape(self):
        """ConfirmatoryPipeline.predict_proba() should output (n_samples, n_classes)."""
        from sklearn.impute import SimpleImputer
        from sklearn.tree import DecisionTreeClassifier
        from src.confirmatory_pipeline import ConfirmatoryPipeline

        df, y, num_cols, cat_cols = self._make_dataset(n=200)
        pipe = ConfirmatoryPipeline(
            clf=DecisionTreeClassifier(random_state=0),
            imputer=SimpleImputer(strategy="mean"),
            num_cols=num_cols, cat_cols=cat_cols,
        )
        pipe.fit(df.iloc[:150].reset_index(drop=True), y[:150])
        proba = pipe.predict_proba(df.iloc[150:].reset_index(drop=True))
        n_classes = len(np.unique(y[:150]))
        assert proba.shape == (50, n_classes)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


# ─── T2: Checkpoint signature invalidation test ───


class TestCheckpointSignature:
    """T2: Verify that a checkpoint with divergent hash is discarded."""

    def test_divergent_signature_discards_checkpoint(self):
        """A checkpoint saved with one signature must be ignored when signature changes."""
        from src.run_protocol import _load_checkpoint, _save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.json"

            # Save a checkpoint with signature A
            old_results = [{"repeat": 0, "outer_fold": 0, "imputer": "Media", "f1_weighted": 0.9}]
            old_completed = {"0__0__Media__XGBoost"}
            _save_checkpoint(ckpt_path, old_completed, old_results, signature="sig_A")

            # Load with a DIFFERENT signature (config/data changed)
            ckpt_new = _load_checkpoint(ckpt_path, current_sig="sig_B")

            # Must be empty: old results must NOT contaminate new run
            assert len(ckpt_new["completed"]) == 0, "Old completed tasks leaked into new run."
            assert len(ckpt_new["results"]) == 0, "Old results leaked into new run."

    def test_matching_signature_loads_checkpoint(self):
        """A checkpoint with a matching signature must be resumed correctly."""
        from src.run_protocol import _load_checkpoint, _save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "ckpt.json"

            old_results = [{"repeat": 0, "outer_fold": 0, "imputer": "Media", "f1_weighted": 0.9}]
            old_completed = {"0__0__Media__XGBoost"}
            _save_checkpoint(ckpt_path, old_completed, old_results, signature="sig_stable")

            ckpt = _load_checkpoint(ckpt_path, current_sig="sig_stable")

            assert "0__0__Media__XGBoost" in ckpt["completed"]
            assert len(ckpt["results"]) == 1
