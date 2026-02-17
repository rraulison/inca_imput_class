"""
Shared statistical utility functions for inferential analyses.

Extracted from run_imputation_effect_stats.py and run_ordinal_sensitivity.py
to eliminate code duplication.
"""

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.random import Generator
from scipy.stats import rankdata, t, wilcoxon


def bootstrap_mean_ci(
    diff: np.ndarray,
    n_boot: int,
    ci_level: float,
    rng: Generator,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the mean of ``diff``."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(diff[0]), float(diff[0]))

    idx = rng.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    alpha = 1.0 - ci_level
    low = float(np.quantile(means, alpha / 2.0))
    high = float(np.quantile(means, 1.0 - alpha / 2.0))
    return (low, high)


def wilcoxon_two_sided(diff: np.ndarray) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test, ignoring zero differences."""
    diff = np.asarray(diff, dtype=float)
    nz = diff[np.abs(diff) > 1e-15]
    if len(nz) == 0:
        return (np.nan, 1.0)
    try:
        stat, pval = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return (float(stat), float(pval))
    except ValueError:
        return (np.nan, 1.0)


def cohen_dz(diff: np.ndarray) -> float:
    """Cohen's d_z for paired samples."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return np.nan
    sd = np.std(diff, ddof=1)
    mean = np.mean(diff)
    if sd <= 1e-15:
        if abs(mean) <= 1e-15:
            return 0.0
        return float(np.sign(mean) * np.inf)
    return float(mean / sd)


def rank_biserial(diff: np.ndarray) -> float:
    """Rank-biserial correlation for Wilcoxon signed-rank test."""
    diff = np.asarray(diff, dtype=float)
    nz = diff[np.abs(diff) > 1e-15]
    n = len(nz)
    if n == 0:
        return np.nan
    ranks = rankdata(np.abs(nz), method="average")
    r_plus = float(ranks[nz > 0].sum())
    r_minus = float(ranks[nz < 0].sum())
    denom = n * (n + 1) / 2.0
    if denom == 0:
        return np.nan
    return float((r_plus - r_minus) / denom)


def equivalence_and_noninferiority(
    diff: np.ndarray,
    margin: float,
    alpha: float,
) -> dict:
    """TOST equivalence test and one-sided non-inferiority test."""
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    mean = float(np.mean(diff)) if n else np.nan

    if n == 0 or margin <= 0:
        return {
            "tost_p_lower": np.nan,
            "tost_p_upper": np.nan,
            "tost_pvalue": np.nan,
            "equivalent": False,
            "noninferiority_p": np.nan,
            "noninferior": False,
        }

    if n == 1:
        lower_ok = mean > -margin
        upper_ok = mean < margin
        p_lower = 0.0 if lower_ok else 1.0
        p_upper = 0.0 if upper_ok else 1.0
        return {
            "tost_p_lower": p_lower,
            "tost_p_upper": p_upper,
            "tost_pvalue": max(p_lower, p_upper),
            "equivalent": bool(lower_ok and upper_ok),
            "noninferiority_p": p_lower,
            "noninferior": bool(lower_ok),
        }

    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-15:
        lower_ok = mean > -margin
        upper_ok = mean < margin
        p_lower = 0.0 if lower_ok else 1.0
        p_upper = 0.0 if upper_ok else 1.0
        return {
            "tost_p_lower": p_lower,
            "tost_p_upper": p_upper,
            "tost_pvalue": max(p_lower, p_upper),
            "equivalent": bool(lower_ok and upper_ok),
            "noninferiority_p": p_lower,
            "noninferior": bool(p_lower < alpha),
        }

    se = sd / math.sqrt(n)
    dof = n - 1

    t_lower = (mean + margin) / se
    p_lower = 1.0 - t.cdf(t_lower, dof)

    t_upper = (mean - margin) / se
    p_upper = t.cdf(t_upper, dof)

    return {
        "tost_p_lower": float(p_lower),
        "tost_p_upper": float(p_upper),
        "tost_pvalue": float(max(p_lower, p_upper)),
        "equivalent": bool((p_lower < alpha) and (p_upper < alpha)),
        "noninferiority_p": float(p_lower),
        "noninferior": bool(p_lower < alpha),
    }


def holm_adjust(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction for multiple comparisons."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan, dtype=float)
    valid = np.isfinite(pvalues)
    if not valid.any():
        return adjusted

    pv = pvalues[valid]
    order = np.argsort(pv)
    sorted_p = pv[order]
    m = len(sorted_p)

    scaled = np.array([(m - i) * p for i, p in enumerate(sorted_p)], dtype=float)
    scaled = np.maximum.accumulate(scaled)
    scaled = np.clip(scaled, 0.0, 1.0)

    unsorted = np.empty_like(scaled)
    unsorted[order] = scaled
    adjusted[valid] = unsorted
    return adjusted


def fmt_time(seconds: float) -> str:
    """Format elapsed time as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"
