#!/usr/bin/env python3
"""
Inferential analysis focused on the impact of imputation on classification performance.

This script does not rerun models. It reads fold-level results from:
    results/raw/all_results.csv

Main outputs:
    - pairwise_global_<metric>.csv
    - pairwise_by_classifier_<metric>.csv
    - baseline_global_<metric>.csv
    - baseline_by_classifier_<metric>.csv
    - manifest_<metric>.json
"""

import argparse
import json
import logging
import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, t, wilcoxon

try:
    from src.config_loader import load_config
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config_loader import load_config

log = logging.getLogger(__name__)


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def _split_csv_list(value):
    if value is None:
        return None
    items = [x.strip() for x in str(value).split(",") if x.strip()]
    return items or None


def _validate_columns(df, columns, source):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {source}: {missing}")


def _bootstrap_mean_ci(diff, n_boot, ci_level, rng):
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


def _wilcoxon_two_sided(diff):
    diff = np.asarray(diff, dtype=float)
    nz = diff[np.abs(diff) > 1e-15]
    if len(nz) == 0:
        return (np.nan, 1.0)
    try:
        stat, pval = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return (float(stat), float(pval))
    except ValueError:
        return (np.nan, 1.0)


def _cohen_dz(diff):
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


def _rank_biserial(diff):
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


def _equivalence_and_noninferiority(diff, margin, alpha):
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


def _holm_adjust(pvalues):
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


def _build_pivot(df, metric, index_cols):
    use_cols = list(index_cols) + ["imputer", metric]
    dfx = df[use_cols].dropna(subset=[metric]).copy()
    pivot = dfx.pivot_table(index=list(index_cols), columns="imputer", values=metric, aggfunc="first")
    return pivot.sort_index()


def _pairwise_stats(pivot, metric, alpha, equivalence_margin, bootstrap_iters, ci_level, rng):
    imputers = sorted([c for c in pivot.columns if pivot[c].notna().any()])
    rows = []
    for a, b in combinations(imputers, 2):
        pair = pivot[[a, b]].dropna()
        if pair.empty:
            continue
        diff = (pair[a] - pair[b]).to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(diff, bootstrap_iters, ci_level, rng)
        stat, pval = _wilcoxon_two_sided(diff)
        eq = _equivalence_and_noninferiority(diff, equivalence_margin, alpha)

        rows.append(
            {
                "metric": metric,
                "imputer_a": a,
                "imputer_b": b,
                "n_pairs": int(len(diff)),
                "mean_a": float(pair[a].mean()),
                "mean_b": float(pair[b].mean()),
                "delta_mean": float(np.mean(diff)),
                "delta_median": float(np.median(diff)),
                "delta_std": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wilcoxon_stat": stat,
                "p_wilcoxon": pval,
                "cohen_dz": _cohen_dz(diff),
                "rank_biserial": _rank_biserial(diff),
                "tost_p_lower": eq["tost_p_lower"],
                "tost_p_upper": eq["tost_p_upper"],
                "tost_pvalue": eq["tost_pvalue"],
                "equivalent": eq["equivalent"],
                "noninferiority_p": eq["noninferiority_p"],
                "noninferior": eq["noninferior"],
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_wilcoxon_holm"] = _holm_adjust(out["p_wilcoxon"].to_numpy(dtype=float))
        out["significant_holm"] = out["p_wilcoxon_holm"] < alpha
        out = out.sort_values(["p_wilcoxon_holm", "delta_mean"], ascending=[True, False]).reset_index(drop=True)
    return out


def _baseline_stats(pivot, metric, baseline, alpha, equivalence_margin, bootstrap_iters, ci_level, rng):
    if baseline not in pivot.columns:
        return pd.DataFrame()

    rows = []
    for imp in sorted([c for c in pivot.columns if c != baseline]):
        pair = pivot[[imp, baseline]].dropna()
        if pair.empty:
            continue

        diff = (pair[imp] - pair[baseline]).to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(diff, bootstrap_iters, ci_level, rng)
        stat, pval = _wilcoxon_two_sided(diff)
        eq = _equivalence_and_noninferiority(diff, equivalence_margin, alpha)

        rows.append(
            {
                "metric": metric,
                "imputer": imp,
                "baseline": baseline,
                "n_pairs": int(len(diff)),
                "mean_imputer": float(pair[imp].mean()),
                "mean_baseline": float(pair[baseline].mean()),
                "delta_mean": float(np.mean(diff)),
                "delta_median": float(np.median(diff)),
                "delta_std": float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "wilcoxon_stat": stat,
                "p_wilcoxon": pval,
                "cohen_dz": _cohen_dz(diff),
                "rank_biserial": _rank_biserial(diff),
                "tost_p_lower": eq["tost_p_lower"],
                "tost_p_upper": eq["tost_p_upper"],
                "tost_pvalue": eq["tost_pvalue"],
                "equivalent": eq["equivalent"],
                "noninferiority_p": eq["noninferiority_p"],
                "noninferior": eq["noninferior"],
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["p_wilcoxon_holm"] = _holm_adjust(out["p_wilcoxon"].to_numpy(dtype=float))
        out["significant_holm"] = out["p_wilcoxon_holm"] < alpha
        out = out.sort_values(["p_wilcoxon_holm", "delta_mean"], ascending=[True, False]).reset_index(drop=True)
    return out


def run_imputation_effect_stats(
    config_path="config/config.yaml",
    input_csv=None,
    output_dir=None,
    metric="f1_weighted",
    alpha=0.05,
    equivalence_margin=0.005,
    baseline="NoImpute",
    bootstrap_iters=5000,
    ci_level=0.95,
    seed=42,
    include_imputers=None,
    exclude_imputers=None,
):
    cfg = load_config(config_path)
    results_csv = Path(input_csv) if input_csv else Path(cfg["paths"]["results_raw"]) / "all_results.csv"
    out_dir = Path(output_dir) if output_dir else Path(cfg["paths"]["results_tables"]) / "imputation_effect"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_csv.exists():
        raise FileNotFoundError(f"Input file not found: {results_csv}")

    df = pd.read_csv(results_csv)
    _validate_columns(df, ["fold", "classifier", "imputer", metric], source=str(results_csv))

    if "error" in df.columns:
        df = df[~df["error"].notna()].copy()
    df = df.dropna(subset=[metric]).copy()

    if include_imputers:
        include_set = set(include_imputers)
        df = df[df["imputer"].isin(include_set)].copy()
    if exclude_imputers:
        exclude_set = set(exclude_imputers)
        df = df[~df["imputer"].isin(exclude_set)].copy()

    if df.empty:
        raise ValueError("No rows available after filtering.")

    rng = np.random.default_rng(seed)

    pivot_global = _build_pivot(df, metric=metric, index_cols=["classifier", "fold"])
    pair_global = _pairwise_stats(
        pivot_global,
        metric=metric,
        alpha=alpha,
        equivalence_margin=equivalence_margin,
        bootstrap_iters=bootstrap_iters,
        ci_level=ci_level,
        rng=rng,
    )
    baseline_global = _baseline_stats(
        pivot_global,
        metric=metric,
        baseline=baseline,
        alpha=alpha,
        equivalence_margin=equivalence_margin,
        bootstrap_iters=bootstrap_iters,
        ci_level=ci_level,
        rng=rng,
    )

    by_classifier_pairs = []
    by_classifier_baseline = []
    for clf in sorted(df["classifier"].dropna().unique()):
        grp = df[df["classifier"] == clf]
        pivot_clf = _build_pivot(grp, metric=metric, index_cols=["fold"])
        if pivot_clf.empty:
            continue

        pair_clf = _pairwise_stats(
            pivot_clf,
            metric=metric,
            alpha=alpha,
            equivalence_margin=equivalence_margin,
            bootstrap_iters=bootstrap_iters,
            ci_level=ci_level,
            rng=rng,
        )
        if not pair_clf.empty:
            pair_clf.insert(0, "classifier", clf)
            by_classifier_pairs.append(pair_clf)

        baseline_clf = _baseline_stats(
            pivot_clf,
            metric=metric,
            baseline=baseline,
            alpha=alpha,
            equivalence_margin=equivalence_margin,
            bootstrap_iters=bootstrap_iters,
            ci_level=ci_level,
            rng=rng,
        )
        if not baseline_clf.empty:
            baseline_clf.insert(0, "classifier", clf)
            by_classifier_baseline.append(baseline_clf)

    pair_clf_all = pd.concat(by_classifier_pairs, ignore_index=True) if by_classifier_pairs else pd.DataFrame()
    baseline_clf_all = (
        pd.concat(by_classifier_baseline, ignore_index=True) if by_classifier_baseline else pd.DataFrame()
    )

    pair_global.to_csv(out_dir / f"pairwise_global_{metric}.csv", index=False)
    pair_clf_all.to_csv(out_dir / f"pairwise_by_classifier_{metric}.csv", index=False)
    baseline_global.to_csv(out_dir / f"baseline_global_{metric}.csv", index=False)
    baseline_clf_all.to_csv(out_dir / f"baseline_by_classifier_{metric}.csv", index=False)

    manifest = {
        "input_csv": str(results_csv),
        "metric": metric,
        "alpha": alpha,
        "equivalence_margin": equivalence_margin,
        "baseline": baseline,
        "bootstrap_iters": bootstrap_iters,
        "ci_level": ci_level,
        "seed": seed,
        "n_rows_used": int(len(df)),
        "n_classifiers": int(df["classifier"].nunique()),
        "n_imputers": int(df["imputer"].nunique()),
        "include_imputers": include_imputers or [],
        "exclude_imputers": exclude_imputers or [],
        "outputs": {
            "pairwise_global": f"pairwise_global_{metric}.csv",
            "pairwise_by_classifier": f"pairwise_by_classifier_{metric}.csv",
            "baseline_global": f"baseline_global_{metric}.csv",
            "baseline_by_classifier": f"baseline_by_classifier_{metric}.csv",
        },
    }
    (out_dir / f"manifest_{metric}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("Input rows used: %d", len(df))
    log.info("Classifiers: %s", sorted(df["classifier"].unique().tolist()))
    log.info("Imputers: %s", sorted(df["imputer"].unique().tolist()))
    log.info("Saved: %s", out_dir / f"pairwise_global_{metric}.csv")
    log.info("Saved: %s", out_dir / f"pairwise_by_classifier_{metric}.csv")
    log.info("Saved: %s", out_dir / f"baseline_global_{metric}.csv")
    log.info("Saved: %s", out_dir / f"baseline_by_classifier_{metric}.csv")

    return {
        "pairwise_global": pair_global,
        "pairwise_by_classifier": pair_clf_all,
        "baseline_global": baseline_global,
        "baseline_by_classifier": baseline_clf_all,
        "manifest": manifest,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Inferential analysis for imputation effect on model performance.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input-csv", default=None, help="Default: <paths.results_raw>/all_results.csv")
    parser.add_argument("--output-dir", default=None, help="Default: <paths.results_tables>/imputation_effect")
    parser.add_argument("--metric", default="f1_weighted")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.005,
        help="Practical margin in metric units (e.g. 0.005 = 0.5 p.p).",
    )
    parser.add_argument("--baseline", default="NoImpute")
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--imputers", default=None, help="Comma-separated allowlist, e.g. Media,Mediana,MICE")
    parser.add_argument("--exclude-imputers", default=None, help="Comma-separated denylist")
    return parser


def main():
    _setup_logging()
    args = _build_arg_parser().parse_args()
    run_imputation_effect_stats(
        config_path=args.config,
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        metric=args.metric,
        alpha=args.alpha,
        equivalence_margin=args.equivalence_margin,
        baseline=args.baseline,
        bootstrap_iters=args.bootstrap_iters,
        ci_level=args.ci_level,
        seed=args.seed,
        include_imputers=_split_csv_list(args.imputers),
        exclude_imputers=_split_csv_list(args.exclude_imputers),
    )


if __name__ == "__main__":
    main()
