#!/usr/bin/env python3
"""
Ordinal sensitivity analysis from saved confusion matrices.

This script reads:
    - results/raw/all_results_detailed.json
    - results/tables/metadata.json

It computes fold-level ordinal metrics for two scenarios:
    1) all_classes
    2) without_88 (if class 88 is present in target mapping)

Outputs are written to:
    results/tables/ordinal_sensitivity/
"""

import argparse
import json
import logging
import math
import sys
from ast import literal_eval
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


def _coerce_confusion_matrix(raw_cm):
    cm_data = raw_cm
    if isinstance(cm_data, str):
        try:
            cm_data = json.loads(cm_data)
        except Exception:
            cm_data = literal_eval(cm_data)

    cm = np.asarray(cm_data)
    cm = np.squeeze(cm)

    if cm.ndim == 1:
        size = cm.size
        n = int(np.sqrt(size))
        if n * n == size:
            cm = cm.reshape(n, n)
        else:
            return None

    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        return None

    return cm.astype(float)


def _qwk_from_confusion(cm):
    total = cm.sum()
    k = cm.shape[0]
    if total <= 0 or k < 2:
        return np.nan

    idx = np.arange(k)
    dist_sq = (idx[:, None] - idx[None, :]) ** 2
    weights = dist_sq / float((k - 1) ** 2)

    observed = cm
    row_marg = observed.sum(axis=1)
    col_marg = observed.sum(axis=0)
    expected = np.outer(row_marg, col_marg) / total

    num = float((weights * observed).sum())
    den = float((weights * expected).sum())
    if den <= 0:
        return np.nan
    return float(1.0 - (num / den))


def _distance_metrics(cm):
    total = cm.sum()
    if total <= 0:
        return {
            "exact_accuracy": np.nan,
            "mae_distance": np.nan,
            "rmse_distance": np.nan,
            "severe_error_rate": np.nan,
            "within_one_rate": np.nan,
        }

    k = cm.shape[0]
    idx = np.arange(k)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(float)
    dist_sq = dist**2

    exact = float(np.trace(cm) / total)
    mae = float((dist * cm).sum() / total)
    rmse = float(np.sqrt((dist_sq * cm).sum() / total))
    severe = float(cm[dist >= 2].sum() / total)
    within_one = float(cm[dist <= 1].sum() / total)

    return {
        "exact_accuracy": exact,
        "mae_distance": mae,
        "rmse_distance": rmse,
        "severe_error_rate": severe,
        "within_one_rate": within_one,
    }


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


def _pairwise_qwk(df, alpha, equivalence_margin, baseline, bootstrap_iters, ci_level, rng):
    rows_pairs = []
    rows_baseline = []

    for (scenario, classifier), grp in df.groupby(["scenario", "classifier"], dropna=False):
        pivot = grp.pivot_table(index="fold", columns="imputer", values="qwk", aggfunc="first")
        imputers = sorted([c for c in pivot.columns if pivot[c].notna().any()])

        for a, b in combinations(imputers, 2):
            pair = pivot[[a, b]].dropna()
            if pair.empty:
                continue
            diff = (pair[a] - pair[b]).to_numpy(dtype=float)
            ci_low, ci_high = _bootstrap_mean_ci(diff, bootstrap_iters, ci_level, rng)
            stat, pval = _wilcoxon_two_sided(diff)
            eq = _equivalence_and_noninferiority(diff, equivalence_margin, alpha)

            rows_pairs.append(
                {
                    "scenario": scenario,
                    "classifier": classifier,
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

        if baseline in pivot.columns:
            for imp in sorted([c for c in pivot.columns if c != baseline]):
                pair = pivot[[imp, baseline]].dropna()
                if pair.empty:
                    continue
                diff = (pair[imp] - pair[baseline]).to_numpy(dtype=float)
                ci_low, ci_high = _bootstrap_mean_ci(diff, bootstrap_iters, ci_level, rng)
                stat, pval = _wilcoxon_two_sided(diff)
                eq = _equivalence_and_noninferiority(diff, equivalence_margin, alpha)

                rows_baseline.append(
                    {
                        "scenario": scenario,
                        "classifier": classifier,
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

    pairs = pd.DataFrame(rows_pairs)
    if not pairs.empty:
        pairs["p_wilcoxon_holm"] = np.nan
        pairs["significant_holm"] = False
        for (scenario, classifier), idx in pairs.groupby(["scenario", "classifier"]).groups.items():
            adj = _holm_adjust(pairs.loc[list(idx), "p_wilcoxon"].to_numpy(dtype=float))
            pairs.loc[list(idx), "p_wilcoxon_holm"] = adj
            pairs.loc[list(idx), "significant_holm"] = adj < alpha
        pairs = pairs.sort_values(["scenario", "classifier", "p_wilcoxon_holm", "delta_mean"]).reset_index(drop=True)

    baseline_df = pd.DataFrame(rows_baseline)
    if not baseline_df.empty:
        baseline_df["p_wilcoxon_holm"] = np.nan
        baseline_df["significant_holm"] = False
        for (scenario, classifier), idx in baseline_df.groupby(["scenario", "classifier"]).groups.items():
            adj = _holm_adjust(baseline_df.loc[list(idx), "p_wilcoxon"].to_numpy(dtype=float))
            baseline_df.loc[list(idx), "p_wilcoxon_holm"] = adj
            baseline_df.loc[list(idx), "significant_holm"] = adj < alpha
        baseline_df = baseline_df.sort_values(["scenario", "classifier", "p_wilcoxon_holm", "delta_mean"]).reset_index(
            drop=True
        )

    return pairs, baseline_df


def run_ordinal_sensitivity(
    config_path="config/config.yaml",
    detailed_json=None,
    metadata_json=None,
    output_dir=None,
    baseline="NoImpute",
    alpha=0.05,
    equivalence_margin=0.005,
    bootstrap_iters=5000,
    ci_level=0.95,
    seed=42,
):
    cfg = load_config(config_path)
    results_raw = Path(cfg["paths"]["results_raw"])
    results_tables = Path(cfg["paths"]["results_tables"])

    detailed_path = Path(detailed_json) if detailed_json else results_raw / "all_results_detailed.json"
    metadata_path = Path(metadata_json) if metadata_json else results_tables / "metadata.json"
    out_dir = Path(output_dir) if output_dir else results_tables / "ordinal_sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not detailed_path.exists():
        raise FileNotFoundError(f"Detailed results not found: {detailed_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    with open(detailed_path, "r", encoding="utf-8") as f:
        detailed = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    target_map = metadata.get("target_mapping", {}) or {}
    idx_88 = None
    if "88" in target_map:
        try:
            idx_88 = int(target_map["88"])
        except Exception:
            idx_88 = None

    rows = []
    for item in detailed:
        if item.get("error"):
            continue
        cm = _coerce_confusion_matrix(item.get("confusion_matrix"))
        if cm is None or cm.shape[0] < 2 or cm.sum() <= 0:
            continue

        base = {
            "fold": item.get("fold"),
            "imputer": item.get("imputer"),
            "classifier": item.get("classifier"),
            "runtime_mode": item.get("runtime_mode"),
            "n_classes": int(cm.shape[0]),
            "n_samples": int(cm.sum()),
        }
        metrics_all = {"qwk": _qwk_from_confusion(cm), **_distance_metrics(cm)}
        rows.append({"scenario": "all_classes", **base, **metrics_all})

        if idx_88 is not None and 0 <= idx_88 < cm.shape[0] and cm.shape[0] > 2:
            cm_wo88 = np.delete(np.delete(cm, idx_88, axis=0), idx_88, axis=1)
            if cm_wo88.shape[0] >= 2 and cm_wo88.sum() > 0:
                metrics_wo = {"qwk": _qwk_from_confusion(cm_wo88), **_distance_metrics(cm_wo88)}
                rows.append(
                    {
                        "scenario": "without_88",
                        **base,
                        "n_classes": int(cm_wo88.shape[0]),
                        "n_samples": int(cm_wo88.sum()),
                        **metrics_wo,
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid confusion matrices were found to compute ordinal metrics.")

    df = df.sort_values(["scenario", "classifier", "imputer", "fold"]).reset_index(drop=True)
    df.to_csv(out_dir / "ordinal_metrics_by_fold.csv", index=False)

    summary = (
        df.groupby(["scenario", "imputer", "classifier"], dropna=False)
        .agg(
            n_folds=("fold", "nunique"),
            qwk_mean=("qwk", "mean"),
            qwk_std=("qwk", "std"),
            exact_accuracy_mean=("exact_accuracy", "mean"),
            exact_accuracy_std=("exact_accuracy", "std"),
            mae_distance_mean=("mae_distance", "mean"),
            mae_distance_std=("mae_distance", "std"),
            rmse_distance_mean=("rmse_distance", "mean"),
            rmse_distance_std=("rmse_distance", "std"),
            severe_error_rate_mean=("severe_error_rate", "mean"),
            severe_error_rate_std=("severe_error_rate", "std"),
            within_one_rate_mean=("within_one_rate", "mean"),
            within_one_rate_std=("within_one_rate", "std"),
            n_samples_mean=("n_samples", "mean"),
        )
        .reset_index()
    )
    summary = summary.sort_values(["scenario", "classifier", "qwk_mean"], ascending=[True, True, False]).reset_index(
        drop=True
    )
    summary.to_csv(out_dir / "ordinal_metrics_summary.csv", index=False)

    rng = np.random.default_rng(seed)
    pairwise_qwk, baseline_qwk = _pairwise_qwk(
        df=df,
        alpha=alpha,
        equivalence_margin=equivalence_margin,
        baseline=baseline,
        bootstrap_iters=bootstrap_iters,
        ci_level=ci_level,
        rng=rng,
    )
    pairwise_qwk.to_csv(out_dir / "ordinal_qwk_pairwise.csv", index=False)
    baseline_qwk.to_csv(out_dir / "ordinal_qwk_baseline.csv", index=False)

    manifest = {
        "input_detailed_json": str(detailed_path),
        "input_metadata_json": str(metadata_path),
        "baseline": baseline,
        "alpha": alpha,
        "equivalence_margin": equivalence_margin,
        "bootstrap_iters": bootstrap_iters,
        "ci_level": ci_level,
        "seed": seed,
        "target_mapping": target_map,
        "encoded_class_88": idx_88,
        "n_rows_by_fold": int(len(df)),
        "n_scenarios": int(df["scenario"].nunique()),
        "n_imputers": int(df["imputer"].nunique()),
        "n_classifiers": int(df["classifier"].nunique()),
        "outputs": {
            "by_fold": "ordinal_metrics_by_fold.csv",
            "summary": "ordinal_metrics_summary.csv",
            "pairwise_qwk": "ordinal_qwk_pairwise.csv",
            "baseline_qwk": "ordinal_qwk_baseline.csv",
        },
    }
    (out_dir / "manifest_ordinal.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("Rows in ordinal_metrics_by_fold: %d", len(df))
    log.info("Scenarios detected: %s", sorted(df["scenario"].unique().tolist()))
    log.info("Saved: %s", out_dir / "ordinal_metrics_by_fold.csv")
    log.info("Saved: %s", out_dir / "ordinal_metrics_summary.csv")
    log.info("Saved: %s", out_dir / "ordinal_qwk_pairwise.csv")
    log.info("Saved: %s", out_dir / "ordinal_qwk_baseline.csv")

    return {
        "by_fold": df,
        "summary": summary,
        "pairwise_qwk": pairwise_qwk,
        "baseline_qwk": baseline_qwk,
        "manifest": manifest,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Ordinal sensitivity analysis (with and without class 88).")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--detailed-json", default=None, help="Default: <paths.results_raw>/all_results_detailed.json")
    parser.add_argument("--metadata-json", default=None, help="Default: <paths.results_tables>/metadata.json")
    parser.add_argument("--output-dir", default=None, help="Default: <paths.results_tables>/ordinal_sensitivity")
    parser.add_argument("--baseline", default="NoImpute")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.005,
        help="Practical margin in QWK units (e.g. 0.005 = 0.5 p.p).",
    )
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    _setup_logging()
    args = _build_arg_parser().parse_args()
    run_ordinal_sensitivity(
        config_path=args.config,
        detailed_json=args.detailed_json,
        metadata_json=args.metadata_json,
        output_dir=args.output_dir,
        baseline=args.baseline,
        alpha=args.alpha,
        equivalence_margin=args.equivalence_margin,
        bootstrap_iters=args.bootstrap_iters,
        ci_level=args.ci_level,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
