#!/usr/bin/env python3
"""
Confirmatory Statistical Analysis for the Protocol.

Reads results/raw/protocol_results.csv and produces:
  - Paired delta comparisons against baseline (NoImpute)
  - Permutation test (primary) + Wilcoxon (secondary)
  - Bootstrap CI
  - Holm adjustment for multiple comparisons
  - TOST equivalence with primary (0.01) and sensitivity (0.005) margins
  - Separate analyses: with_88 (full) and without_88 (qwk_0_4)
  - Class-88 guardrails
  - Final conclusion following protocol §7 criteria

Outputs go to results/tables/protocol/.
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from src.config_loader import load_config
except ModuleNotFoundError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.config_loader import load_config

from src.stats_utils import (
    bootstrap_mean_ci,
    cohen_dz,
    equivalence_and_noninferiority,
    holm_adjust,
    permutation_test_paired,
    rank_biserial,
    wilcoxon_two_sided,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _build_pivot(df, metric, baseline, global_agg="mean"):
    """Build pivot with classifier-aggregated values to avoid pseudoreplication.

    Aggregation rule (P3): average each imputer's metric across classifiers per
    (repeat, outer_fold) before running paired tests. This keeps the unit of
    observation at the split level, not the split×classifier level.
    Complete-case: only splits where ALL imputers have valid values are kept.
    """
    agg_fn = global_agg  # 'mean' or 'median' — pre-fixed in config before run
    df_agg = (
        df.groupby(["repeat", "outer_fold", "imputer"])[metric]
        .agg(agg_fn)
        .reset_index()
    )
    pivot = df_agg.pivot_table(
        index=["repeat", "outer_fold"], columns="imputer", values=metric
    )
    # Complete-case: drop splits where any imputer is missing
    pivot = pivot.dropna()
    if baseline not in pivot.columns:
        log.warning("Baseline '%s' not found in pivot columns: %s", baseline, list(pivot.columns))
    return pivot


def _rng_for(metric: str, scope: str, imputer: str, baseline: str, base_seed: int = 42) -> np.random.Generator:
    """Deterministic, order-independent RNG per comparison (P7).

    Derived via key hash so that adding/removing comparisons doesn't
    shift the residual RNG state and alter other p-values.
    """
    key = f"{metric}:{scope}:{imputer}:{baseline}"
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(base_seed ^ h)


def _export_best_method(global_rows, guardrails, alpha, margin, metric, out_dir) -> Path:
    """Export best_method.json with the explicit imputer selection rule (A3).

    Selection order:
      1. Imputers with Holm-adjusted p < alpha AND delta_mean > margin
      2. Among those, pick the one with the highest delta_mean
      3. If none qualify, export {"imputer": null} and temporal is skipped
    """
    eq_key = f"equivalent_{str(margin).replace('.', '_')}"
    candidates = [
        r for r in global_rows
        if r.get("permutation_pvalue_holm", 1.0) < alpha
        and r.get("delta_mean", 0.0) > margin
    ]
    # Exclude any candidate that degrades class-88
    if not guardrails.empty:
        degraded_imps = set(guardrails[guardrails.get("degraded", False) == True]["imputer"].tolist())
        candidates = [r for r in candidates if r["imputer"] not in degraded_imps]

    if candidates:
        best = max(candidates, key=lambda r: r["delta_mean"])
        payload = {
            "imputer": best["imputer"],
            "metric": metric,
            "delta_mean": float(best["delta_mean"]),
            "p_holm": float(best.get("permutation_pvalue_holm", float("nan"))),
            "selection_rule": f"p_holm<{alpha} AND delta_mean>{margin}",
        }
    else:
        payload = {"imputer": None, "metric": metric, "selection_rule": "no_candidate"}

    best_path = out_dir / "best_method.json"
    best_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("best_method.json exported: %s", payload)
    return best_path


def _baseline_comparison(pivot, metric, baseline, alpha, margins, bootstrap_iters, ci_level, rng, n_perms):
    """Compare each imputer against baseline, return list of result dicts."""
    if baseline not in pivot.columns:
        return []

    base_vals = pivot[baseline].values
    rows = []
    for imp_name in pivot.columns:
        if imp_name == baseline:
            continue
        imp_vals = pivot[imp_name].values
        mask = np.isfinite(base_vals) & np.isfinite(imp_vals)
        if mask.sum() < 3:
            continue

        diff = imp_vals[mask] - base_vals[mask]
        n = len(diff)
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1))

        ci_lo, ci_hi = bootstrap_mean_ci(diff, bootstrap_iters, ci_level, rng)
        w_stat, w_pval = wilcoxon_two_sided(diff)
        perm_mean, perm_pval = permutation_test_paired(imp_vals[mask], base_vals[mask], n_perms=n_perms, seed=int(rng.integers(0, 100000)))
        dz = cohen_dz(diff)
        rb = rank_biserial(diff)

        row = {
            "metric": metric,
            "imputer": imp_name,
            "baseline": baseline,
            "n_obs": n,
            "delta_mean": mean_diff,
            "delta_std": std_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "permutation_pvalue": perm_pval,
            "wilcoxon_stat": w_stat,
            "wilcoxon_pvalue": w_pval,
            "cohen_dz": dz,
            "rank_biserial": rb,
        }

        for margin in margins:
            eq = equivalence_and_noninferiority(diff, margin, alpha)
            suffix = str(margin).replace(".", "_")
            row[f"tost_pvalue_{suffix}"] = eq["tost_pvalue"]
            row[f"equivalent_{suffix}"] = eq["equivalent"]
            row[f"noninferiority_p_{suffix}"] = eq["noninferiority_p"]
            row[f"noninferior_{suffix}"] = eq["noninferior"]

        rows.append(row)

    return rows


def _by_classifier_comparison(df, metric, baseline, alpha, margins, bootstrap_iters, ci_level, rng, n_perms):
    """Same as baseline comparison but per classifier."""
    rows = []
    for clf_name, grp in df.groupby("classifier"):
        idx_cols = ["repeat", "outer_fold"]
        pivot = grp.pivot_table(index=idx_cols, columns="imputer", values=metric, aggfunc="first")
        if baseline not in pivot.columns:
            continue

        base_vals = pivot[baseline].values
        for imp_name in pivot.columns:
            if imp_name == baseline:
                continue
            imp_vals = pivot[imp_name].values
            mask = np.isfinite(base_vals) & np.isfinite(imp_vals)
            if mask.sum() < 3:
                continue

            diff = imp_vals[mask] - base_vals[mask]
            n = len(diff)

            ci_lo, ci_hi = bootstrap_mean_ci(diff, bootstrap_iters, ci_level, rng)
            _, w_pval = wilcoxon_two_sided(diff)
            _, perm_pval = permutation_test_paired(imp_vals[mask], base_vals[mask], n_perms=n_perms, seed=int(rng.integers(0, 100000)))
            dz = cohen_dz(diff)

            row = {
                "metric": metric,
                "classifier": clf_name,
                "imputer": imp_name,
                "baseline": baseline,
                "n_obs": n,
                "delta_mean": float(np.mean(diff)),
                "delta_std": float(np.std(diff, ddof=1)),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "permutation_pvalue": perm_pval,
                "wilcoxon_pvalue": w_pval,
                "cohen_dz": dz,
            }

            for margin in margins:
                eq = equivalence_and_noninferiority(diff, margin, alpha)
                suffix = str(margin).replace(".", "_")
                row[f"tost_pvalue_{suffix}"] = eq["tost_pvalue"]
                row[f"equivalent_{suffix}"] = eq["equivalent"]
                row[f"noninferior_{suffix}"] = eq["noninferior"]

            rows.append(row)

    return rows


def _apply_holm(rows, pvalue_col="permutation_pvalue"):
    """Apply Holm adjustment to a list of result dicts."""
    if not rows:
        return rows
    pvals = np.array([r[pvalue_col] for r in rows])
    adjusted = holm_adjust(pvals)
    for r, adj in zip(rows, adjusted):
        r[f"{pvalue_col}_holm"] = float(adj)
    return rows


def _class88_guardrails(df, baseline, class88_margin: float = 0.01, bootstrap_iters: int = 2000):
    """Check class-88 metrics for degradation using bootstrap CI (P6).

    Degradation criterion: the 95% CI upper bound is below -class88_margin,
    i.e., the entire CI is in the degradation zone. This is more rigorous
    than a fixed mean threshold.
    """
    metrics_88 = ["f1_88", "recall_88", "f1_estadiavel_bin"]
    rows = []
    for imp_name in df["imputer"].unique():
        if imp_name == baseline:
            continue
        imp_df = df[df["imputer"] == imp_name]
        base_df = df[df["imputer"] == baseline]

        for m in metrics_88:
            if m not in imp_df.columns or m not in base_df.columns:
                continue
            imp_vals = imp_df.groupby(["repeat", "outer_fold"])[m].mean()
            base_vals = base_df.groupby(["repeat", "outer_fold"])[m].mean()
            merged = pd.merge(imp_vals.rename("imp"), base_vals.rename("base"),
                              left_index=True, right_index=True)
            if len(merged) < 3:
                continue
            diff = merged["imp"].values - merged["base"].values
            rng88 = _rng_for(m, "guardrail", imp_name, baseline)
            ci_lo, ci_hi = bootstrap_mean_ci(diff, bootstrap_iters, 0.95, rng88)
            _, w_pval = wilcoxon_two_sided(diff)
            rows.append({
                "imputer": imp_name,
                "metric": m,
                "baseline_mean": float(merged["base"].mean()),
                "imputer_mean": float(merged["imp"].mean()),
                "delta_mean": float(np.mean(diff)),
                "ci_lo": float(ci_lo),
                "ci_hi": float(ci_hi),
                "wilcoxon_pvalue": float(w_pval),
                "n_obs": len(diff),
                # Degraded: entire CI is below the negative margin (strict criterion)
                "degraded": bool(ci_hi < -class88_margin),
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _generate_plots(pivot, metric, baseline, out_dir):
    """Generate paired distribution plots (e.g. boxplot of deltas vs baseline)."""
    try:
        import warnings
        import matplotlib.pyplot as plt
        import seaborn as sns

        if baseline not in pivot.columns:
            return

        deltas = []
        imputer_names = []

        base_vals = pivot[baseline].values
        for imp_name in pivot.columns:
            if imp_name == baseline:
                continue
            imp_vals = pivot[imp_name].values
            mask = np.isfinite(base_vals) & np.isfinite(imp_vals)
            if mask.sum() < 3:
                continue
            diff = imp_vals[mask] - base_vals[mask]
            deltas.extend(diff)
            imputer_names.extend([imp_name] * mask.sum())

        if not deltas:
            return

        plt.figure(figsize=(10, 6))
        # Suppress seaborn FutureWarnings from its internal use of deprecated pandas API
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
            sns.boxplot(x=imputer_names, y=deltas, palette="Set2")
            sns.stripplot(x=imputer_names, y=deltas, color="black", alpha=0.5, jitter=True)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)
        plt.title(f"Distribuição de Delta ({metric}) vs {baseline}")
        plt.ylabel(f"Delta {metric}")
        plt.xlabel("Imputador")
        plt.tight_layout()
        plt.savefig(out_dir / f"protocol_deltas_{metric}.png", dpi=300)
        plt.close()
        log.info(f"Plot saved: protocol_deltas_{metric}.png")
    except ImportError:
        log.warning("matplotlib or seaborn not installed; skipping plots.")
    except Exception as e:
        log.warning(f"Could not generate plot: {e}")


# ─────────────────────────────────────────────────────────────────────
# Main Analysis
# ─────────────────────────────────────────────────────────────────────

def run_protocol_stats(
    config_path: str = "config/config.yaml",
    cfg: dict = None,
    input_csv: str = None,
    baseline: str = "NoImpute",
):
    """Run confirmatory statistical analysis on protocol results."""
    if cfg is None:
        cfg = load_config(config_path)

    proto = cfg.get("protocol", {})
    alpha = proto.get("alpha", 0.05)
    eq_margin = proto.get("equivalence_margin", 0.01)
    sens_margin = proto.get("sensitivity_margin", 0.005)
    margins = [eq_margin, sens_margin]
    bootstrap_iters = proto.get("bootstrap_iters", 5000)
    n_perms = proto.get("permutation_n_perms", 10000)
    global_agg = proto.get("global_agg", "mean")  # P3: pre-fixed aggregation rule
    class88_margin = proto.get("class88_margin", 0.01)  # P6
    base_seed = proto.get("stats_seed", 42)  # P7: base seed for derived RNGs
    ci_level = 0.95

    # ── Load results ──
    csv_path = Path(input_csv or (Path(cfg["paths"]["results_raw"]) / "protocol_results.csv"))
    if not csv_path.exists():
        log.error("Protocol results not found: %s", csv_path)
        return None
    df = pd.read_csv(csv_path)
    log.info("Loaded %d protocol results from %s", len(df), csv_path)

    out_dir = Path(cfg["paths"]["results_tables"]) / "protocol"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ──═ PHASE C: Primary metric (f1_weighted) ═──
    metric = proto.get("primary_metric", "f1_weighted")
    log.info("Analyzing primary metric: %s (global_agg=%s)", metric, global_agg)

    pivot = _build_pivot(df, metric, baseline, global_agg)
    global_rows = _baseline_comparison(
        pivot, metric, baseline, alpha, margins, bootstrap_iters, ci_level,
        _rng_for(metric, "global", "*", baseline, base_seed), n_perms,
    )
    global_rows = _apply_holm(global_rows)
    pd.DataFrame(global_rows).to_csv(out_dir / f"baseline_global_{metric}.csv", index=False)
    _generate_plots(pivot, metric, baseline, out_dir)

    by_clf_rows = _by_classifier_comparison(
        df, metric, baseline, alpha, margins, bootstrap_iters, ci_level,
        _rng_for(metric, "by_clf", "*", baseline, base_seed), n_perms,
    )
    by_clf_rows = _apply_holm(by_clf_rows)
    pd.DataFrame(by_clf_rows).to_csv(out_dir / f"baseline_by_classifier_{metric}.csv", index=False)

    # ─── Secondary metric: f1_macro ───
    if "f1_macro" in df.columns:
        pivot_m = _build_pivot(df, "f1_macro", baseline, global_agg)
        macro_rows = _baseline_comparison(
            pivot_m, "f1_macro", baseline, alpha, margins, bootstrap_iters, ci_level,
            _rng_for("f1_macro", "global", "*", baseline, base_seed), n_perms,
        )
        macro_rows = _apply_holm(macro_rows)
        pd.DataFrame(macro_rows).to_csv(out_dir / "baseline_global_f1_macro.csv", index=False)
        _generate_plots(pivot_m, "f1_macro", baseline, out_dir)

    # ─── Ordinal metric: qwk_0_4 ───
    if "qwk_0_4" in df.columns and df["qwk_0_4"].notna().sum() > 0:
        pivot_q = _build_pivot(df, "qwk_0_4", baseline, global_agg)
        qwk_rows = _baseline_comparison(
            pivot_q, "qwk_0_4", baseline, alpha, margins, bootstrap_iters, ci_level,
            _rng_for("qwk_0_4", "global", "*", baseline, base_seed), n_perms,
        )
        qwk_rows = _apply_holm(qwk_rows)
        pd.DataFrame(qwk_rows).to_csv(out_dir / "baseline_global_qwk_0_4.csv", index=False)
        _generate_plots(pivot_q, "qwk_0_4", baseline, out_dir)

    # ─── Class-88 guardrails ───
    guardrails = _class88_guardrails(df, baseline, class88_margin, bootstrap_iters)
    if not guardrails.empty:
        guardrails.to_csv(out_dir / "class88_guardrails.csv", index=False)

    # ──═ PHASE E: Conclusion + best_method.json ═──
    conclusion = _generate_conclusion(global_rows, guardrails, alpha, eq_margin, metric)
    conclusion_path = out_dir / "conclusion.md"
    conclusion_path.write_text(conclusion, encoding="utf-8")
    log.info("Conclusion saved: %s", conclusion_path)

    # ── Export best_method.json (A3: explicit selection rule) ──
    best_path = _export_best_method(global_rows, guardrails, alpha, eq_margin, metric, out_dir)

    log.info("Protocol statistical analysis complete. Outputs: %s", out_dir)
    return best_path


def _generate_conclusion(global_rows, guardrails, alpha, margin, metric):
    """Generate the final conclusion markdown."""
    lines = ["# Conclusão do Protocolo Confirmatório\n"]
    lines.append(f"**Métrica primária**: `{metric}`  ")
    lines.append(f"**Alpha**: {alpha}  ")
    lines.append(f"**Margem de equivalência**: {margin}\n")

    if not global_rows:
        lines.append("Nenhum resultado disponível para análise.\n")
        return "\n".join(lines)

    lines.append("## Resultados por Imputador\n")
    lines.append("| Imputador | Δ média | IC 95% | p (perm) | p (Holm) | Equivalente? | Veredicto |")
    lines.append("|---|---|---|---|---|---|---|")

    for r in global_rows:
        delta = r["delta_mean"]
        ci = f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
        p_perm = r["permutation_pvalue"]
        p_holm = r.get("permutation_pvalue_holm", p_perm)
        eq_key = f"equivalent_{str(margin).replace('.', '_')}"
        equiv = "Sim" if r.get(eq_key, False) else "Não"

        if p_holm < alpha and delta > margin:
            verdict = "✅ **Melhora**"
        elif r.get(eq_key, False):
            verdict = "⚖️ Equivalente"
        elif p_holm >= alpha:
            verdict = "❌ Sem evidência"
        else:
            verdict = "⚠️ Inconclusivo"

        lines.append(
            f"| {r['imputer']} | {delta:+.4f} | {ci} | {p_perm:.4f} | {p_holm:.4f} | {equiv} | {verdict} |"
        )

    # Class-88 check
    if not guardrails.empty:
        degraded = guardrails[guardrails["degraded"] == True]
        if not degraded.empty:
            lines.append("\n## ⚠️ Alerta: Degradação na Classe 88\n")
            for _, row in degraded.iterrows():
                lines.append(f"- **{row['imputer']}**: `{row['metric']}` Δ={row['delta_mean']:+.4f}")
        else:
            lines.append("\n## ✅ Classe 88: Sem degradação significativa\n")

    lines.append("\n---\n")
    lines.append("*Gerado automaticamente por `run_protocol_stats.py`*\n")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Protocol Confirmatory Statistical Analysis")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input-csv", default=None, help="Path to protocol_results.csv")
    parser.add_argument("--baseline", default="NoImpute")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    cfg = load_config(args.config)
    run_protocol_stats(config_path=args.config, cfg=cfg, input_csv=args.input_csv, baseline=args.baseline)


if __name__ == "__main__":
    main()
