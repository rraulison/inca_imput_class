"""
Step 4 - Result analysis and visualization.
Input: results/raw/
Output: results/tables/ and results/figures/
"""

import json
import logging
import os
import warnings
from ast import literal_eval
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon

# Suppress harmless warnings for cleaner terminal output
os.environ["NUMEXPR_MAX_THREADS"] = "16"
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")
warnings.filterwarnings("ignore", message="When grouping with a length-1 list-like")

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config_loader import load_config

log = logging.getLogger(__name__)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "font.family": "serif",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)

IMP_ORDER = ["Media", "Mediana", "kNN", "MICE", "MICE_XGBoost", "MissForest", "NoImpute", "RawSemEncoding"]
CLF_ORDER = ["cuML_RF", "XGBoost", "CatBoost", "cuML_SVM", "cuML_MLP"]

IMP_LABELS = {
    "Media": "Media/Moda",
    "Mediana": "Mediana/Moda",
    "kNN": "kNN (k=5)",
    "MICE": "MICE",
    "MICE_XGBoost": "MICE-XGBoost",
    "MissForest": "MissForest",
    "NoImpute": "Sem imputacao (NaN nativo)",
    "RawSemEncoding": "Cru sem encoding (CatBoost)",
}

CLF_LABELS = {
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "cuML_RF": "cuML RF",
    "cuML_SVM": "cuML SVM",
    "cuML_MLP": "cuML MLP",
}

IMP_COLORS = {
    "Media": "#1f77b4",
    "Mediana": "#ff7f0e",
    "kNN": "#2ca02c",
    "MICE": "#d62728",
    "MICE_XGBoost": "#9467bd",
    "MissForest": "#8c564b",
    "NoImpute": "#17becf",
    "RawSemEncoding": "#bcbd22",
}


def _save_fig(fig, name, out_dir):
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _fmt(mean, std, d=4):
    return f"{mean:.{d}f} +- {std:.{d}f}"


def _fmt_latex(mean, std, d=4):
    return f"${mean:.{d}f} \\pm {std:.{d}f}$"


def _ordered(order, available):
    return [x for x in order if x in available]


def _aggregate(df):
    metrics = [
        "accuracy",
        "recall_weighted",
        "f1_weighted",
        "auc_weighted",
        "recall_macro",
        "f1_macro",
        "auc_macro",
    ]
    tcols = [c for c in df.columns if c.startswith("time_")]
    cols = [c for c in metrics + tcols if c in df.columns]

    rows = []
    for (imp, clf), group in df.groupby(["imputer", "classifier"]):
        row = {"imputer": imp, "classifier": clf, "n_folds": len(group)}
        for col in cols:
            values = group[col].dropna()
            if len(values):
                row[f"{col}_mean"] = values.mean()
                row[f"{col}_std"] = values.std()
        rows.append(row)

    summary = pd.DataFrame(rows)
    imp_ord = {v: i for i, v in enumerate(IMP_ORDER)}
    clf_ord = {v: i for i, v in enumerate(CLF_ORDER)}
    summary = summary.assign(_imp=summary["imputer"].map(imp_ord), _clf=summary["classifier"].map(clf_ord))
    return summary.sort_values(["_imp", "_clf"]).drop(columns=["_imp", "_clf"]).reset_index(drop=True)


def _main_table(summary, out_dir):
    rows_csv = []
    rows_tex = []

    for _, row in summary.iterrows():
        base = {
            "Imputador": IMP_LABELS.get(row["imputer"], row["imputer"]),
            "Classificador": CLF_LABELS.get(row["classifier"], row["classifier"]),
        }
        metrics = ["accuracy", "recall_weighted", "f1_weighted", "auc_weighted"]
        labels = ["Acuracia", "Revocacao", "F1-Score", "AUC"]

        row_csv = dict(base)
        row_tex = dict(base)
        for metric, label in zip(metrics, labels):
            row_csv[label] = _fmt(row[f"{metric}_mean"], row[f"{metric}_std"])
            row_tex[label] = _fmt_latex(row[f"{metric}_mean"], row[f"{metric}_std"])

        row_csv["Tempo (s)"] = f"{row.get('time_total_mean', 0):.0f}"
        row_tex["Tempo (s)"] = f"${row.get('time_total_mean', 0):.0f}$"
        rows_csv.append(row_csv)
        rows_tex.append(row_tex)

    pd.DataFrame(rows_csv).to_csv(out_dir / "main_table.csv", index=False)

    tex = pd.DataFrame(rows_tex).to_latex(
        index=False,
        escape=False,
        column_format="llccccc",
        caption="Resultados por metodo de imputacao e classificador (media +- std, 5-fold CV).",
        label="tab:main_results",
    )
    (out_dir / "main_table.tex").write_text(tex, encoding="utf-8")
    log.info("Main table saved")


def _per_class_table(detailed, summary, out_dir, target_map):
    best = summary.loc[summary["f1_weighted_mean"].idxmax()]
    inv = {int(v): str(k) for k, v in target_map.items()}

    per_class = {}
    for row in detailed:
        if (
            row.get("imputer") == best["imputer"]
            and row.get("classifier") == best["classifier"]
            and "classification_report" in row
            and not row.get("error")
        ):
            for key, value in row["classification_report"].items():
                if isinstance(value, dict) and "precision" in value:
                    per_class.setdefault(key, {m: [] for m in ["precision", "recall", "f1-score", "support"]})
                    for metric in per_class[key]:
                        per_class[key][metric].append(value[metric])

    rows = []
    for key in sorted(per_class, key=str):
        values = per_class[key]
        rows.append(
            {
                "Classe": inv.get(int(key), key) if str(key).isdigit() else key,
                "Precision": _fmt(np.mean(values["precision"]), np.std(values["precision"])),
                "Recall": _fmt(np.mean(values["recall"]), np.std(values["recall"])),
                "F1-Score": _fmt(np.mean(values["f1-score"]), np.std(values["f1-score"])),
                "Support": f"{np.mean(values['support']):.0f}",
            }
        )

    pd.DataFrame(rows).to_csv(out_dir / "per_class_report.csv", index=False)
    log.info("Per-class report saved")


def _ranking(summary, out_dir):
    metrics = [c for c in ["accuracy_mean", "f1_weighted_mean", "f1_macro_mean", "auc_weighted_mean"] if c in summary.columns]
    ranking = summary[["imputer", "classifier"]].copy()
    ranking["method"] = ranking["imputer"] + " + " + ranking["classifier"].map(lambda x: CLF_LABELS.get(x, x))

    for metric in metrics:
        ranking[f"rank_{metric}"] = summary[metric].rank(ascending=False).astype(int)
        ranking[metric] = summary[metric]

    rank_cols = [c for c in ranking.columns if c.startswith("rank_")]
    ranking["rank_medio"] = ranking[rank_cols].mean(axis=1)
    ranking.sort_values("rank_medio").to_csv(out_dir / "ranking.csv", index=False)
    log.info("Ranking saved")


def _stat_tests(df, out_dir, metric="f1_weighted"):
    df = df.dropna(subset=[metric]).copy()
    df["method"] = df["imputer"] + "_" + df["classifier"]
    pivot = df.pivot_table(index="fold", columns="method", values=metric, aggfunc="first").dropna(axis=1)
    methods = pivot.columns.tolist()

    if len(methods) < 3:
        log.warning("Less than 3 methods available for Friedman test (%s)", metric)
        return

    stat, pval = friedmanchisquare(*[pivot[c].values for c in methods])
    pd.DataFrame(
        [{"metric": metric, "friedman_stat": stat, "friedman_p": pval, "significant": pval < 0.05}]
    ).to_csv(out_dir / f"stat_friedman_{metric}.csv", index=False)

    if pval < 0.05:
        pairs = []
        n_comp = len(methods) * (len(methods) - 1) / 2
        for m1, m2 in combinations(methods, 2):
            try:
                stat_w, p_w = wilcoxon(pivot[m1].values, pivot[m2].values)
                p_bonf = min(p_w * n_comp, 1.0)
                pairs.append(
                    {
                        "m1": m1,
                        "m2": m2,
                        "stat": stat_w,
                        "p": p_w,
                        "p_bonf": p_bonf,
                        "sig": p_bonf < 0.05,
                    }
                )
            except Exception:
                continue
        pd.DataFrame(pairs).to_csv(out_dir / f"stat_wilcoxon_{metric}.csv", index=False)


def _plot_missing(mr, out_dir, threshold=0.60):
    mr = mr.sort_values("pct_missing", ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(6, len(mr) * 0.45)))
    colors = ["#d32f2f" if v > threshold else "#1976d2" for v in mr["pct_missing"]]
    bars = ax.barh(range(len(mr)), mr["pct_missing"] * 100, color=colors)

    ax.set_yticks(range(len(mr)))
    ax.set_yticklabels(mr.index)
    ax.axvline(threshold * 100, color="red", linestyle="--", linewidth=2, label=f"Threshold ({threshold*100:.0f}%)")

    for bar, value in zip(bars, mr["pct_missing"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{value*100:.1f}%", va="center", fontsize=9)

    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing Rate per Variable")
    ax.legend()
    ax.set_xlim(0, 100)
    _save_fig(fig, "missing_rates", out_dir)


def _annotate_heatmap(ax, data, fmt=".4f", fontsize=10, fontweight="bold"):
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        return

    finite = arr[np.isfinite(arr)]
    threshold = (finite.min() + finite.max()) / 2.0 if finite.size else 0.0

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            if not np.isfinite(value):
                continue

            if fmt == "d":
                text = f"{int(round(value))}"
            else:
                text = format(value, fmt)

            color = "white" if value > threshold else "#1a1a1a"
            ax.text(
                j + 0.5,
                i + 0.5,
                text,
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize,
                fontweight=fontweight,
            )


def _plot_heatmaps(summary, out_dir):
    metrics = {
        "accuracy_mean": "Acuracia",
        "f1_weighted_mean": "F1-Score (Weighted)",
        "f1_macro_mean": "F1-Score (Macro)",
        "auc_weighted_mean": "AUC (Weighted)",
        "recall_weighted_mean": "Revocacao",
    }
    for col, label in metrics.items():
        if col not in summary.columns:
            continue

        pivot = summary.pivot(index="imputer", columns="classifier", values=col)
        pivot = pivot.reindex(index=_ordered(IMP_ORDER, pivot.index), columns=_ordered(CLF_ORDER, pivot.columns))
        pivot.index = [IMP_LABELS.get(x, x) for x in pivot.index]
        pivot.columns = [CLF_LABELS.get(x, x) for x in pivot.columns]

        fig, ax = plt.subplots(figsize=(9, 6))
        sns.heatmap(
            pivot,
            annot=False,
            cmap="YlOrRd",
            ax=ax,
            linewidths=1,
        )
        _annotate_heatmap(ax, pivot.values, fmt=".4f", fontsize=10, fontweight="bold")
        ax.set_title(f"{label} - Imputador x Classificador")
        ax.set_ylabel("Imputador")
        ax.tick_params(axis="y", rotation=0)
        _save_fig(fig, f"heatmap_{col.replace('_mean', '')}", out_dir)


def _plot_boxplots(df, out_dir):
    metrics = {
        "accuracy": "Acuracia",
        "f1_weighted": "F1-Score",
        "auc_weighted": "AUC",
        "recall_weighted": "Revocacao",
    }
    available = {k: v for k, v in metrics.items() if k in df.columns}

    dfx = df.copy()
    dfx["method"] = dfx["imputer"].map(lambda x: IMP_LABELS.get(x, x)) + "\n+ " + dfx["classifier"].map(
        lambda x: CLF_LABELS.get(x, x)
    )

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    for ax, (col, label) in zip(axes.flat, available.items()):
        data = dfx[["method", col]].dropna()
        order = data.groupby("method")[col].median().sort_values(ascending=False).index
        sns.boxplot(data=data, x="method", y=col, order=order, ax=ax, palette="Set2")
        sns.stripplot(data=data, x="method", y=col, order=order, ax=ax, color="black", size=4, alpha=0.5)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Metric distribution across 5-fold CV", fontsize=15, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, "boxplots_metrics", out_dir)


def _plot_timing(summary, out_dir):
    tcols = {
        c: c.replace("time_", "").replace("_mean", "")
        for c in summary.columns
        if c.startswith("time_") and c.endswith("_mean") and c != "time_total_mean"
    }
    if not tcols:
        return

    dfx = summary.copy()
    dfx["method"] = dfx["imputer"].map(lambda x: IMP_LABELS.get(x, x)) + " + " + dfx["classifier"].map(
        lambda x: CLF_LABELS.get(x, x)
    )
    if "time_total_mean" in dfx.columns:
        dfx = dfx.sort_values("time_total_mean")

    fig, ax = plt.subplots(figsize=(14, max(8, len(dfx) * 0.5)))
    bottom = np.zeros(len(dfx))
    colors = plt.cm.Set3(np.linspace(0.1, 0.9, len(tcols)))

    for (col, label), color in zip(tcols.items(), colors):
        vals = dfx[col].fillna(0).values
        ax.barh(dfx["method"], vals, left=bottom, label=label, color=color)
        bottom += vals

    ax.set_xlabel("Time (seconds)")
    ax.set_title("Compute time by method and stage")
    ax.legend(loc="lower right")
    _save_fig(fig, "timing_stacked", out_dir)


from src.metrics_utils import coerce_confusion_matrix as _coerce_confusion_matrix


def _plot_confusion(detailed, summary, out_dir, target_map):
    best = summary.loc[summary["f1_weighted_mean"].idxmax()]
    inv = {int(v): str(k) for k, v in target_map.items()}

    expected_classes = (max(inv.keys()) + 1) if inv else None
    matrices = []

    for row in detailed:
        if (
            row.get("imputer") == best["imputer"]
            and row.get("classifier") == best["classifier"]
            and "confusion_matrix" in row
            and not row.get("error")
        ):
            cm = _coerce_confusion_matrix(row["confusion_matrix"], expected_classes=expected_classes)
            if cm is not None:
                matrices.append(cm)

    if not matrices:
        return

    if expected_classes is None:
        expected_classes = max(cm.shape[0] for cm in matrices)

    exact = [cm for cm in matrices if cm.shape == (expected_classes, expected_classes)]
    if exact:
        cm = np.sum(exact, axis=0)
    else:
        base = max(matrices, key=lambda x: x.shape[0])
        cm = np.zeros((expected_classes, expected_classes), dtype=int)
        r = min(base.shape[0], expected_classes)
        c = min(base.shape[1], expected_classes)
        cm[:r, :c] = base[:r, :c]

    labels = [inv.get(i, str(i)) for i in range(expected_classes)]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0], linewidths=0.5)
    _annotate_heatmap(axes[0], cm, fmt="d")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Absolute")

    sns.heatmap(
        cm_norm,
        annot=False,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
        linewidths=0.5,
        vmin=0,
        vmax=1,
    )
    _annotate_heatmap(axes[1], cm_norm, fmt=".2%")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Normalized by class")

    imp_label = IMP_LABELS.get(best["imputer"], best["imputer"])
    clf_label = CLF_LABELS.get(best["classifier"], best["classifier"])
    plt.suptitle(f"Best model: {imp_label} + {clf_label}", fontweight="bold")
    _save_fig(fig, "confusion_matrix_best", out_dir)


def _plot_per_class_f1(detailed, summary, out_dir, target_map):
    inv = {int(v): str(k) for k, v in target_map.items()}
    best_per_imp = summary.loc[summary.groupby("imputer")["f1_weighted_mean"].idxmax()]

    class_keys = sorted(
        {
            key
            for row in detailed
            if "classification_report" in row
            for key, value in row["classification_report"].items()
            if isinstance(value, dict)
            and "precision" in value
            and key not in ("accuracy", "macro avg", "weighted avg")
        },
        key=str,
    )

    if not class_keys:
        return

    fig, ax = plt.subplots(figsize=(13, 7))
    width = 0.12
    x = np.arange(len(class_keys))

    for i, (_, row) in enumerate(best_per_imp.iterrows()):
        f1_scores = {k: [] for k in class_keys}
        for item in detailed:
            if (
                item.get("imputer") == row["imputer"]
                and item.get("classifier") == row["classifier"]
                and "classification_report" in item
                and not item.get("error")
            ):
                for key in class_keys:
                    if key in item["classification_report"]:
                        f1_scores[key].append(item["classification_report"][key].get("f1-score", 0))

        means = [np.mean(f1_scores[k]) if f1_scores[k] else 0 for k in class_keys]
        color = IMP_COLORS.get(row["imputer"], f"C{i}")
        label = f"{IMP_LABELS.get(row['imputer'], row['imputer'])} ({CLF_LABELS.get(row['classifier'], row['classifier'])})"
        ax.bar(x + i * width, means, width, label=label, color=color)

    labels = [inv.get(int(k), k) if str(k).isdigit() else k for k in class_keys]
    ax.set_xticks(x + width * len(best_per_imp) / 2)
    ax.set_xticklabels([f"Est. {lbl}" for lbl in labels])
    ax.set_ylabel("F1-Score")
    ax.set_title("Per-class F1 by imputer")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    _save_fig(fig, "per_class_f1", out_dir)


def _plot_radar(summary, out_dir):
    best_per_imp = summary.loc[summary.groupby("imputer")["f1_weighted_mean"].idxmax()]
    metrics = ["accuracy_mean", "recall_weighted_mean", "f1_weighted_mean", "auc_weighted_mean"]
    labels = ["Acuracia", "Revocacao", "F1", "AUC"]

    available = [m for m in metrics if m in summary.columns]
    labels = labels[: len(available)]

    if not available:
        return

    n = len(available)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist() + [0]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    for _, row in best_per_imp.iterrows():
        values = [row[m] for m in available]
        values += [values[0]]
        color = IMP_COLORS.get(row["imputer"])
        label = f"{IMP_LABELS.get(row['imputer'], row['imputer'])} ({CLF_LABELS.get(row['classifier'], row['classifier'])})"
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    ax.set_title("Comparison of best models", fontweight="bold", pad=30)
    ax.legend(bbox_to_anchor=(1.3, 1.1), fontsize=9)
    _save_fig(fig, "radar_best", out_dir)


def run_analysis(config_path="config/config.yaml", cfg=None):
    if cfg is None:
        cfg = load_config(config_path)

    res_dir = Path(cfg["paths"]["results_raw"])
    tab_dir = Path(cfg["paths"]["results_tables"])
    fig_dir = Path(cfg["paths"]["results_figures"])

    tab_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(res_dir / "all_results.csv")
    if "error" in df.columns:
        df_valid = df[df["error"].isna()].copy()
    else:
        df_valid = df.copy()

    detailed = []
    detailed_path = res_dir / "all_results_detailed.json"
    if detailed_path.exists():
        with open(detailed_path, "r", encoding="utf-8") as f:
            detailed = json.load(f)

    with open(tab_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    target_map = meta.get("target_mapping", {})

    missing_report = None
    for missing_name in ("missing_report_raw.csv", "missing_report_post_filter.csv"):
        missing_path = tab_dir / missing_name
        if missing_path.exists():
            missing_report = pd.read_csv(missing_path, index_col=0)
            log.info("Missing report source for plot: %s", missing_name)
            break

    summary = _aggregate(df_valid)
    summary.to_csv(tab_dir / "summary.csv", index=False)

    _main_table(summary, tab_dir)
    if detailed:
        _per_class_table(detailed, summary, tab_dir, target_map)

    _ranking(summary, tab_dir)
    _stat_tests(df_valid, tab_dir, "f1_weighted")
    _stat_tests(df_valid, tab_dir, "f1_macro")
    _stat_tests(df_valid, tab_dir, "auc_weighted")

    if missing_report is not None:
        _plot_missing(missing_report, fig_dir)
    _plot_heatmaps(summary, fig_dir)
    _plot_boxplots(df_valid, fig_dir)
    _plot_timing(summary, fig_dir)

    if detailed:
        _plot_confusion(detailed, summary, fig_dir, target_map)
        _plot_per_class_f1(detailed, summary, fig_dir, target_map)

    _plot_radar(summary, fig_dir)

    top = summary.nlargest(3, "f1_weighted_mean")
    for pos, (_, row) in enumerate(top.iterrows(), start=1):
        imp_label = IMP_LABELS.get(row["imputer"], row["imputer"])
        clf_label = CLF_LABELS.get(row["classifier"], row["classifier"])
        log.info(
            "Top %d: %s + %s | F1=%.4f | AUC=%.4f",
            pos,
            imp_label,
            clf_label,
            row.get("f1_weighted_mean", np.nan),
            row.get("auc_weighted_mean", np.nan),
        )

    log.info("Tables generated: %d CSV | %d TeX", len(list(tab_dir.glob("*.csv"))), len(list(tab_dir.glob("*.tex"))))
    log.info("Figures generated: %d PNG | %d PDF", len(list(fig_dir.glob("*.png"))), len(list(fig_dir.glob("*.pdf"))))
    log.info("Analysis finished.")
