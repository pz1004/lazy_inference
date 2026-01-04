#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate summary tables and 2x2 subplot figures for thesis using:
  - raw_results_perf.csv  (main experiments)
  - raw_results_sweep.csv (threshold sweeps for LazyRF / LazyGBM)

Outputs
-------
Tables (LaTeX):
  tables/main_results_by_dataset_method.tex
  tables/summary_by_method.tex
  tables/lazy_rf_sweep_by_dataset.tex
  tables/lazy_gbm_sweep_by_dataset.tex
  tables/timing_results.tex

Figures (PDF + PNG), each as 2x2 subplots (one panel per dataset):
  figures/accuracy_by_method_2x2.pdf / .png
  figures/work_by_method_2x2.pdf / .png
  figures/lazy_rf_tradeoff_2x2.pdf / .png
  figures/lazy_gbm_tradeoff_2x2.pdf / .png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------

# Input files (adjust paths if needed)
PERF_PATH = Path("raw_results_perf.csv")
SWEEP_PATH = Path("raw_results_sweep.csv")

# Output directories
TABLE_DIR = Path("tables")
FIG_DIR = Path("figures")
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Plot style tuned for academic papers (journal quality)
plt.rcParams.update({
    # Figure
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,

    # Font settings (use serif for journals)
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # Axes
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.axisbelow": True,
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,

    # Ticks
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Lines and markers
    "lines.linewidth": 1.5,
    "lines.markersize": 6,

    # Error bars
    "errorbar.capsize": 3,
})

# Color palette for methods (colorblind-friendly)
METHOD_COLORS = {
    "Baseline A - Full RF": "#1f77b4",        # blue
    "Baseline B - Fixed Cascade": "#ff7f0e",  # orange
    "Cascade RF (Two-Stage)": "#2ca02c",      # green
    "LazyRF": "#d62728",                       # red
    "Baseline C - QuickScorer": "#9467bd",    # purple
    "Full GBM": "#8c564b",                     # brown
    "LazyGBM": "#e377c2",                      # pink
    "Baseline D - Full BranchyNet": "#7f7f7f", # gray
    "Baseline D - Early Exit BranchyNet": "#bcbd22",  # olive
}

# Hatching patterns for additional distinction (optional for B&W printing)
METHOD_HATCHES = {
    "Baseline A - Full RF": "",
    "Baseline B - Fixed Cascade": "//",
    "Cascade RF (Two-Stage)": "\\\\",
    "LazyRF": "",
    "Baseline C - QuickScorer": "xx",
    "Full GBM": "",
    "LazyGBM": "..",
    "Baseline D - Full BranchyNet": "++",
    "Baseline D - Early Exit BranchyNet": "||",
}

# Canonical method ordering for plots / tables
METHOD_ORDER = [
    "Baseline A - Full RF",
    "Baseline B - Fixed Cascade",
    "Cascade RF (Two-Stage)",
    "LazyRF",
    "Baseline C - QuickScorer",
    "Full GBM",
    "LazyGBM",
    "Baseline D - Full BranchyNet",
    "Baseline D - Early Exit BranchyNet",
]

# Shortened labels for plotting
SHORT_METHOD_LABELS = {
    "Baseline A - Full RF": "Full RF",
    "Baseline B - Fixed Cascade": "Fixed Cascade",
    "Cascade RF (Two-Stage)": "2-Stage RF",
    "LazyRF": "LazyRF",
    "Baseline C - QuickScorer": "QuickScorer",
    "Full GBM": "Full GBM",
    "LazyGBM": "LazyGBM",
    "Baseline D - Full BranchyNet": "Full BranchyNet",
    "Baseline D - Early Exit BranchyNet": "Early-Exit BranchyNet",
}


# ----------------------------------------------------------------------
# 2. Utility functions
# ----------------------------------------------------------------------

def method_sort_key(method_name: str) -> int:
    """
    Map method name to an integer sort key based on METHOD_ORDER.
    Unknown methods are placed at the end.
    """
    try:
        return METHOD_ORDER.index(method_name)
    except ValueError:
        return len(METHOD_ORDER)


def short_label(method_name: str) -> str:
    """
    Return a shortened label for a method name for plotting.
    """
    return SHORT_METHOD_LABELS.get(method_name, method_name)


def compute_summary(df: pd.DataFrame,
                    group_cols,
                    metrics=("accuracy", "avg_work_units",
                             "energy_reduction", "inference_time", "speedup")) -> pd.DataFrame:
    """
    Group by `group_cols` and compute mean, std, and 95% CI for selected metrics.
    Returns a flat DataFrame with columns: group_cols, metric_mean, metric_std, metric_ci95, n.
    """
    # Keep only metrics that actually exist in df
    metrics = [m for m in metrics if m in df.columns]

    agg_dict = {m: ["mean", "std"] for m in metrics}
    # We assume there is a 'run' column; if not, you can adapt this.
    agg_dict["run"] = ["count"]

    grouped = df.groupby(group_cols).agg(agg_dict)
    # Flatten MultiIndex columns
    grouped.columns = ["{}_{}".format(col[0], col[1]) for col in grouped.columns]
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={"run_count": "n"})

    # Add 95% CI columns
    for m in metrics:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        ci_col = f"{m}_ci95"
        grouped[ci_col] = 1.96 * grouped[std_col] / np.sqrt(grouped["n"])

    return grouped


def format_mean_pm_ci(row: pd.Series, mean_col: str, ci_col: str, digits=4) -> str:
    """
    Format mean ± CI as a LaTeX-friendly string: '0.9046 ± 0.0012'.
    """
    return f"{row[mean_col]:.{digits}f} $\\pm$ {row[ci_col]:.{digits}f}"


# ----------------------------------------------------------------------
# 3. Main-results tables
# ----------------------------------------------------------------------

def make_main_results_tables(perf_df: pd.DataFrame) -> None:
    """
    Create main LaTeX tables:
      - By dataset × method
      - Aggregated by method
    """
    # (a) by dataset × method
    by_dataset_method = compute_summary(
        perf_df,
        group_cols=["dataset", "method"],
        metrics=("accuracy", "avg_work_units", "energy_reduction",
                 "inference_time", "speedup"),
    )

    # Sort for readability
    by_dataset_method = by_dataset_method.sort_values(
        ["dataset", "method"],
        key=lambda s: (s if s.name != "method"
                       else s.map(method_sort_key))
    )

    # Create formatted columns for LaTeX (accuracy ± CI, work, energy)
    by_dataset_method["accuracy_tex"] = by_dataset_method.apply(
        lambda r: format_mean_pm_ci(r, "accuracy_mean", "accuracy_ci95", digits=4), axis=1
    )
    by_dataset_method["work_tex"] = by_dataset_method["avg_work_units_mean"].map(
        lambda v: f"{v:.2f}"
    )
    by_dataset_method["energy_tex"] = by_dataset_method["energy_reduction_mean"].map(
        lambda v: f"{v:.2f}"
    )

    # Select columns for LaTeX table
    latex_cols = [
        "dataset", "method",
        "accuracy_tex", "work_tex", "energy_tex"
    ]
    main_table = by_dataset_method[latex_cols].copy()
    main_table = main_table.rename(columns={
        "dataset": "Dataset",
        "method": "Method",
        "accuracy_tex": "Accuracy (mean $\\pm$ 95\\% CI)",
        "work_tex": "Avg. work units",
        "energy_tex": "Relative work reduction",
    })

    main_tex = main_table.to_latex(
        index=False,
        escape=False,
        column_format="llccc",
        longtable=False,
        caption="Main results by dataset and method.",
        label="tab:main-results",
    )
    (TABLE_DIR / "main_results_by_dataset_method.tex").write_text(main_tex)

    # (b) aggregated by method (across datasets)
    by_method = compute_summary(
        perf_df,
        group_cols=["method"],
        metrics=("accuracy", "avg_work_units", "energy_reduction",
                 "inference_time", "speedup"),
    )
    by_method = by_method.sort_values(
        "method", key=lambda s: s.map(method_sort_key)
    )

    by_method["accuracy_tex"] = by_method.apply(
        lambda r: format_mean_pm_ci(r, "accuracy_mean", "accuracy_ci95", digits=4), axis=1
    )
    by_method["work_tex"] = by_method["avg_work_units_mean"].map(
        lambda v: f"{v:.2f}"
    )
    by_method["energy_tex"] = by_method["energy_reduction_mean"].map(
        lambda v: f"{v:.2f}"
    )

    latex_cols_m = [
        "method", "accuracy_tex", "work_tex", "energy_tex",
    ]
    agg_table = by_method[latex_cols_m].copy()
    agg_table = agg_table.rename(columns={
        "method": "Method",
        "accuracy_tex": "Accuracy (mean $\\pm$ 95\\% CI)",
        "work_tex": "Avg. work units",
        "energy_tex": "Relative work reduction",
    })

    agg_tex = agg_table.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
        caption="Aggregate performance across datasets by method.",
        label="tab:summary-by-method",
    )
    (TABLE_DIR / "summary_by_method.tex").write_text(agg_tex)


# ----------------------------------------------------------------------
# 4. Sweep tables (LazyRF / LazyGBM)
# ----------------------------------------------------------------------

def make_sweep_tables(sweep_df: pd.DataFrame) -> None:
    """
    Create LaTeX tables summarizing threshold sweeps for:
      - LazyRF (by dataset × lazy_rf_threshold)
      - LazyGBM (by dataset × lazy_gbm_threshold)
    """
    # LazyRF
    if "lazy_rf_threshold" in sweep_df.columns:
        lazy_rf_df = sweep_df[sweep_df["lazy_rf_threshold"].notna()].copy()
    else:
        lazy_rf_df = pd.DataFrame()

    if not lazy_rf_df.empty:
        rf_summary = compute_summary(
            lazy_rf_df,
            group_cols=["dataset", "lazy_rf_threshold"],
            metrics=("accuracy", "avg_work_units", "energy_reduction", "speedup"),
        )
        rf_summary = rf_summary.sort_values(["dataset", "lazy_rf_threshold"])

        rf_summary["accuracy_tex"] = rf_summary.apply(
            lambda r: format_mean_pm_ci(r, "accuracy_mean", "accuracy_ci95", digits=4), axis=1
        )
        rf_summary["work_tex"] = rf_summary["avg_work_units_mean"].map(
            lambda v: f"{v:.2f}"
        )
        rf_summary["energy_tex"] = rf_summary["energy_reduction_mean"].map(
            lambda v: f"{v:.2f}"
        )

        rf_table = rf_summary[[
            "dataset", "lazy_rf_threshold",
            "accuracy_tex", "work_tex", "energy_tex"
        ]].copy()
        rf_table = rf_table.rename(columns={
            "dataset": "Dataset",
            "lazy_rf_threshold": "Threshold $\\alpha$",
            "accuracy_tex": "Accuracy (mean $\\pm$ 95\\% CI)",
            "work_tex": "Avg. work units",
            "energy_tex": "Relative work reduction",
        })

        rf_tex = rf_table.to_latex(
            index=False,
            escape=False,
            column_format="lcccc",
            caption="LazyRF threshold sweep: accuracy--efficiency summary.",
            label="tab:lazy-rf-sweep",
        )
        (TABLE_DIR / "lazy_rf_sweep_by_dataset.tex").write_text(rf_tex)

    # LazyGBM
    if "lazy_gbm_threshold" in sweep_df.columns:
        lazy_gbm_df = sweep_df[sweep_df["lazy_gbm_threshold"].notna()].copy()
    else:
        lazy_gbm_df = pd.DataFrame()

    if not lazy_gbm_df.empty:
        gbm_summary = compute_summary(
            lazy_gbm_df,
            group_cols=["dataset", "lazy_gbm_threshold"],
            metrics=("accuracy", "avg_work_units", "energy_reduction", "speedup"),
        )
        gbm_summary = gbm_summary.sort_values(["dataset", "lazy_gbm_threshold"])

        gbm_summary["accuracy_tex"] = gbm_summary.apply(
            lambda r: format_mean_pm_ci(r, "accuracy_mean", "accuracy_ci95", digits=4), axis=1
        )
        gbm_summary["work_tex"] = gbm_summary["avg_work_units_mean"].map(
            lambda v: f"{v:.2f}"
        )
        gbm_summary["energy_tex"] = gbm_summary["energy_reduction_mean"].map(
            lambda v: f"{v:.2f}"
        )

        gbm_table = gbm_summary[[
            "dataset", "lazy_gbm_threshold",
            "accuracy_tex", "work_tex", "energy_tex"
        ]].copy()
        gbm_table = gbm_table.rename(columns={
            "dataset": "Dataset",
            "lazy_gbm_threshold": "SPRT threshold",
            "accuracy_tex": "Accuracy (mean $\\pm$ 95\\% CI)",
            "work_tex": "Avg. work units",
            "energy_tex": "Relative work reduction",
        })

        gbm_tex = gbm_table.to_latex(
            index=False,
            escape=False,
            column_format="lcccc",
            caption="LazyGBM threshold sweep: accuracy--efficiency summary.",
            label="tab:lazy-gbm-sweep",
        )
        (TABLE_DIR / "lazy_gbm_sweep_by_dataset.tex").write_text(gbm_tex)


# ----------------------------------------------------------------------
# 5. 2x2 bar-plot figures (accuracy / work by method)
# ----------------------------------------------------------------------

def plot_metric_by_method_2x2(perf_summary: pd.DataFrame,
                              metric: str,
                              ylabel: str,
                              filename: str,
                              show_legend: bool = True) -> None:
    """
    Create a 2x2 subplot figure (journal quality):
      - Each panel: one dataset
      - Bars: methods (with 95% CI error bars)
      - Uses consistent colors and hatching for B&W printing
      - Shared legend at bottom
    """
    datasets = sorted(perf_summary["dataset"].unique())
    n_datasets = len(datasets)

    # Set up 2x2 grid with proper sizing for two-column journal format
    # Extra height for legend at bottom
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))
    axes_flat = axes.flatten()

    # Global y-limits for shared comparison
    all_means = perf_summary[f"{metric}_mean"].values
    all_cis = perf_summary[f"{metric}_ci95"].values
    y_min = max(0.0, (all_means - all_cis).min() * 0.95)
    y_max = (all_means + all_cis).max() * 1.05
    if metric == "accuracy":
        y_max = min(1.0, y_max)

    # Track methods for legend (preserve order)
    legend_handles = []
    legend_labels = []
    seen_methods = set()

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        sub = perf_summary[perf_summary["dataset"] == dataset].copy()
        sub = sub.sort_values("method", key=lambda s: s.map(method_sort_key))

        x = np.arange(len(sub))
        means = sub[f"{metric}_mean"].values
        ci = sub[f"{metric}_ci95"].values
        methods = sub["method"].tolist()

        # Get colors and hatches for each method
        colors = [METHOD_COLORS.get(m, "#333333") for m in methods]
        hatches = [METHOD_HATCHES.get(m, "") for m in methods]

        # Draw bars with individual colors and hatches
        bar_width = 0.7
        bars = ax.bar(x, means, width=bar_width, color=colors,
                      edgecolor="black", linewidth=0.8)

        # Apply hatching patterns and collect legend handles
        for bar, hatch, method in zip(bars, hatches, methods):
            bar.set_hatch(hatch)
            if method not in seen_methods:
                seen_methods.add(method)
                legend_handles.append(bar)
                legend_labels.append(short_label(method))

        # Add error bars separately for better control
        ax.errorbar(x, means, yerr=ci, fmt='none', color='black',
                    capsize=3, capthick=0.8, elinewidth=0.8)

        # Subplot title (dataset name)
        ax.set_title(f"({chr(97 + idx)}) {dataset}", fontweight='bold', pad=8)

        # Hide x-tick labels (legend serves this purpose)
        ax.set_xticks(x)
        ax.set_xticklabels([])

        # Y-axis label on left column only
        if idx % 2 == 0:
            ax.set_ylabel(ylabel)

        # Apply consistent y-limits
        ax.set_ylim(y_min, y_max)

        # Remove top and right spines (already in rcParams but ensure)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots if fewer than 4 datasets
    for j in range(n_datasets, 4):
        fig.delaxes(axes_flat[j])

    # Reorder legend handles/labels by METHOD_ORDER
    sorted_pairs = sorted(
        zip(legend_handles, legend_labels),
        key=lambda x: method_sort_key(
            next((k for k, v in SHORT_METHOD_LABELS.items() if v == x[1]), x[1])
        )
    )
    legend_handles, legend_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])

    # Add shared legend at bottom
    if show_legend and legend_handles:
        fig.legend(
            legend_handles, legend_labels,
            loc='lower center',
            ncol=min(4, len(legend_handles)),
            bbox_to_anchor=(0.5, -0.02),
            frameon=True,
            fancybox=False,
            edgecolor='black',
            fontsize=8,
        )

    # Adjust layout with space for legend
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.subplots_adjust(hspace=0.25, wspace=0.15)

    # Save as both PDF (vector) and PNG (raster)
    fig.savefig(FIG_DIR / f"{filename}.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_main_result_figures(perf_df: pd.DataFrame) -> None:
    """
    Create 2x2 subplot figures for:
      - accuracy by method
      - avg_work_units by method
    """
    summary = compute_summary(
        perf_df,
        group_cols=["dataset", "method"],
        metrics=("accuracy", "avg_work_units"),
    )

    # Accuracy by method (2x2)
    plot_metric_by_method_2x2(
        summary, metric="accuracy",
        ylabel="Accuracy",
        filename="accuracy_by_method_2x2",
    )

    # Average work units by method (2x2)
    plot_metric_by_method_2x2(
        summary, metric="avg_work_units",
        ylabel="Avg. work units",
        filename="work_by_method_2x2",
    )


# ----------------------------------------------------------------------
# 6. 2x2 trade-off figures (LazyRF / LazyGBM)
# ----------------------------------------------------------------------

def plot_tradeoff_2x2(summary: pd.DataFrame,
                      threshold_col: str,
                      filename: str,
                      suptitle: str) -> None:
    """
    Create a 2x2 subplot figure (journal quality) showing accuracy–efficiency trade-offs:
      - X: avg_work_units_mean (with CI)
      - Y: accuracy_mean (with CI)
      - Each subplot: one dataset; points labeled by threshold
      - Gradient coloring from low to high threshold
    """
    datasets = sorted(summary["dataset"].unique())
    n_datasets = len(datasets)

    # Set up 2x2 grid with proper sizing
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 6))
    axes_flat = axes.flatten()

    # Global y-limits for consistency across panels
    all_acc_means = summary["accuracy_mean"].values
    all_acc_cis = summary["accuracy_ci95"].values
    y_min = max(0.0, (all_acc_means - all_acc_cis).min() * 0.995)
    y_max = min(1.0, (all_acc_means + all_acc_cis).max() * 1.005)

    # Color scheme for threshold gradient (low=blue, high=red)
    cmap = plt.cm.RdYlBu_r  # Red-Yellow-Blue reversed

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        sub = summary[summary["dataset"] == dataset].copy()
        sub = sub.sort_values(threshold_col)

        # Normalize thresholds for colormap
        thresholds = sub[threshold_col].values
        if len(thresholds) > 1:
            norm = plt.Normalize(thresholds.min(), thresholds.max())
        else:
            norm = plt.Normalize(0, 1)

        # Plot line connecting points
        ax.plot(
            sub["avg_work_units_mean"],
            sub["accuracy_mean"],
            linestyle="-",
            linewidth=1.2,
            color="#666666",
            zorder=1,
        )

        # Plot error bars and points with gradient colors
        for _, row in sub.iterrows():
            color = cmap(norm(row[threshold_col]))
            ax.errorbar(
                row["avg_work_units_mean"],
                row["accuracy_mean"],
                xerr=row["avg_work_units_ci95"],
                yerr=row["accuracy_ci95"],
                marker="o",
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.8,
                linestyle="none",
                color=color,
                capsize=3,
                capthick=0.8,
                elinewidth=0.8,
                zorder=2,
            )

        # Annotate each point with threshold value
        for i, (_, row) in enumerate(sub.iterrows()):
            # Alternate annotation position to avoid overlap
            offset_x = 5 if i % 2 == 0 else -25
            offset_y = 5 if i % 2 == 0 else -10
            ha = "left" if i % 2 == 0 else "right"

            ax.annotate(
                f"α={row[threshold_col]:.2f}",
                (row["avg_work_units_mean"], row["accuracy_mean"]),
                textcoords="offset points",
                xytext=(offset_x, offset_y),
                fontsize=7,
                ha=ha,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          edgecolor="none", alpha=0.7),
            )

        # Subplot title with panel label
        ax.set_title(f"({chr(97 + idx)}) {dataset}", fontweight='bold', pad=8)

        # Axis labels
        ax.set_xlabel("Avg. trees evaluated")
        if idx % 2 == 0:
            ax.set_ylabel("Accuracy")

        ax.set_ylim(y_min, y_max)
        ax.ticklabel_format(style="plain", axis="x")

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Hide unused subplots if fewer than 4 datasets
    for j in range(n_datasets, 4):
        fig.delaxes(axes_flat[j])

    # Adjust layout
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    plt.subplots_adjust(hspace=0.35, wspace=0.2)

    # Save as both PDF (vector) and PNG (raster)
    fig.savefig(FIG_DIR / f"{filename}.pdf", format='pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_tradeoff_figures(sweep_df: pd.DataFrame) -> None:
    """
    Create 2x2 accuracy–efficiency trade-off figures:
      - LazyRF (lazy_rf_threshold)
      - LazyGBM (lazy_gbm_threshold)
    """
    # LazyRF
    if "lazy_rf_threshold" in sweep_df.columns:
        lazy_rf_df = sweep_df[sweep_df["lazy_rf_threshold"].notna()].copy()
    else:
        lazy_rf_df = pd.DataFrame()

    if not lazy_rf_df.empty:
        rf_summary = compute_summary(
            lazy_rf_df,
            group_cols=["dataset", "lazy_rf_threshold"],
            metrics=("accuracy", "avg_work_units"),
        )
        plot_tradeoff_2x2(
            rf_summary,
            threshold_col="lazy_rf_threshold",
            filename="lazy_rf_tradeoff_2x2",
            suptitle="LazyRF accuracy–efficiency trade-off",
        )

    # LazyGBM
    if "lazy_gbm_threshold" in sweep_df.columns:
        lazy_gbm_df = sweep_df[sweep_df["lazy_gbm_threshold"].notna()].copy()
    else:
        lazy_gbm_df = pd.DataFrame()

    if not lazy_gbm_df.empty:
        gbm_summary = compute_summary(
            lazy_gbm_df,
            group_cols=["dataset", "lazy_gbm_threshold"],
            metrics=("accuracy", "avg_work_units"),
        )
        plot_tradeoff_2x2(
            gbm_summary,
            threshold_col="lazy_gbm_threshold",
            filename="lazy_gbm_tradeoff_2x2",
            suptitle="LazyGBM accuracy–efficiency trade-off",
        )


# ----------------------------------------------------------------------
# 7. Timing table (inference time, speedup, worst-case latency)
# ----------------------------------------------------------------------

# Ordering for timing table methods
TIMING_METHOD_ORDER = [
    "Baseline A - Full RF",
    "LazyRF",
    "Full GBM",
    "LazyGBM",
    "Baseline D - Full BranchyNet",
    "Baseline D - Early Exit BranchyNet",
]

# Dataset ordering for timing table
TIMING_DATASET_ORDER = ["Covertype", "Credit Card", "Higgs", "MNIST"]


def format_mean_pm_ci_timing(mean: float, ci: float, digits: int = 3) -> str:
    """
    Format mean ± CI as a LaTeX-friendly string for timing table.
    Uses $\\pm$ for proper LaTeX rendering.
    """
    return f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"


def make_timing_table(perf_df: pd.DataFrame) -> None:
    """
    Create LaTeX table for runtime and latency measurements.

    Columns:
      - Dataset
      - Method
      - Inference time (s)
      - Speedup vs. reference
      - Worst-case latency (s)

    Output: tables/timing_results.tex
    """
    # Filter to only include methods in TIMING_METHOD_ORDER
    timing_df = perf_df[perf_df["method"].isin(TIMING_METHOD_ORDER)].copy()

    # Compute summary statistics grouped by dataset and method
    timing_summary = compute_summary(
        timing_df,
        group_cols=["dataset", "method"],
        metrics=("inference_time", "speedup", "worst_case_latency"),
    )

    # Sort by dataset order, then method order within each dataset
    timing_summary["dataset_order"] = timing_summary["dataset"].map(
        lambda x: TIMING_DATASET_ORDER.index(x) if x in TIMING_DATASET_ORDER else len(TIMING_DATASET_ORDER)
    )
    timing_summary["method_order"] = timing_summary["method"].map(
        lambda x: TIMING_METHOD_ORDER.index(x) if x in TIMING_METHOD_ORDER else len(TIMING_METHOD_ORDER)
    )
    timing_summary = timing_summary.sort_values(["dataset_order", "method_order"])

    # Format columns for LaTeX
    timing_summary["inference_time_tex"] = timing_summary.apply(
        lambda r: format_mean_pm_ci_timing(r["inference_time_mean"], r["inference_time_ci95"], digits=3),
        axis=1
    )
    timing_summary["speedup_tex"] = timing_summary.apply(
        lambda r: format_mean_pm_ci_timing(r["speedup_mean"], r["speedup_ci95"], digits=3),
        axis=1
    )
    timing_summary["worst_case_latency_tex"] = timing_summary.apply(
        lambda r: format_mean_pm_ci_timing(r["worst_case_latency_mean"], r["worst_case_latency_ci95"], digits=3),
        axis=1
    )

    # Build the LaTeX table manually for proper formatting with midrules
    latex_lines = []
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"\caption{Runtime and latency measurements (wall-clock seconds for the full test set).")
    latex_lines.append(r"``Speedup'' is the ratio of reference time to candidate time (values $<1$ indicate slower runtime).")
    latex_lines.append(r"``Worst-case latency'' equals the inference time of the corresponding full model (since lazy methods cap at full evaluation; excluding wrapper overhead variation in the current prototype). Mean $\pm$ 95\% CI over 30 seeds.}")
    latex_lines.append(r"\label{tab:timing}")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\begin{tabular}{llccc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"Dataset & Method & Inference time (s) & Speedup vs. reference & Worst-case latency (s) \\")
    latex_lines.append(r"\midrule")

    prev_dataset = None
    for _, row in timing_summary.iterrows():
        dataset = row["dataset"]
        method = row["method"]
        inference_tex = row["inference_time_tex"]
        speedup_tex = row["speedup_tex"]
        latency_tex = row["worst_case_latency_tex"]

        # Add midrule between datasets
        if prev_dataset is not None and dataset != prev_dataset:
            latex_lines.append(r"\midrule")

        # Only show dataset name on first row of each dataset group
        if dataset != prev_dataset:
            dataset_col = dataset
        else:
            dataset_col = ""

        latex_lines.append(f"{dataset_col} & {method} & {inference_tex} & {speedup_tex} & {latency_tex} \\\\")
        prev_dataset = dataset

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"\end{table*}")

    latex_content = "\n".join(latex_lines)
    (TABLE_DIR / "timing_results.tex").write_text(latex_content)


# ----------------------------------------------------------------------
# 8. Main entry point
# ----------------------------------------------------------------------

def main():
    # Load data
    perf_df = pd.read_csv(PERF_PATH)
    sweep_df = pd.read_csv(SWEEP_PATH)

    # 1) Main result tables
    make_main_results_tables(perf_df)

    # 2) Sweep tables
    make_sweep_tables(sweep_df)

    # 3) Timing table (inference time, speedup, worst-case latency)
    make_timing_table(perf_df)

    # 4) Main performance 2x2 figures
    make_main_result_figures(perf_df)

    # 5) Trade-off 2x2 figures
    make_tradeoff_figures(sweep_df)

    print("Tables written to:", TABLE_DIR.resolve())
    print("Figures written to:", FIG_DIR.resolve())


if __name__ == "__main__":
    main()
