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
from scipy.stats import t


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

# Plot style tuned for IEEE Access journal quality
# Meets publication standards for font, sizing, and output quality
plt.rcParams.update({
    # Figure DPI and output
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "savefig.dpi": 600,  # 600 dpi for publication
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,  # Minimal padding
    "savefig.format": "png",
    
    # Font settings: Times New Roman for IEEE Access
    "font.family": 'sans-serif',
    "font.sans-serif": ['Source Sans 3', 'Helvetica', 'Arial', 'DejaVu Sans'],
    "font.size": 8,  # Base font size
    "axes.labelsize": 8,        # Axis labels: 8 pt
    "axes.titlesize": 9,        # Subplot titles: 9 pt (a), (b), etc.
    "axes.titleweight": "bold",
    "legend.fontsize": 7,       # Legend: 7 pt
    "xtick.labelsize": 7,       # Tick labels: 7 pt
    "ytick.labelsize": 7,

    # Axes styling
    "axes.linewidth": 0.5,      # Thinner axis lines
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.axisbelow": True,     # Grid behind bars
    "grid.linestyle": "--",
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,          # Light gridlines
    "axes.spines.top": False,   # Remove top/right spines
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,

    # Ticks: inward, ~3 pt length
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,      # 3 pt tick length
    "ytick.major.size": 3,
    "xtick.direction": "in",    # Inward ticks
    "ytick.direction": "in",
    "xtick.minor.size": 0,
    "ytick.minor.size": 0,

    # Lines and markers
    "lines.linewidth": 1.0,
    "lines.markersize": 5,

    # Error bars: 2 pt cap, 0.75 pt linewidth
    "errorbar.capsize": 2,
})

# Color palette: Okabe-Ito colorblind-friendly palette
# Distinguishable by people with color blindness (protanopia, deuteranopia, tritanopia)
METHOD_COLORS = {
    "Baseline A - Full RF": "#0173B2",        # blue
    "Baseline B - Fixed Cascade": "#DE8F05",  # orange
    "Cascade RF (Two-Stage)": "#CC78BC",      # purple
    "LazyRF": "#CA9161",                       # tan
    "Baseline C - QuickScorer": "#56B4E9",    # light blue
    "Full GBM": "#029E73",                     # green
    "LazyGBM": "#ECE133",                      # yellow
    "Baseline D - Full BranchyNet": "#D5A6BD", # light purple
    "Baseline D - Early Exit BranchyNet": "#999933",  # olive
}

# Simplified hatching patterns: max 4 distinct styles for B&W printing
# Differentiate primarily by color, secondarily by hatch
METHOD_HATCHES = {
    "Baseline A - Full RF": "",        # no hatch
    "Baseline B - Fixed Cascade": "//", # diagonal
    "Cascade RF (Two-Stage)": "",       # no hatch
    "LazyRF": "",                       # no hatch
    "Baseline C - QuickScorer": "\\\\\\\\",   # diagonal (other direction)
    "Full GBM": "",                    # no hatch
    "LazyGBM": "..",                   # dots
    "Baseline D - Full BranchyNet": "xx",  # cross-hatch
    "Baseline D - Early Exit BranchyNet": "++",  # plus
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

    # Add 95% CI columns using the Student-t critical value for the observed run count.
    critical = pd.Series(t.ppf(0.975, grouped["n"] - 1), index=grouped.index)
    critical = critical.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    critical = critical.where(grouped["n"] > 1, 0.0)
    for m in metrics:
        mean_col = f"{m}_mean"
        std_col = f"{m}_std"
        ci_col = f"{m}_ci95"
        grouped[ci_col] = critical * grouped[std_col] / np.sqrt(grouped["n"])

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
            "lazy_gbm_threshold": "Stability threshold $\\gamma$",
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
                              show_legend: bool = False) -> None:
    """
    Create a 2x2 subplot figure meeting IEEE Access publication standards:
      - Each panel: one dataset labeled (a), (b), (c), (d) in lowercase
      - Bars: single unified color (steel blue) with 95% CI error bars
      - Method names labeled on x-axis (rotated for readability)
      - Cropped y-axis ranges, light horizontal gridlines
      - Proper font sizing: subplot titles 9pt, axis labels 8pt, ticks 7pt
      - Bar edges: black, linewidth 0.5
      - Error bars: cap size 2pt, linewidth 0.75pt
      - Export as PDF (vector) and 600 dpi PNG
    """
    datasets = sorted(perf_summary["dataset"].unique())
    n_datasets = len(datasets)

    # IEEE Access double-column figure width: 7.16 inches
    # Height adjusted for 2×2 grid (no legend space needed)
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8))
    axes_flat = axes.flatten()

    # Per-dataset y-limits to crop to data range (avoid wasting space)
    # Compute dataset-specific ranges
    dataset_limits = {}
    for dataset in datasets:
        sub = perf_summary[perf_summary["dataset"] == dataset]
        means = sub[f"{metric}_mean"].values
        cis = sub[f"{metric}_ci95"].values
        y_min = max(0.0, (means - cis).min() * 0.98)  # Small margin below min
        y_max = (means + cis).max() * 1.02            # Small margin above max
        if metric == "accuracy":
            y_max = min(1.0, y_max)
        dataset_limits[dataset] = (y_min, y_max)

    # Single unified color for all bars (steel blue)
    bar_color = "#4472C4"

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        sub = perf_summary[perf_summary["dataset"] == dataset].copy()
        sub = sub.sort_values("method", key=lambda s: s.map(method_sort_key))

        x = np.arange(len(sub))
        means = sub[f"{metric}_mean"].values
        ci = sub[f"{metric}_ci95"].values
        methods = sub["method"].tolist()

        # Draw bars with single unified color and black edges (0.5 pt)
        bar_width = 0.7
        bars = ax.bar(x, means, width=bar_width, color=bar_color,
                      edgecolor="black", linewidth=0.5)  # 0.5 pt edges

        # Add error bars: 2 pt cap, 0.75 pt linewidth
        ax.errorbar(x, means, yerr=ci, fmt='none', color='black',
                    capsize=2, capthick=0.75, elinewidth=0.75)  # 0.75 pt lines

        # Subplot title: lowercase letters (a), (b), etc., 9 pt bold
        ax.set_title(f"({chr(97 + idx)}) {dataset}", fontweight='bold', pad=8)

        # Display method names on x-axis only for bottom row (bottom-left: idx 2, bottom-right: idx 3)
        ax.set_xticks(x)
        if idx >= 2:  # Bottom row only
            method_labels = [short_label(m) for m in methods]
            ax.set_xticklabels(method_labels, rotation=45, ha='right', fontsize=7)
        else:  # Top row: no x-axis labels
            ax.set_xticklabels([])

        # Y-axis label on left column only (8 pt via rcParams)
        if idx % 2 == 0:
            ax.set_ylabel(ylabel)

        # Apply dataset-specific y-limits (cropped to data range)
        # y_min, y_max = dataset_limits[dataset]
        # ax.set_ylim(y_min, y_max)
        if metric == "accuracy":
            ax.set_ylim(0.7, 1.0)
        else:
            ax.set_ylim(0, 100)

        # Ensure light horizontal gridlines are visible
        ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=0)

        # Remove top and right spines for clean appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    # Hide unused subplots if fewer than 4 datasets
    for j in range(n_datasets, 4):
        fig.delaxes(axes_flat[j])

    # Adjust layout (no legend space needed)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.25)

    # Save as both PDF (vector) and high-resolution PNG
    # PDF: vector format with embedded fonts for publication
    fig.savefig(
        FIG_DIR / f"{filename}.pdf",
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.01,
    )
    # PNG: 600 dpi for print-quality raster
    fig.savefig(
        FIG_DIR / f"{filename}.png",
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.01,
    )
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
