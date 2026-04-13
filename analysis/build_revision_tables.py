#!/usr/bin/env python3
"""
Build revision-specific tables from refreshed experiment outputs.

Inputs:
  - main raw-results CSV from run_multi_experiments.py
  - optional LazyRF ablation raw-results CSVs
  - optional LazyGBM ablation raw-results CSVs
  - optional LazyRF calibration summary CSVs

Outputs:
  - tables/revision_runtime_context.csv/.tex
  - tables/revision_lazyrf_ablation.csv/.tex
  - tables/revision_lazygbm_ablation.csv/.tex
  - tables/revision_calibration_summary.csv/.tex
"""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
from typing import Iterable, List

import pandas as pd


DATASET_ORDER = ["Covertype", "Credit Card", "Higgs", "MNIST"]
GBM_VARIANT_ORDER = {
    "certificate_only": 0,
    "certificate_plus_flip": 1,
    "current": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-csv", type=str, required=True)
    parser.add_argument("--rf-ablation-glob", type=str, default="")
    parser.add_argument("--gbm-ablation-glob", type=str, default="")
    parser.add_argument("--calibration-glob", type=str, default="")
    parser.add_argument("--tables-dir", type=str, default="tables")
    return parser.parse_args()


def _load_glob(pattern: str) -> pd.DataFrame:
    if not pattern:
        return pd.DataFrame()
    paths = [Path(p) for p in sorted(glob(pattern))]
    frames: List[pd.DataFrame] = []
    for path in paths:
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _dataset_sort(df: pd.DataFrame) -> pd.DataFrame:
    if "dataset" not in df.columns:
        return df
    ordering = {name: idx for idx, name in enumerate(DATASET_ORDER)}
    return df.sort_values("dataset", key=lambda col: col.map(ordering).fillna(len(ordering)))


def _fmt(value: float, digits: int) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}"


def _write_table(df: pd.DataFrame, out_dir: Path, stem: str, caption: str, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{stem}.csv"
    tex_path = out_dir / f"{stem}.tex"
    df.to_csv(csv_path, index=False)
    tex = df.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
    )
    tex = tex.replace("\\begin{table}", "\\begin{table*}[t]\n\\centering")
    tex = tex.replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}", 1)
    tex = tex.replace("\\end{tabular}", "\\end{tabular}%\n}", 1)
    tex = tex.replace("\\end{table}", "\\end{table*}")
    tex_path.write_text(tex)
    print(f"Wrote {csv_path}")
    print(f"Wrote {tex_path}")


def build_runtime_context(main_df: pd.DataFrame) -> pd.DataFrame:
    subset = main_df[main_df["method"].isin(["LazyRF", "LazyGBM"])].copy()
    grouped = (
        subset.groupby(["dataset", "method"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            disagreement_mean=("disagreement_rate", "mean"),
            runtime_mean=("inference_time", "mean"),
            worst_case_latency_mean=("worst_case_latency", "mean"),
            work_p50_mean=("work_p50", "mean"),
            work_p90_mean=("work_p90", "mean"),
            work_p95_mean=("work_p95", "mean"),
        )
    )
    grouped = _dataset_sort(grouped)
    for col in [
        "accuracy_mean",
        "disagreement_mean",
        "runtime_mean",
        "worst_case_latency_mean",
        "work_p50_mean",
        "work_p90_mean",
        "work_p95_mean",
    ]:
        digits = 2 if "work_" in col else 4
        grouped[col] = grouped[col].map(lambda v, d=digits: _fmt(v, d))
    return grouped.rename(
        columns={
            "dataset": "Dataset",
            "method": "Method",
            "accuracy_mean": "Accuracy",
            "disagreement_mean": "Disagreement",
            "runtime_mean": "Mean runtime (s)",
            "worst_case_latency_mean": "Full-model latency (s)",
            "work_p50_mean": "Work p50",
            "work_p90_mean": "Work p90",
            "work_p95_mean": "Work p95",
        }
    )


def build_rf_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    subset = df[df["method"].str.startswith("LazyRF")].copy()
    grouped = (
        subset.groupby(["dataset", "block_size", "min_trees"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            work_mean=("avg_work_units", "mean"),
            disagreement_mean=("disagreement_rate", "mean"),
        )
    )

    def ablation_label(row: pd.Series) -> str:
        block_size = int(row["block_size"])
        min_trees = int(row["min_trees"])
        if min_trees == 10:
            return f"Block size $B={block_size}$"
        if block_size == 10:
            return f"Minimum trees $t_{{\\min}}={min_trees}$"
        return f"$B={block_size},\\ t_{{\\min}}={min_trees}$"

    grouped["Setting"] = grouped.apply(ablation_label, axis=1)
    grouped = _dataset_sort(grouped).sort_values(["dataset", "min_trees", "block_size"])
    grouped["Accuracy"] = grouped["accuracy_mean"].map(lambda v: _fmt(v, 4))
    grouped["Avg. work"] = grouped["work_mean"].map(lambda v: _fmt(v, 2))
    grouped["Disagreement"] = grouped["disagreement_mean"].map(lambda v: _fmt(v, 4))
    return grouped[["dataset", "Setting", "Accuracy", "Avg. work", "Disagreement"]].rename(
        columns={"dataset": "Dataset"}
    )


def build_gbm_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    subset = df[df["method"].str.startswith("LazyGBM")].copy()
    grouped = (
        subset.groupby(["dataset", "variant", "min_trees"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            work_mean=("avg_work_units", "mean"),
            disagreement_mean=("disagreement_rate", "mean"),
        )
    )

    def variant_label(row: pd.Series) -> str:
        variant = row["variant"]
        min_trees = int(row["min_trees"])
        if variant == "current" and min_trees != 10:
            return f"Current, $t_{{\\min}}={min_trees}$"
        if variant == "current":
            return "Current prototype"
        if variant == "certificate_plus_flip":
            return "Certificate + flip score"
        if variant == "certificate_only":
            return "Certificate only"
        return f"{variant}, $t_{{\\min}}={min_trees}$"

    grouped["Setting"] = grouped.apply(variant_label, axis=1)
    grouped["variant_order"] = grouped["variant"].map(GBM_VARIANT_ORDER).fillna(99)
    grouped = _dataset_sort(grouped).sort_values(["dataset", "variant_order", "min_trees"])
    grouped["Accuracy"] = grouped["accuracy_mean"].map(lambda v: _fmt(v, 4))
    grouped["Avg. work"] = grouped["work_mean"].map(lambda v: _fmt(v, 2))
    grouped["Disagreement"] = grouped["disagreement_mean"].map(lambda v: _fmt(v, 4))
    return grouped[["dataset", "Setting", "Accuracy", "Avg. work", "Disagreement"]].rename(
        columns={"dataset": "Dataset"}
    )


def build_calibration_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    grouped = (
        df.groupby("dataset", as_index=False)
        .agg(
            flip_rate=("flip_rate", "mean"),
            mean_predicted_risk=("mean_predicted_risk", "mean"),
            spearman_rho=("spearman_rho", "mean"),
            ece=("ece", "mean"),
            monotone_seed_count=("monotone_bins", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
            run_count=("dataset", "count"),
        )
    )
    grouped = _dataset_sort(grouped)
    grouped["flip_rate"] = grouped["flip_rate"].map(lambda v: _fmt(v, 4))
    grouped["mean_predicted_risk"] = grouped["mean_predicted_risk"].map(lambda v: _fmt(v, 4))
    grouped["spearman_rho"] = grouped["spearman_rho"].map(lambda v: _fmt(v, 4))
    grouped["ece"] = grouped["ece"].map(lambda v: _fmt(v, 4))
    grouped["monotone_seeds"] = grouped.apply(
        lambda row: f"{int(row['monotone_seed_count'])}/{int(row['run_count'])}",
        axis=1,
    )
    return grouped[
        ["dataset", "flip_rate", "mean_predicted_risk", "spearman_rho", "ece", "monotone_seeds"]
    ].rename(
        columns={
            "dataset": "Dataset",
            "flip_rate": "Observed flip rate",
            "mean_predicted_risk": "Mean risk proxy",
            "spearman_rho": "Spearman $\\rho$",
            "ece": "ECE",
            "monotone_seeds": "Monotone seeds",
        }
    )


def main() -> None:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    main_df = pd.read_csv(args.main_csv)

    runtime_table = build_runtime_context(main_df)
    _write_table(
        runtime_table,
        tables_dir,
        "revision_runtime_context",
        "Mean over 30 seeds for lazy-method runtime context, full-model latency baseline, and stopping-time work quantiles.",
        "tab:revision_runtime_context",
    )

    rf_df = _load_glob(args.rf_ablation_glob)
    if not rf_df.empty:
        rf_table = build_rf_ablation_table(rf_df)
        _write_table(
            rf_table,
            tables_dir,
            "revision_lazyrf_ablation",
            "Mean over 30 seeds for LazyRF ablation results on Covertype and Higgs.",
            "tab:revision_lazyrf_ablation",
        )

    gbm_df = _load_glob(args.gbm_ablation_glob)
    if not gbm_df.empty:
        gbm_table = build_gbm_ablation_table(gbm_df)
        _write_table(
            gbm_table,
            tables_dir,
            "revision_lazygbm_ablation",
            "Mean over 30 seeds for LazyGBM ablation results on Covertype and Higgs.",
            "tab:revision_lazygbm_ablation",
        )

    calibration_df = _load_glob(args.calibration_glob)
    if not calibration_df.empty:
        calibration_table = build_calibration_table(calibration_df)
        _write_table(
            calibration_table,
            tables_dir,
            "revision_calibration_summary",
            "Mean over 30 seeds for cross-dataset LazyRF calibration summary.",
            "tab:revision_calibration_summary",
        )


if __name__ == "__main__":
    main()
