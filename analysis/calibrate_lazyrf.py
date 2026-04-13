#!/usr/bin/env python3
"""
LazyRF surrogate calibration / validation analysis.

This script empirically tests whether LazyRF's stopping-time stability score
correlates with (and roughly calibrates) disagreement with the full model.
The LazyRF implementation evaluates stop checks only at fixed block boundaries.

Outputs:
  - figures/lazyrf_calibration_<DATASET>.png  (reliability diagram for flip risk)
  - tables/lazyrf_calibration_<DATASET>.csv   (bin-level summary)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import run_experiments as rex
from baselines import run_baseline_full_rf
from lazy_evaluation import LazyRF


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Source Sans 3", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.5,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.axisbelow": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "in",
        "ytick.direction": "in",
    }
)


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)
    if edges.size < 3:
        # Fall back to uniform bins if the distribution is highly concentrated.
        edges = np.linspace(float(np.min(values)), float(np.max(values)), n_bins + 1)
        edges = np.unique(edges)
    return edges


def _reliability_table(
    predicted_risk: np.ndarray, flip: np.ndarray, n_bins: int
) -> Tuple[pd.DataFrame, float]:
    edges = _quantile_bins(predicted_risk, n_bins=n_bins)
    if edges.size < 2:
        raise ValueError("Unable to construct bins for reliability table.")

    bin_ids = np.digitize(predicted_risk, edges[1:-1], right=False)
    rows = []
    ece = 0.0
    n = len(predicted_risk)
    for bin_idx in range(edges.size - 1):
        mask = bin_ids == bin_idx
        count = int(np.sum(mask))
        if count == 0:
            continue
        risk_mean = float(np.mean(predicted_risk[mask]))
        flip_rate = float(np.mean(flip[mask]))
        weight = count / max(n, 1)
        ece += weight * abs(risk_mean - flip_rate)
        rows.append(
            {
                "bin_low": float(edges[bin_idx]),
                "bin_high": float(edges[bin_idx + 1]),
                "count": count,
                "predicted_risk_mean": risk_mean,
                "observed_flip_rate": flip_rate,
            }
        )
    return pd.DataFrame(rows), float(ece)


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), float("nan")
    rho, p_value = spearmanr(x, y)
    return float(rho), float(p_value)


def _build_output_dirs(args: argparse.Namespace) -> Tuple[Path, Path]:
    csv_dir = Path(args.output_dir) if args.output_dir else Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, figures_dir


def _single_seed_calibration(
    spec: rex.DatasetSpec,
    args: argparse.Namespace,
    seed: int,
    run_idx: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    seed_args = argparse.Namespace(**{**vars(args), "random_state": seed})

    X_train, X_test, y_train, y_test, _ = spec.loader(spec, seed_args)
    y_train, y_test, _ = rex._ensure_zero_based_labels(y_train, y_test)

    rf_full = run_baseline_full_rf(
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators=args.rf_trees,
        random_state=seed,
        n_jobs=-1,
    )

    rng = np.random.default_rng(seed)
    if args.calibration_samples and 0 < args.calibration_samples < len(X_test):
        calib_idx = rng.choice(len(X_test), size=args.calibration_samples, replace=False)
        X_cal = X_test[calib_idx]
    else:
        X_cal = X_test

    full_preds = rf_full.model.predict(X_cal)

    lazy_rf = LazyRF(
        rf_full.model,
        threshold=args.threshold,
        min_trees=args.min_trees,
        block_size=args.block_size,
        random_state=seed,
    )
    lazy_preds, avg_trees, details = lazy_rf.predict_lazy_with_details(X_cal)

    flip = (lazy_preds != full_preds).astype(np.float64)
    stop_scores = details["stop_scores"].astype(np.float64)
    predicted_risk = 1.0 - stop_scores

    rho, p_value = _safe_spearman(predicted_risk, flip)
    brier = float(np.mean((predicted_risk - flip) ** 2))
    mean_flip = float(np.mean(flip))
    mean_predicted_risk = float(np.mean(predicted_risk))

    table, ece = _reliability_table(predicted_risk, flip, n_bins=int(args.bins))
    table = table.assign(seed=seed, run=run_idx, dataset=spec.name)

    summary = {
        "dataset": spec.name,
        "seed": seed,
        "run": run_idx,
        "n_samples": int(len(X_cal)),
        "avg_trees": float(avg_trees),
        "threshold": float(args.threshold),
        "min_trees": int(args.min_trees),
        "block_size": int(args.block_size),
        "bins": int(args.bins),
        "flip_rate": mean_flip,
        "mean_predicted_risk": mean_predicted_risk,
        "spearman_rho": rho,
        "spearman_p_value": p_value,
        "brier_score": brier,
        "ece": ece,
        "monotone_bins": bool(table["observed_flip_rate"].is_monotonic_increasing),
    }
    return table, summary


def _aggregate_summary(per_seed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        per_seed_df.groupby("dataset", as_index=False)
        .agg(
            runs=("seed", "count"),
            n_samples=("n_samples", "mean"),
            avg_trees=("avg_trees", "mean"),
            threshold=("threshold", "mean"),
            min_trees=("min_trees", "mean"),
            block_size=("block_size", "mean"),
            bins=("bins", "mean"),
            flip_rate=("flip_rate", "mean"),
            mean_predicted_risk=("mean_predicted_risk", "mean"),
            spearman_rho=("spearman_rho", "mean"),
            spearman_p_value=("spearman_p_value", "mean"),
            brier_score=("brier_score", "mean"),
            ece=("ece", "mean"),
            monotone_seed_count=("monotone_bins", lambda s: int(pd.Series(s).fillna(False).astype(bool).sum())),
        )
    )
    grouped["bins"] = grouped["bins"].round().astype(int)
    grouped["min_trees"] = grouped["min_trees"].round().astype(int)
    grouped["block_size"] = grouped["block_size"].round().astype(int)
    return grouped


def _plot_reliability(table: pd.DataFrame, fig_path: Path) -> None:
    plt.figure(figsize=(4.0, 3.0))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="gray", label="Ideal")
    plt.plot(
        table["predicted_risk_mean"],
        table["observed_flip_rate"],
        marker="x",
        linewidth=1.5,
        label="LazyRF (binned)",
    )
    plt.xlabel("Predicted flip risk (1 - stability score)")
    plt.ylabel("Observed flip rate (vs full RF)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=600)
    plt.close()


def parse_args() -> argparse.Namespace:
    dataset_choices = sorted(rex.DATASET_REGISTRY.keys())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=str, default="covertype", choices=dataset_choices)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1, help="Number of independent seeds to evaluate.")
    parser.add_argument(
        "--start-seed",
        type=int,
        default=None,
        help="Base seed for multi-run calibration. Defaults to --random-state.",
    )

    # Dataset paths / sizing (match run_experiments defaults)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--higgs-path", type=str, default="data/higgs/HIGGS.csv")
    parser.add_argument("--credit-path", type=str, default="data/creditcard/creditcard.csv")
    parser.add_argument("--higgs-max-rows", type=int, default=200_000)
    parser.add_argument("--credit-max-rows", type=int, default=200_000)
    parser.add_argument("--covertype-max-rows", type=int, default=200_000)
    parser.add_argument("--mnist-max-rows", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)

    # Model / LazyRF configuration
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--min-trees", type=int, default=10)
    parser.add_argument(
        "--block-size",
        type=int,
        default=10,
        help="LazyRF block size. Stop checks occur only after each block is evaluated.",
    )

    # Calibration configuration
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=20_000,
        help="Subsample size from the test set for calibration (0 = use all).",
    )
    parser.add_argument("--bins", type=int, default=10)

    # Outputs
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory for CSV outputs. If omitted, --tables-dir is used for compatibility.",
    )
    parser.add_argument("--figures-dir", type=str, default="figures")
    parser.add_argument("--tables-dir", type=str, default="tables")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_key = args.dataset.lower()
    spec = rex.DATASET_REGISTRY[dataset_key]
    start_seed = args.start_seed if args.start_seed is not None else args.random_state
    csv_dir, figures_dir = _build_output_dirs(args)

    per_seed_tables: List[pd.DataFrame] = []
    per_seed_summaries: List[Dict[str, float]] = []
    representative_table: pd.DataFrame | None = None
    representative_summary: Dict[str, float] | None = None

    for run_idx in range(int(args.runs)):
        seed = start_seed + run_idx
        table, summary = _single_seed_calibration(spec, args, seed=seed, run_idx=run_idx)
        per_seed_tables.append(table)
        per_seed_summaries.append(summary)
        if representative_table is None:
            representative_table = table.drop(columns=["seed", "run", "dataset"]).copy()
            representative_summary = summary
        print(
            f"[{spec.name}] run={run_idx+1}/{args.runs} seed={seed} "
            f"flip_rate={summary['flip_rate']:.5f} ece={summary['ece']:.5f}"
        )

    if representative_table is None or representative_summary is None:
        raise RuntimeError("No calibration runs were executed.")

    per_seed_bins_df = pd.concat(per_seed_tables, ignore_index=True)
    per_seed_summary_df = pd.DataFrame(per_seed_summaries)
    aggregate_df = _aggregate_summary(per_seed_summary_df)

    dataset_name = spec.name.replace(" ", "_")
    legacy_bins_path = csv_dir / f"lazyrf_calibration_{dataset_name}.csv"
    bins_path = csv_dir / f"lazyrf_calibration_bins_{dataset_name}.csv"
    per_seed_summary_path = csv_dir / f"lazyrf_calibration_summary_runs_{dataset_name}.csv"
    aggregate_summary_path = csv_dir / f"lazyrf_calibration_summary_{dataset_name}.csv"
    fig_path = figures_dir / f"lazyrf_calibration_{dataset_name}.png"

    representative_table.to_csv(legacy_bins_path, index=False)
    per_seed_bins_df.to_csv(bins_path, index=False)
    per_seed_summary_df.to_csv(per_seed_summary_path, index=False)
    aggregate_df.to_csv(aggregate_summary_path, index=False)
    _plot_reliability(representative_table, fig_path)

    agg_row = aggregate_df.iloc[0]
    monotone_runs = f"{int(agg_row['monotone_seed_count'])}/{int(agg_row['runs'])}"
    print(
        f"[{spec.name}] aggregated over {int(agg_row['runs'])} seeds: "
        f"flip_rate={agg_row['flip_rate']:.5f}, "
        f"mean_pred_risk={agg_row['mean_predicted_risk']:.5f}, "
        f"spearman={agg_row['spearman_rho']:.4f}, "
        f"ece={agg_row['ece']:.5f}, monotone={monotone_runs}"
    )
    print(f"[{spec.name}] representative figure seed={start_seed}")
    print(f"[{spec.name}] wrote {fig_path}")
    print(f"[{spec.name}] wrote {legacy_bins_path}")
    print(f"[{spec.name}] wrote {bins_path}")
    print(f"[{spec.name}] wrote {per_seed_summary_path}")
    print(f"[{spec.name}] wrote {aggregate_summary_path}")


if __name__ == "__main__":
    main()
