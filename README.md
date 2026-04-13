# Instance-Adaptive Lazy Inference for Tree Ensembles

This repository contains inference-time wrappers and experiment scripts for adaptive early stopping in tree ensembles. The current codebase is centered on two model-preserving wrappers for fitted scikit-learn classifiers:

- `LazyRF` wraps `RandomForestClassifier` and stops at fixed block boundaries when a posterior stability score is high enough.
- `LazyGBM` wraps `GradientBoostingClassifier` and combines residual-capacity certificates with optional heuristic stopping rules.

The repo also includes comparison baselines, multi-seed experiment runners, plotting utilities, analysis scripts, and bundled CSV snapshots from prior runs.

## What Is In The Repo?

- `lazy_evaluation.py`: `LazyRF` and `LazyGBM`, plus per-sample stopping diagnostics.
- `baselines.py`: full RF, fixed cascade, two-stage cascade, QuickScorer, and BranchyNet baselines.
- `run_experiments.py`: single-run experiment harness.
- `run_multi_experiments.py`: repeated-run harness with aggregate statistics, plots, and optional LaTeX output.
- `table_figures.py`: publication-style tables and 2x2 figures from `raw_results_perf.csv` and `raw_results_sweep.csv`.
- `analysis/`: LazyRF calibration, sanity checks, and table-building utilities.
- `rerun_all_tables.sh`: end-to-end 30-seed regeneration pipeline for tables, figures, ablations, and calibration outputs.

This is a script-first research repo rather than an installable Python package. Run commands from the repository root.

## Installation

There is no `requirements.txt` or package metadata in the repo right now, so install dependencies manually.

```bash
git clone https://github.com/pz1004/lazy_inference.git
cd lazy_inference
```

Minimal dependencies for the lazy wrappers:

```bash
pip install numpy scipy scikit-learn
```

Full experiment and plotting stack:

```bash
pip install numpy scipy pandas scikit-learn matplotlib torch torchvision
# optional, used automatically when available
pip install numba
```

Notes:

- `torch` and `torchvision` are needed for the BranchyNet baseline and MNIST loading.
- `matplotlib` and `pandas` are needed for `run_multi_experiments.py`, `table_figures.py`, and the analysis scripts.
- `numba` is optional. The wrappers fall back to pure NumPy if it is not installed.

## Quick Start

```python
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from lazy_evaluation import LazyGBM, LazyRF

rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
lazy_rf = LazyRF(
    rf,
    threshold=0.95,
    min_trees=10,
    block_size=10,
    random_state=42,
)
y_pred_rf, avg_trees, rf_details = lazy_rf.predict_lazy_with_details(X_test)

gbm = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
lazy_gbm = LazyGBM(
    gbm,
    spr_threshold=3.0,
    min_trees=10,
    block_size=1,
    enable_ratio_heuristic=True,
    enable_flip_heuristic=True,
    enable_late_margin_fallback=True,
)
y_pred_gbm, avg_stages, gbm_details = lazy_gbm.predict_lazy_with_details(X_test)
```

Returned diagnostics:

- `LazyRF`: `trees_used`, `stop_scores`, `stop_reasons`, `class_posteriors`
- `LazyGBM`: `trees_used`, `stop_reasons`, `margins_at_stop`, `flip_scores`

Important detail: the `LazyGBM` class constructor currently defaults to `spr_threshold=4.6`, while `run_experiments.py` defaults to `--lazy-gbm-threshold 3.0`. Pass the threshold explicitly if you want script and direct-library behavior to match.

## Datasets

The experiment harness supports four datasets:

| CLI name | Dataset | Loader | Default location / behavior |
| --- | --- | --- | --- |
| `mnist` | MNIST | `torchvision.datasets.MNIST` | Auto-downloads into `data/mnist` |
| `covertype` | Covertype | `sklearn.datasets.fetch_covtype` | Cached under `--data-dir` |
| `higgs` | Higgs | Local CSV | Default `data/higgs/HIGGS.csv` or `HIGGS_PATH` |
| `credit` | Credit Card Fraud | Local CSV | Default `data/creditcard/creditcard.csv` or `CREDIT_CARD_PATH` |

Default row limits in the current scripts:

- `--covertype-max-rows 200000`
- `--higgs-max-rows 200000`
- `--credit-max-rows 200000`
- `--mnist-max-rows` defaults to no cap

If Higgs or Credit Card files are missing, `run_experiments.py` logs an error and skips that dataset instead of aborting the entire run.

## Methods

`run_experiments.py --methods ...` accepts the following identifiers:

| CLI identifier | Reported method name(s) | Notes |
| --- | --- | --- |
| `full_rf` | `Baseline A - Full RF` | Standard `RandomForestClassifier` baseline |
| `fixed_cascade` | `Baseline B - Fixed Cascade` | Static RF checkpoint policy |
| `cascade` | `Cascade RF (Two-Stage)` | Small RF first stage plus full RF fallback |
| `lazy_rf` | `LazyRF` or parameterized `LazyRF[...]` names | Threshold, block-size, and `min_trees` sweeps |
| `quickscorer` | `Baseline C - QuickScorer` | Bitmask-style tree traversal baseline |
| `branchynet` | `Baseline D - Full BranchyNet` and `Baseline D - Early Exit BranchyNet` | Simple 3-block MLP baseline |
| `full_gbm` | `Full GBM` | Standard `GradientBoostingClassifier` baseline |
| `lazy_gbm` | `LazyGBM` or parameterized `LazyGBM[...]` names | Threshold and variant sweeps |

## Running Experiments

Common single-run commands:

```bash
# all configured datasets and methods
python run_experiments.py

# only selected datasets and methods
python run_experiments.py \
  --datasets mnist covertype \
  --methods full_rf lazy_rf full_gbm lazy_gbm

# datasets backed by local CSVs
python run_experiments.py \
  --datasets higgs credit \
  --higgs-path /path/to/HIGGS.csv \
  --credit-path /path/to/creditcard.csv

# LazyRF threshold sweep
python run_experiments.py \
  --methods lazy_rf \
  --lazy-rf-thresholds 0.90 0.95 0.97 0.99 \
  --output-json results/lazy_rf_sweep.json

# LazyGBM threshold + variant sweep
python run_experiments.py \
  --methods full_gbm lazy_gbm \
  --lazy-gbm-variant certificate_only \
  --lazy-gbm-thresholds 2.0 3.0 4.0 \
  --output-json results/lazy_gbm_ablation.json
```

High-value CLI flags:

- `--rf-trees`, `--gbm-trees`
- `--fixed-checkpoints`, `--fixed-thresholds`
- `--cascade-stage1`, `--cascade-threshold`
- `--lazy-rf-threshold`, `--lazy-rf-min-trees`, `--lazy-rf-block-size`, `--lazy-rf-thresholds`
- `--lazy-gbm-threshold`, `--lazy-gbm-min-trees`, `--lazy-gbm-block-size`
- `--lazy-gbm-variant`, `--lazy-gbm-ratio-threshold`, `--lazy-gbm-flip-scale`, `--lazy-gbm-late-margin-fraction`
- `--max-train-samples`, `--max-test-samples`
- `--output-json`

Use `python run_experiments.py -h` for the full CLI surface.

The single-run script prints a per-dataset summary table to stdout and can optionally write a nested JSON report via `--output-json`.

## Multi-Run Experiments

`run_multi_experiments.py` extends the single-run harness with repeated seeds, aggregate statistics, plots, and optional LaTeX export:

```bash
python run_multi_experiments.py \
  --runs 30 \
  --datasets mnist covertype higgs credit \
  --result-dir results/multi \
  --no-display
```

Outputs under `--result-dir`:

- `raw_results.csv`
- `raw_results.json`
- `figure1_efficiency_curve.png`
- `lazy_rf_tradeoff_<dataset>.png`
- `lazy_gbm_tradeoff_<dataset>.png`
- optional LaTeX table via `--table-path`

Fresh `raw_results.csv` files produced by the current harness can include richer metadata than the bundled snapshot CSVs, including disagreement rate, AUROC/AUPRC on binary tasks, work quantiles, and stop-reason fractions when present.

## Bundled Result Snapshots

The repository currently ships two consolidated CSV exports:

- `raw_results_perf.csv`: main benchmark snapshot with 1080 rows (`4 datasets x 9 reported methods x 30 runs`)
- `raw_results_sweep.csv`: LazyRF and LazyGBM sweep snapshot with 600 rows

From the bundled `raw_results_perf.csv`, the default lazy methods have the following mean work reductions:

- `LazyRF`: Covertype `80.8%`, Credit Card `90.0%`, Higgs `60.4%`, MNIST `82.5%`
- `LazyGBM`: Covertype `19.0%`, Credit Card `68.3%`, Higgs `19.7%`, MNIST `19.0%`

See the CSVs for the full per-run accuracy, runtime, and threshold data.

## Plotting And Analysis

Useful repo scripts beyond the main experiment runners:

- `python table_figures.py`
  Reads `raw_results_perf.csv` and `raw_results_sweep.csv` from the current directory and writes LaTeX tables to `tables/` plus PDF/PNG figures to `figures/`.
- `python analysis/check_lazyrf_algorithm1.py`
  Runs semantic checks for block-boundary LazyRF behavior.
- `python analysis/calibrate_lazyrf.py --dataset covertype --runs 30 ...`
  Measures whether `1 - stop_score` tracks disagreement with the full RF; writes calibration CSVs and `lazyrf_calibration_<dataset>.png`.
- `python analysis/build_revision_tables.py --main-csv ...`
  Builds revision-specific summary tables from refreshed raw outputs and ablations.
- `bash rerun_all_tables.sh`
  Orchestrates the full 30-seed pipeline, ablations, calibration runs, and downstream table generation.

`rerun_all_tables.sh` writes to `results/full_tables_runs30_<timestamp>` and `logs/full_tables_runs30_<timestamp>` by default. You can override those roots with the `ROOT` and `LOG` environment variables.

## Caveats

- Work units are proxy metrics: RF trees, GBM stages, or BranchyNet depth. They are not guaranteed wall-clock speedups.
- `LazyRF` stop checks happen only at block boundaries.
- `LazyGBM` in this repo targets scikit-learn `GradientBoostingClassifier`; it is not a generic wrapper for every boosting library.
- `QuickScorer` here is a research baseline implementation with fallback tracking, not a production-optimized systems implementation.
- Experiment logging defaults to `logs/run_experiments.log` and can be overridden with `RUN_EXPERIMENTS_LOG` and `RUN_EXPERIMENTS_LOG_LEVEL`.
