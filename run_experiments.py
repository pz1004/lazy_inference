"""
Command-line runner for evaluating lazy tree ensembles and baselines.

This refactored version keeps the original experiment semantics but introduces
type hints, structured logging, and clearer data-loading utilities.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torchvision.datasets import MNIST

from baselines import (
    BaselineResult,
    run_baseline_full_rf,
    run_cascade_baseline,
    run_early_exit_baseline,
    run_fixed_cascade_baseline,
    run_quickscorer_baseline,
)
from lazy_evaluation import LazyGBM, LazyRF

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _configure_logger() -> logging.Logger:
    log_path = Path(os.environ.get("RUN_EXPERIMENTS_LOG", "logs/run_experiments.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("lazy_experiments")
    if logger.handlers:
        return logger
    level = os.environ.get("RUN_EXPERIMENTS_LOG_LEVEL", "INFO").upper()
    logger.setLevel(level)
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    try:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as err:  # pragma: no cover - fallback path
        logger.warning(
            "Unable to open log file %s (%s); continuing with console logging only",
            log_path,
            err,
        )
    return logger


LOGGER = _configure_logger()


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        LOGGER.debug("Torch not installed; skipping torch seed configuration.")


def _maybe_subsample(X: np.ndarray, y: np.ndarray, limit: Optional[int], rng: np.random.Generator):
    if limit is None or limit <= 0 or limit >= len(X):
        return X, y
    idx = rng.choice(len(X), size=limit, replace=False)
    return X[idx], y[idx]


def _normalize_tabular(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test)


# Available methods for --methods option
# Keys are user-facing names, values are internal identifiers
ALL_METHODS = [
    "full_rf",       # Baseline A - Full Random Forest
    "fixed_cascade", # Baseline B - Fixed Cascade
    "cascade",       # Cascade RF
    "lazy_rf",       # LazyRF (with optional threshold sweep)
    "quickscorer",   # QuickScorer
    "branchynet",    # BranchyNet (Full + Early Exit)
    "full_gbm",      # Full GBM
    "lazy_gbm",      # LazyGBM (with optional threshold sweep)
]


def _ensure_zero_based_labels(y_train: np.ndarray, y_test: np.ndarray):
    """Remap arbitrary labels to contiguous [0, K) integers."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    combined = np.concatenate([y_train, y_test])
    if combined.size == 0:
        return y_train, y_test, None
    unique = np.unique(combined)
    is_zero_based_int = (
        np.issubdtype(unique.dtype, np.integer) and unique.min() == 0 and np.array_equal(unique, np.arange(unique.size))
    )
    if is_zero_based_int:
        return y_train, y_test, None
    encoder = LabelEncoder()
    encoder.fit(unique)
    return encoder.transform(y_train), encoder.transform(y_test), encoder


@dataclass
class DatasetSpec:
    """Metadata and loader callback for a dataset."""

    name: str
    modality: str
    features: int
    instances: int
    difficulty: str
    task_type: str
    loader: Callable[
        ["DatasetSpec", argparse.Namespace],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]],
    ]


def load_mnist(spec: DatasetSpec, args: argparse.Namespace):
    data_dir = Path(args.data_dir) / "mnist"
    train_ds = MNIST(data_dir, train=True, download=True)
    test_ds = MNIST(data_dir, train=False, download=True)
    X_train = train_ds.data.view(len(train_ds), -1).numpy().astype(np.float32) / 255.0
    y_train = train_ds.targets.numpy()
    X_test = test_ds.data.view(len(test_ds), -1).numpy().astype(np.float32) / 255.0
    y_test = test_ds.targets.numpy()
    total_rows = len(X_train) + len(X_test)
    loaded_rows = total_rows
    max_rows = getattr(args, "mnist_max_rows", None)
    if max_rows is not None and max_rows > 0 and max_rows < total_rows:
        train_float = max_rows * (len(X_train) / total_rows)
        train_limit = int(train_float)
        test_float = max_rows - train_float
        test_limit = int(test_float)
        remainder = max_rows - (train_limit + test_limit)
        split_order = [
            ("train", train_float - train_limit, len(X_train) - train_limit),
            ("test", test_float - test_limit, len(X_test) - test_limit),
        ]
        split_order.sort(key=lambda entry: entry[1], reverse=True)
        for name, _, capacity in split_order:
            if remainder <= 0:
                break
            if capacity <= 0:
                continue
            take = min(remainder, capacity)
            if name == "train":
                train_limit += take
            else:
                test_limit += take
            remainder -= take
        train_limit = min(train_limit, len(X_train))
        test_limit = min(test_limit, len(X_test))
        if train_limit == 0 and len(X_train) > 0 and max_rows > 0:
            train_limit = min(1, len(X_train))
        if test_limit == 0 and len(X_test) > 0 and max_rows > 1 and train_limit > 0:
            train_limit -= 1
            test_limit = 1
        loaded_rows = train_limit + test_limit
        LOGGER.info(
            "Subsampling MNIST from %d to %d rows (train=%d, test=%d)",
            total_rows,
            loaded_rows,
            train_limit,
            test_limit,
        )
        X_train = X_train[:train_limit]
        y_train = y_train[:train_limit]
        X_test = X_test[:test_limit]
        y_test = y_test[:test_limit]
    rng = np.random.default_rng(args.random_state)
    X_train, y_train = _maybe_subsample(X_train, y_train, args.max_train_samples, rng)
    X_test, y_test = _maybe_subsample(X_test, y_test, args.max_test_samples, rng)
    LOGGER.info("Loaded MNIST rows=%d train=%d test=%d", loaded_rows, len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, {"loaded_rows": loaded_rows}


def load_covertype(spec: DatasetSpec, args: argparse.Namespace):
    X, y = fetch_covtype(data_home=args.data_dir, return_X_y=True)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    max_rows = getattr(args, "covertype_max_rows", None)
    if max_rows is not None and max_rows < len(X):
        LOGGER.info("Subsampling Covertype from %d to %d rows", len(X), max_rows)
        rng = np.random.default_rng(args.random_state)
        idx = rng.choice(len(X), size=max_rows, replace=False)
        X, y = X[idx], y[idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )
    X_train, X_test = _normalize_tabular(X_train, X_test)
    rng = np.random.default_rng(args.random_state)
    X_train, y_train = _maybe_subsample(X_train, y_train, args.max_train_samples, rng)
    X_test, y_test = _maybe_subsample(X_test, y_test, args.max_test_samples, rng)
    LOGGER.info("Loaded Covertype train=%d test=%d", len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, {}


def _load_csv(path: Path, max_rows: Optional[int]) -> pd.DataFrame:
    return pd.read_csv(path, nrows=max_rows)


def load_higgs(spec: DatasetSpec, args: argparse.Namespace):
    raw_path = args.higgs_path or os.environ.get("HIGGS_PATH")
    if raw_path is None:
        raise FileNotFoundError("Set --higgs-path or HIGGS_PATH to point to the Higgs CSV.")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Higgs dataset not found at {path}")
    column_names = ["label"] + [f"f{i}" for i in range(spec.features)]
    df = _load_csv(path, args.higgs_max_rows)
    if df.shape[1] == spec.features + 1:
        df.columns = column_names
    y = df["label"].astype(np.int64).values
    X = df.drop(columns=["label"]).values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )
    X_train, X_test = _normalize_tabular(X_train, X_test)
    rng = np.random.default_rng(args.random_state)
    X_train, y_train = _maybe_subsample(X_train, y_train, args.max_train_samples, rng)
    X_test, y_test = _maybe_subsample(X_test, y_test, args.max_test_samples, rng)
    LOGGER.info("Loaded Higgs rows=%d train=%d test=%d", len(df), len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, {"loaded_rows": len(df)}


def load_credit(spec: DatasetSpec, args: argparse.Namespace):
    raw_path = args.credit_path or os.environ.get("CREDIT_CARD_PATH")
    if raw_path is None:
        raise FileNotFoundError("Set --credit-path or CREDIT_CARD_PATH for the credit card CSV.")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Credit card dataset not found at {path}")
    df = _load_csv(path, args.credit_max_rows)
    target_col = "Class" if "Class" in df.columns else df.columns[-1]
    y = df[target_col].astype(np.int64).values
    X = df.drop(columns=[target_col]).values.astype(np.float32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )
    X_train, X_test = _normalize_tabular(X_train, X_test)
    rng = np.random.default_rng(args.random_state)
    X_train, y_train = _maybe_subsample(X_train, y_train, args.max_train_samples, rng)
    X_test, y_test = _maybe_subsample(X_test, y_test, args.max_test_samples, rng)
    LOGGER.info("Loaded Credit rows=%d train=%d test=%d", len(df), len(X_train), len(X_test))
    return X_train, X_test, y_train, y_test, {"loaded_rows": len(df)}


DATASET_REGISTRY: Dict[str, DatasetSpec] = {
    "mnist": DatasetSpec(
        name="MNIST",
        modality="Image",
        features=784,
        instances=70000,
        difficulty="Low (Easy digits exit early)",
        task_type="multiclass",
        loader=load_mnist,
    ),
    "covertype": DatasetSpec(
        name="Covertype",
        modality="Tabular",
        features=54,
        instances=581000,
        difficulty="Medium (Complex boundaries)",
        task_type="multiclass",
        loader=load_covertype,
    ),
    "higgs": DatasetSpec(
        name="Higgs",
        modality="Physics",
        features=28,
        instances=11_000_000,
        difficulty="High (Noise, requires depth)",
        task_type="binary",
        loader=load_higgs,
    ),
    "credit": DatasetSpec(
        name="Credit Card",
        modality="Anomaly",
        features=30,
        instances=284000,
        difficulty="Variable (Fraud is hard, Normal is easy)",
        task_type="binary",
        loader=load_credit,
    ),
}

DATASET_REQUIREMENTS = {
    "higgs": ("higgs_path", "HIGGS_PATH", "Set --higgs-path or HIGGS_PATH to the Higgs CSV"),
    "credit": ("credit_path", "CREDIT_CARD_PATH", "Set --credit-path or CREDIT_CARD_PATH to the credit-card CSV"),
}


def ensure_dataset_prereqs(dataset_key: str, args: argparse.Namespace) -> None:
    """Guard against running datasets without required files."""
    if dataset_key not in DATASET_REQUIREMENTS:
        return
    arg_name, env_name, message = DATASET_REQUIREMENTS[dataset_key]
    if getattr(args, arg_name, None) or os.environ.get(env_name):
        return
    raise FileNotFoundError(message)


def energy_proxy(result: BaselineResult) -> float:
    """Implementation-agnostic work estimate."""
    return max(result.avg_work_units, 1e-9)


def _work_quantile_summary(work_units: np.ndarray) -> Dict[str, float]:
    quantiles = {
        "work_p50": 0.50,
        "work_p90": 0.90,
        "work_p95": 0.95,
        "work_p99": 0.99,
    }
    return {
        name: float(np.quantile(work_units, quantile))
        for name, quantile in quantiles.items()
    }


def _stop_reason_summary(stop_reasons: np.ndarray, labels: Dict[int, str]) -> Dict[str, float]:
    n = max(len(stop_reasons), 1)
    summary: Dict[str, float] = {}
    for code, label in labels.items():
        summary[f"stop_{label}_fraction"] = float(np.sum(stop_reasons == code) / n)
    return summary


def _lazy_gbm_variant_config(args: argparse.Namespace) -> Dict[str, Any]:
    variant = getattr(args, "lazy_gbm_variant", "current")
    base = {
        "block_size": args.lazy_gbm_block_size,
        "ratio_threshold": args.lazy_gbm_ratio_threshold,
        "multiclass_flip_scale": args.lazy_gbm_flip_scale,
        "late_margin_fraction": args.lazy_gbm_late_margin_fraction,
    }
    if variant == "current":
        return {
            **base,
            "variant": variant,
            "enable_ratio_heuristic": True,
            "enable_flip_heuristic": True,
            "enable_late_margin_fallback": True,
        }
    if variant == "certificate_only":
        return {
            **base,
            "variant": variant,
            "enable_ratio_heuristic": False,
            "enable_flip_heuristic": False,
            "enable_late_margin_fallback": False,
        }
    if variant == "certificate_plus_flip":
        return {
            **base,
            "variant": variant,
            "enable_ratio_heuristic": False,
            "enable_flip_heuristic": True,
            "enable_late_margin_fallback": False,
        }
    raise ValueError(f"Unsupported LazyGBM variant: {variant}")


def summarize_metrics(reference: BaselineResult, candidate: BaselineResult) -> Dict[str, float]:
    speedup = reference.inference_time / candidate.inference_time if candidate.inference_time > 0 else float("inf")
    accuracy_drop = reference.accuracy - candidate.accuracy
    worst_case_latency = reference.inference_time
    disagreement_rate = float(np.mean(reference.predictions != candidate.predictions))
    base_energy = energy_proxy(reference)
    candidate_energy = energy_proxy(candidate)
    energy_reduction = 1.0 - (candidate_energy / base_energy) if base_energy > 0 else 0.0
    return {
        "speedup": speedup,
        "accuracy_drop": accuracy_drop,
        "worst_case_latency": worst_case_latency,
        "disagreement_rate": disagreement_rate,
        "energy_reduction": energy_reduction,
        "work_reduction": energy_reduction,
    }


def _binary_score_metrics(y_test: Optional[np.ndarray], result: BaselineResult) -> Dict[str, float]:
    if y_test is None or result.scores is None:
        return {}
    y = np.asarray(y_test)
    scores = np.asarray(result.scores, dtype=np.float64)
    if y.shape[0] != scores.shape[0] or np.unique(y).size != 2:
        return {}
    finite = np.isfinite(scores)
    if not np.all(finite):
        y = y[finite]
        scores = scores[finite]
    if y.size == 0 or np.unique(y).size != 2:
        return {}
    try:
        return {
            "auroc": float(roc_auc_score(y, scores)),
            "auprc": float(average_precision_score(y, scores)),
        }
    except ValueError:
        return {}


def baseline_summary(
    result: BaselineResult,
    metrics: Dict[str, float],
    y_test: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    metrics = {**metrics, **_binary_score_metrics(y_test, result)}
    return {
        "name": result.name,
        "accuracy": result.accuracy,
        "inference_time": result.inference_time,
        "avg_work_units": result.avg_work_units,
        "metadata": result.metadata,
        "metrics": metrics,
    }


def train_full_gbm(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, n_estimators: int, random_state: int
) -> BaselineResult:
    gbm = GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state)
    train_start = time.time()
    gbm.fit(X_train, y_train)
    train_time = time.time() - train_start
    _ = gbm.predict(X_test[: min(100, len(X_test))])
    inf_start = time.time()
    preds = gbm.predict(X_test)
    inf_time = time.time() - inf_start
    acc = accuracy_score(y_test, preds)
    scores = gbm.predict_proba(X_test)[:, 1] if len(gbm.classes_) == 2 else None
    return BaselineResult(
        name="Full GBM",
        accuracy=acc,
        inference_time=inf_time,
        avg_work_units=n_estimators,
        predictions=preds,
        model=gbm,
        metadata={"train_time": train_time},
        scores=scores,
    )


def evaluate_lazy_rf(
    rf_result: BaselineResult,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    min_trees: int,
    block_size: int,
    random_state: Optional[int] = None,
    name: str = "LazyRF",
) -> BaselineResult:
    """Evaluate LazyRF on a fixed block-boundary checkpoint grid."""
    lazy_rf = LazyRF(
        rf_result.model,
        threshold=threshold,
        min_trees=min_trees,
        block_size=block_size,
        random_state=random_state,
    )
    # Warm-up to avoid one-off timing overhead
    _ = lazy_rf.predict_lazy(X_test[: min(100, len(X_test))])
    start_time = time.time()
    preds, avg_trees, details = lazy_rf.predict_lazy_with_details(X_test)
    inf_time = time.time() - start_time
    acc = accuracy_score(y_test, preds)
    scores = details["class_posteriors"][:, 1] if details["class_posteriors"].shape[1] == 2 else None
    metadata = {
        "threshold": threshold,
        "min_trees": min_trees,
        "block_size": block_size,
        "mean_stop_score": float(np.mean(details["stop_scores"])),
        "median_stop_score": float(np.median(details["stop_scores"])),
        **_work_quantile_summary(details["trees_used"]),
        **_stop_reason_summary(details["stop_reasons"], LazyRF.STOP_REASON_LABELS),
    }
    return BaselineResult(
        name=name,
        accuracy=acc,
        inference_time=inf_time,
        avg_work_units=avg_trees,
        predictions=preds,
        model=lazy_rf,
        metadata=metadata,
        scores=scores,
    )


def evaluate_lazy_gbm(
    gbm_result: BaselineResult,
    X_test: np.ndarray,
    y_test: np.ndarray,
    spr_threshold: float,
    min_trees: int,
    variant_config: Dict[str, Any],
    name: str = "LazyGBM",
) -> BaselineResult:
    """Evaluate LazyGBM at a given stability/flip-score threshold."""
    lazy_gbm = LazyGBM(
        gbm_result.model,
        spr_threshold=spr_threshold,
        min_trees=min_trees,
        block_size=variant_config["block_size"],
        enable_ratio_heuristic=variant_config["enable_ratio_heuristic"],
        ratio_threshold=variant_config["ratio_threshold"],
        enable_flip_heuristic=variant_config["enable_flip_heuristic"],
        multiclass_flip_scale=variant_config["multiclass_flip_scale"],
        enable_late_margin_fallback=variant_config["enable_late_margin_fallback"],
        late_margin_fraction=variant_config["late_margin_fraction"],
    )
    # Warm-up
    _ = lazy_gbm.predict_lazy(X_test[: min(100, len(X_test))])
    start_time = time.time()
    preds, avg_trees, details = lazy_gbm.predict_lazy_with_details(X_test)
    inf_time = time.time() - start_time
    acc = accuracy_score(y_test, preds)
    margins = details["margins_at_stop"]
    scores = None
    if len(gbm_result.model.classes_) == 2:
        clipped_margins = np.clip(margins, -709.0, 709.0)
        scores = 1.0 / (1.0 + np.exp(-clipped_margins))
    finite_flip_scores = details["flip_scores"][np.isfinite(details["flip_scores"])]
    metadata = {
        "spr_threshold": spr_threshold,
        "min_trees": min_trees,
        "block_size": variant_config["block_size"],
        "variant": variant_config["variant"],
        "enable_ratio_heuristic": variant_config["enable_ratio_heuristic"],
        "ratio_threshold": variant_config["ratio_threshold"],
        "enable_flip_heuristic": variant_config["enable_flip_heuristic"],
        "multiclass_flip_scale": variant_config["multiclass_flip_scale"],
        "enable_late_margin_fallback": variant_config["enable_late_margin_fallback"],
        "late_margin_fraction": variant_config["late_margin_fraction"],
        "mean_abs_margin_at_stop": float(np.mean(np.abs(margins))),
        "median_abs_margin_at_stop": float(np.median(np.abs(margins))),
        "mean_flip_score": (
            float(np.mean(finite_flip_scores)) if finite_flip_scores.size else float("nan")
        ),
        **_work_quantile_summary(details["trees_used"]),
        **_stop_reason_summary(details["stop_reasons"], LazyGBM.STOP_REASON_LABELS),
    }
    return BaselineResult(
        name=name,
        accuracy=acc,
        inference_time=inf_time,
        avg_work_units=avg_trees,
        predictions=preds,
        model=lazy_gbm,
        metadata=metadata,
        scores=scores,
    )


def _build_fixed_cascade_config(args: argparse.Namespace) -> Tuple[List[int], List[float]]:
    if args.fixed_checkpoints:
        checkpoints = sorted({c for c in args.fixed_checkpoints if 0 < c <= args.rf_trees})
    else:
        checkpoints = sorted({min(10, args.rf_trees), min(50, args.rf_trees), args.rf_trees})
    if not checkpoints:
        checkpoints = [args.rf_trees]
    elif checkpoints[-1] != args.rf_trees:
        checkpoints.append(args.rf_trees)
    if args.fixed_thresholds:
        thresholds = list(args.fixed_thresholds)
        if not thresholds:
            thresholds = [0.9] * (len(checkpoints) - 1) + [0.0]
        if len(thresholds) < len(checkpoints):
            thresholds.extend([thresholds[-1]] * (len(checkpoints) - len(thresholds)))
        elif len(thresholds) > len(checkpoints):
            LOGGER.warning("Truncating thresholds (%d) to match checkpoints (%d).", len(thresholds), len(checkpoints))
            thresholds = thresholds[: len(checkpoints)]
    else:
        thresholds = [0.9] * (len(checkpoints) - 1) + [0.0]
    return checkpoints, thresholds


def run_single_dataset(spec: DatasetSpec, args: argparse.Namespace) -> Dict[str, Any]:
    _set_random_seed(args.random_state)
    X_train, X_test, y_train, y_test, loader_meta = spec.loader(spec, args)
    y_train, y_test, label_encoder = _ensure_zero_based_labels(y_train, y_test)
    if label_encoder is not None:
        loader_meta = {**loader_meta, "original_classes": label_encoder.classes_.tolist()}
        LOGGER.info(
            "Re-encoded labels to zero-based indices for %s (classes=%s)",
            spec.name,
            ", ".join(map(str, label_encoder.classes_)),
        )
    dataset_meta = {
        "type": spec.modality,
        "features": spec.features,
        "instances": spec.instances,
        "difficulty_proxy": spec.difficulty,
        **loader_meta,
    }
    results: List[Dict[str, Any]] = []
    LOGGER.info(
        "Dataset %s | modality=%s features=%d instances=%d",
        spec.name,
        spec.modality,
        spec.features,
        spec.instances,
    )

    # Determine which methods to run
    methods_to_run = set(getattr(args, "methods", None) or ALL_METHODS)

    # Helper to check if a method should run
    def should_run(method: str) -> bool:
        return method in methods_to_run

    # Track created objects for cleanup
    rf_full = None
    fixed_baseline = None
    cascade = None
    lazy_rf_result = None
    quickscorer = None
    ee_results = None
    full_nn = None
    lazy_nn = None
    gbm_full = None
    lazy_gbm_result = None

    # RF-based methods need rf_full as baseline
    needs_rf = any(m in methods_to_run for m in ["full_rf", "fixed_cascade", "cascade", "lazy_rf", "quickscorer"])

    if needs_rf:
        rf_full = run_baseline_full_rf(
            X_train, y_train, X_test, y_test, n_estimators=args.rf_trees, random_state=args.random_state
        )
        if should_run("full_rf"):
            results.append(
                baseline_summary(
                    rf_full,
                    {
                        "speedup": 1.0,
                        "accuracy_drop": 0.0,
                        "worst_case_latency": rf_full.inference_time,
                        "energy_reduction": 0.0,
                        "work_reduction": 0.0,
                    },
                    y_test=y_test,
                )
            )
            LOGGER.info("[%s] Baseline A acc=%.4f time=%.4fs", spec.name, rf_full.accuracy, rf_full.inference_time)

        if should_run("fixed_cascade"):
            checkpoints, thresholds = _build_fixed_cascade_config(args)
            fixed_baseline = run_fixed_cascade_baseline(rf_full.model, X_test, y_test, checkpoints, thresholds)
            results.append(baseline_summary(fixed_baseline, summarize_metrics(rf_full, fixed_baseline), y_test=y_test))
            LOGGER.info(
                "[%s] Baseline B acc=%.4f avg_trees=%.1f checkpoints=%s",
                spec.name,
                fixed_baseline.accuracy,
                fixed_baseline.avg_work_units,
                checkpoints,
            )

        if should_run("cascade"):
            cascade = run_cascade_baseline(
                X_train,
                y_train,
                X_test,
                y_test,
                stage2_model=rf_full.model,
                stage1_trees=args.cascade_stage1,
                threshold=args.cascade_threshold,
                random_state=args.random_state,
            )
            results.append(baseline_summary(cascade, summarize_metrics(rf_full, cascade), y_test=y_test))
            LOGGER.info(
                "[%s] Cascade RF acc=%.4f hard_fraction=%.2f",
                spec.name,
                cascade.accuracy,
                cascade.metadata.get("hard_fraction", 0.0),
            )

        # --- LazyRF: single operating point or threshold sweep ---
        if should_run("lazy_rf"):
            if getattr(args, "lazy_rf_thresholds", None):
                rf_thresholds = sorted({float(t) for t in args.lazy_rf_thresholds})
            else:
                rf_thresholds = [float(args.lazy_rf_threshold)]

            for thr in rf_thresholds:
                is_default_lazy_rf = (
                    len(rf_thresholds) == 1
                    and args.lazy_rf_block_size == 10
                    and args.lazy_rf_min_trees == 10
                )
                if is_default_lazy_rf:
                    name = "LazyRF"
                else:
                    name = (
                        f"LazyRF[t={thr:.3f},B={args.lazy_rf_block_size},m={args.lazy_rf_min_trees}]"
                    )
                lazy_rf_result = evaluate_lazy_rf(
                    rf_full,
                    X_test,
                    y_test,
                    threshold=thr,
                    min_trees=args.lazy_rf_min_trees,
                    block_size=args.lazy_rf_block_size,
                    random_state=args.random_state,
                    name=name,
                )
                results.append(baseline_summary(lazy_rf_result, summarize_metrics(rf_full, lazy_rf_result), y_test=y_test))
                LOGGER.info(
                    "[%s] %s acc=%.4f avg_trees=%.1f threshold=%.3f",
                    spec.name,
                    name,
                    lazy_rf_result.accuracy,
                    lazy_rf_result.avg_work_units,
                    thr,
                )

        # --- QuickScorer ---
        if should_run("quickscorer"):
            quickscorer = run_quickscorer_baseline(rf_full.model, X_test, y_test)
            results.append(baseline_summary(quickscorer, summarize_metrics(rf_full, quickscorer), y_test=y_test))
            LOGGER.info(
                "[%s] QuickScorer acc=%.4f time=%.4fs fallback=%.4f",
                spec.name,
                quickscorer.accuracy,
                quickscorer.inference_time,
                quickscorer.metadata.get("fallback_rate", 0.0),
            )

    # --- BranchyNet ---
    if should_run("branchynet"):
        num_classes = int(len(np.unique(y_train)))
        LOGGER.info(
            "[DEBUG] BranchyNet setup for %s: X_train=%s, num_classes=%d, unique_labels=%s",
            spec.name, X_train.shape, num_classes, np.unique(y_train)
        )
        ee_results = run_early_exit_baseline(
            X_train,
            y_train,
            X_test,
            y_test,
            num_classes=num_classes,
            thresholds=(args.early_exit_threshold1, args.early_exit_threshold2),
            epochs=args.early_exit_epochs,
        )
        full_nn = ee_results["full"]
        lazy_nn = ee_results["early_exit"]
        results.append(
            baseline_summary(
                full_nn,
                {
                    "speedup": 1.0,
                    "accuracy_drop": 0.0,
                    "worst_case_latency": full_nn.inference_time,
                    "energy_reduction": 0.0,
                    "work_reduction": 0.0,
                },
                y_test=y_test,
            )
        )
        results.append(baseline_summary(lazy_nn, summarize_metrics(full_nn, lazy_nn), y_test=y_test))
        LOGGER.info(
            "[%s] BranchyNet lazy acc=%.4f avg_depth=%.2f",
            spec.name,
            lazy_nn.accuracy,
            lazy_nn.avg_work_units,
        )

    # GBM-based methods
    needs_gbm = any(m in methods_to_run for m in ["full_gbm", "lazy_gbm"])

    if needs_gbm:
        gbm_full = train_full_gbm(
            X_train, y_train, X_test, y_test, n_estimators=args.gbm_trees, random_state=args.random_state
        )
        if should_run("full_gbm"):
            results.append(
                baseline_summary(
                    gbm_full,
                    {
                        "speedup": 1.0,
                        "accuracy_drop": 0.0,
                        "worst_case_latency": gbm_full.inference_time,
                        "energy_reduction": 0.0,
                        "work_reduction": 0.0,
                    },
                    y_test=y_test,
                )
            )
            LOGGER.info("[%s] Full GBM acc=%.4f time=%.4fs", spec.name, gbm_full.accuracy, gbm_full.inference_time)

        # --- LazyGBM: single operating point or stability-threshold sweep ---
        if should_run("lazy_gbm"):
            if getattr(args, "lazy_gbm_thresholds", None):
                gbm_thresholds = sorted({float(t) for t in args.lazy_gbm_thresholds})
            else:
                gbm_thresholds = [float(args.lazy_gbm_threshold)]
            variant_config = _lazy_gbm_variant_config(args)

            for spr_thr in gbm_thresholds:
                try:
                    if len(gbm_thresholds) == 1 and variant_config["variant"] == "current":
                        name = "LazyGBM"
                    else:
                        name = f"LazyGBM[{variant_config['variant']},s={spr_thr:.2f}]"
                    lazy_gbm_result = evaluate_lazy_gbm(
                        gbm_full,
                        X_test,
                        y_test,
                        spr_threshold=spr_thr,
                        min_trees=args.lazy_gbm_min_trees,
                        variant_config=variant_config,
                        name=name,
                    )
                    results.append(baseline_summary(lazy_gbm_result, summarize_metrics(gbm_full, lazy_gbm_result), y_test=y_test))
                    LOGGER.info(
                        "[%s] %s acc=%.4f avg_trees=%.1f stability_threshold=%.2f",
                        spec.name,
                        name,
                        lazy_gbm_result.accuracy,
                        lazy_gbm_result.avg_work_units,
                        spr_thr,
                    )
                except NotImplementedError as err:
                    results.append({"name": "LazyGBM", "error": str(err)})
                    LOGGER.warning("[%s] LazyGBM(s=%.2f) skipped: %s", spec.name, spr_thr, err)

    # Clean up large objects to prevent memory leaks across runs
    del X_train, X_test, y_train, y_test
    for obj in [rf_full, fixed_baseline, cascade, lazy_rf_result, quickscorer,
                ee_results, full_nn, lazy_nn, gbm_full, lazy_gbm_result]:
        if obj is not None:
            del obj
    gc.collect()
    # Clear PyTorch CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return {"metadata": dataset_meta, "results": results}


def format_table(dataset: str, report: Dict[str, Any]) -> str:
    header = f"\n=== {dataset} ({report['metadata']['difficulty_proxy']}) ==="
    lines = [header]
    lines.append(
        f"Type: {report['metadata']['type']} | Features: {report['metadata']['features']} | Instances: {report['metadata']['instances']}"
    )
    columns = ["Method", "Acc", "Speedup", "Acc Drop", "Disagree", "Worst-Case (s)", "WorkRed"]
    rows: List[List[str]] = []
    for entry in report["results"]:
        if "metrics" not in entry:
            rows.append([entry.get("name", "N/A"), "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        metrics = entry["metrics"]
        rows.append(
            [
                entry["name"],
                f"{entry['accuracy']:.4f}",
                f"{metrics['speedup']:.2f}",
                f"{metrics['accuracy_drop']:.4f}",
                f"{metrics.get('disagreement_rate', float('nan')):.4f}",
                f"{metrics['worst_case_latency']:.4f}",
                f"{metrics.get('work_reduction', metrics.get('energy_reduction', 0.0))*100:.1f}%",
            ]
        )
    col_widths = [max(len(col), max((len(row[i]) for row in rows), default=0)) for i, col in enumerate(columns)]
    header_row = " | ".join(col.ljust(col_widths[i]) for i, col in enumerate(columns))
    lines.append(header_row)
    lines.append("-" * len(header_row))
    for row in rows:
        lines.append(" | ".join(row[i].ljust(col_widths[i]) for i in range(len(columns))))
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lazy evaluation experiment runner")
    parser.add_argument("--datasets", nargs="*", default=list(DATASET_REGISTRY.keys()), help="Datasets to run")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        choices=ALL_METHODS,
        help=(
            f"Methods to run. If not specified, all methods are run. "
            f"Available: {', '.join(ALL_METHODS)}"
        ),
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset cache directory")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples per dataset")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit evaluation samples per dataset")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rf-trees", type=int, default=100)
    parser.add_argument("--fixed-checkpoints", type=int, nargs="*", default=None, help="Checkpoint sizes for Baseline B")
    parser.add_argument("--fixed-thresholds", type=float, nargs="*", default=None, help="Confidence thresholds for Baseline B")
    parser.add_argument("--cascade-stage1", type=int, default=10)
    parser.add_argument("--cascade-threshold", type=float, default=0.9)
    # LazyRF: single operating point and optional threshold sweep
    parser.add_argument("--lazy-rf-threshold", type=float, default=0.95,
                        help="Default LazyRF posterior threshold when no sweep is requested.")
    parser.add_argument("--lazy-rf-min-trees", type=int, default=10,
                        help="Minimum number of evaluated trees before a block-boundary LazyRF stop check can trigger.")
    parser.add_argument("--lazy-rf-block-size", type=int, default=10,
                        help="Number of trees evaluated per LazyRF block; stop checks occur only at block boundaries.")
    parser.add_argument(
        "--lazy-rf-thresholds",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional list of LazyRF thresholds to sweep (e.g., 0.90 0.95 0.97 0.99). "
            "If provided, LazyRF is evaluated once per threshold and the single "
            "--lazy-rf-threshold value is ignored."
        ),
    )
    parser.add_argument("--early-exit-threshold1", type=float, default=0.5)
    parser.add_argument("--early-exit-threshold2", type=float, default=0.3)
    parser.add_argument("--early-exit-epochs", type=int, default=5)
    parser.add_argument("--gbm-trees", type=int, default=100)
    # LazyGBM: single operating point and optional stability-threshold sweep
    parser.add_argument("--lazy-gbm-threshold", type=float, default=3.0,
                        help="Default LazyGBM stability/flip-score threshold.")
    parser.add_argument("--lazy-gbm-min-trees", type=int, default=10,
                        help="Minimum number of boosting stages before LazyGBM can stop.")
    parser.add_argument("--lazy-gbm-block-size", type=int, default=1,
                        help="Number of boosting stages evaluated between LazyGBM stop checks.")
    parser.add_argument(
        "--lazy-gbm-variant",
        type=str,
        default="current",
        choices=["current", "certificate_only", "certificate_plus_flip"],
        help=(
            "Prototype stopping variant for LazyGBM. "
            "'current' matches the implementation used in the manuscript; "
            "'certificate_only' disables all heuristics beyond the residual-capacity certificate; "
            "'certificate_plus_flip' keeps only the flip-score heuristic in addition to the certificate."
        ),
    )
    parser.add_argument("--lazy-gbm-ratio-threshold", type=float, default=1.5,
                        help="Margin-to-bound ratio used by the multiclass ratio heuristic.")
    parser.add_argument("--lazy-gbm-flip-scale", type=float, default=2.0,
                        help="Multiplier applied to the flip-score threshold in multiclass LazyGBM.")
    parser.add_argument("--lazy-gbm-late-margin-fraction", type=float, default=0.8,
                        help="Fraction of total stages after which the late positive-margin fallback is enabled.")
    parser.add_argument(
        "--lazy-gbm-thresholds",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional list of LazyGBM stability/flip-score thresholds to sweep. "
            "If provided, LazyGBM is evaluated once per threshold and the single "
            "--lazy-gbm-threshold value is ignored."
        ),
    )
    parser.add_argument("--higgs-path", type=str, default="data/higgs/HIGGS.csv")
    parser.add_argument("--credit-path", type=str, default="data/creditcard/creditcard.csv")
    parser.add_argument("--higgs-max-rows", type=int, default=200_000)
    parser.add_argument("--credit-max-rows", type=int, default=200_000)
    parser.add_argument("--covertype-max-rows", type=int, default=200_000)
    parser.add_argument("--mnist-max-rows", type=int, default=None, help="Max rows for MNIST dataset")
    parser.add_argument("--output-json", type=str, default=None)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports: Dict[str, Any] = {}
    for ds in args.datasets:
        key = ds.lower()
        if key not in DATASET_REGISTRY:
            print(f"[WARN] Unknown dataset '{ds}', skipping.")
            continue
        try:
            ensure_dataset_prereqs(key, args)
        except FileNotFoundError as prereq_err:
            LOGGER.error("%s", prereq_err)
            print(f"[ERROR] {prereq_err}")
            continue
        spec = DATASET_REGISTRY[key]
        try:
            LOGGER.info("=== Begin %s ===", spec.name)
            report = run_single_dataset(spec, args)
            reports[spec.name] = report
            print(format_table(spec.name, report))
            LOGGER.info("=== Completed %s ===", spec.name)
        except Exception as err:  # pragma: no cover - runtime safety
            reports[spec.name] = {"error": str(err)}
            print(f"[ERROR] Failed on {spec.name}: {err}")
            LOGGER.exception("Dataset %s failed", spec.name)
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(reports, f, indent=2)


if __name__ == "__main__":
    main()
