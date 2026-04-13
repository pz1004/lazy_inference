#!/usr/bin/env python3
"""
Quick semantic checks for the fixed-grid LazyRF implementation.

This script validates the block-boundary behavior required by Algorithm 1:
  - t_min values below the first block boundary do not change the checkpoint grid
  - stop checks happen only at multiples of B
  - larger block sizes move the first eligible checkpoint to the next block boundary
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lazy_evaluation import LazyRF


def _run_lazy_rf(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    *,
    threshold: float,
    min_trees: int,
    block_size: int,
    random_state: int,
):
    lazy_rf = LazyRF(
        model,
        threshold=threshold,
        min_trees=min_trees,
        block_size=block_size,
        random_state=random_state,
    )
    preds, _, details = lazy_rf.predict_lazy_with_details(X_test)
    return preds, details


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.35,
        random_state=123,
        stratify=y,
    )
    rf = RandomForestClassifier(
        n_estimators=40,
        random_state=123,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    threshold = 0.90
    random_state = 123

    preds_m0, details_m0 = _run_lazy_rf(
        rf,
        X_test,
        threshold=threshold,
        min_trees=0,
        block_size=10,
        random_state=random_state,
    )
    preds_m5, details_m5 = _run_lazy_rf(
        rf,
        X_test,
        threshold=threshold,
        min_trees=5,
        block_size=10,
        random_state=random_state,
    )
    preds_m20, details_m20 = _run_lazy_rf(
        rf,
        X_test,
        threshold=threshold,
        min_trees=20,
        block_size=10,
        random_state=random_state,
    )
    preds_b20, details_b20 = _run_lazy_rf(
        rf,
        X_test,
        threshold=threshold,
        min_trees=10,
        block_size=20,
        random_state=random_state,
    )

    expected_keys = {"trees_used", "stop_scores", "stop_reasons", "class_posteriors"}
    _assert(expected_keys.issubset(details_m0.keys()), "LazyRF details are missing expected keys.")
    _assert(preds_m0.shape == preds_m5.shape == preds_m20.shape == preds_b20.shape, "Prediction shapes differ.")
    _assert(details_m0["class_posteriors"].shape[0] == X_test.shape[0], "Posterior output shape mismatch.")

    _assert(np.array_equal(preds_m0, preds_m5), "t_min=0 and t_min=5 should be identical at B=10.")
    _assert(
        np.array_equal(details_m0["trees_used"], details_m5["trees_used"]),
        "trees_used differs between t_min=0 and t_min=5 at B=10.",
    )
    _assert(
        np.array_equal(details_m0["stop_reasons"], details_m5["stop_reasons"]),
        "stop_reasons differ between t_min=0 and t_min=5 at B=10.",
    )
    _assert(
        np.allclose(details_m0["stop_scores"], details_m5["stop_scores"]),
        "stop_scores differ between t_min=0 and t_min=5 at B=10.",
    )

    _assert(
        np.all(details_m20["trees_used"] >= 20),
        "t_min=20 should prevent stopping before 20 evaluated trees.",
    )
    _assert(
        np.all(details_m20["trees_used"] % 10 == 0),
        "With B=10, all stopping counts should lie on 10-tree block boundaries.",
    )
    _assert(
        np.all(details_b20["trees_used"] >= 20),
        "With B=20 and t_min=10, the first eligible checkpoint should be at 20 trees.",
    )
    _assert(
        np.all(details_b20["trees_used"] % 20 == 0),
        "With B=20, all stopping counts should lie on 20-tree block boundaries.",
    )

    print("LazyRF Algorithm 1 semantics check passed.")
    print(
        "B=10,t_min=0 avg_work={:.2f}; B=10,t_min=5 avg_work={:.2f}; "
        "B=10,t_min=20 avg_work={:.2f}; B=20,t_min=10 avg_work={:.2f}".format(
            float(np.mean(details_m0["trees_used"])),
            float(np.mean(details_m5["trees_used"])),
            float(np.mean(details_m20["trees_used"])),
            float(np.mean(details_b20["trees_used"])),
        )
    )


if __name__ == "__main__":
    main()
