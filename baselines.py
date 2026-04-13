"""
Baseline implementations used to evaluate lazy decision-tree ensembles.

The module exposes utilities to train the reference models, run heuristic
baselines, and compare energy/speed trade-offs. Algorithms mirror the original
research code but now feature docstrings, type hints, and clearer structure.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

NDArray = np.ndarray


def _normalize_label(label: Any) -> Any:
    """Convert numpy scalars to Python scalars for dictionary lookups."""
    value = label.item() if isinstance(label, np.generic) else label
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


class ClassIndexMapper:
    """Utility to convert arbitrary class labels into contiguous indices."""

    def __init__(self, classes: NDArray) -> None:
        self.classes = np.asarray(classes)
        self.dtype = self.classes.dtype
        self._mapping: Dict[Any, int] = {}
        for idx, label in enumerate(self.classes):
            normalized = _normalize_label(label)
            self._mapping[normalized] = idx
            coerced = self._coerce_dtype(normalized)
            self._mapping.setdefault(coerced, idx)

    def _coerce_dtype(self, label: Any) -> Any:
        try:
            return self.dtype.type(label)
        except Exception:
            return label

    def to_index(self, label: Any) -> int:
        normalized = _normalize_label(label)
        if normalized in self._mapping:
            return self._mapping[normalized]
        if isinstance(normalized, (int, np.integer)) and 0 <= normalized < len(self.classes):
            idx = int(normalized)
            self._mapping[normalized] = idx
            return idx
        coerced = self._coerce_dtype(normalized)
        if coerced in self._mapping:
            idx = self._mapping[coerced]
            self._mapping[normalized] = idx
            return idx
        matches = np.where(self.classes == normalized)[0]
        if matches.size:
            idx = int(matches[0])
            self._mapping[normalized] = idx
            return idx
        matches = np.where(self.classes == label)[0]
        if matches.size:
            idx = int(matches[0])
            self._mapping[normalized] = idx
            return idx
        raise KeyError(f"Unknown class label encountered: {label!r}")

    def batch_to_indices(self, labels: Sequence[Any]) -> np.ndarray:
        mapped = np.empty(len(labels), dtype=np.int32)
        for i, label in enumerate(labels):
            mapped[i] = self.to_index(label)
        return mapped


def _warmup_trees(rf_model: RandomForestClassifier, X: NDArray, max_trees: int = 10) -> None:
    """Run a few tree predictions to avoid cold-start penalties."""
    if len(X) == 0:
        return
    for tree in rf_model.estimators_[:max_trees]:
        _ = tree.predict(X[: min(100, len(X))])


@dataclass
class BaselineResult:
    """Summary of a baseline evaluation."""

    name: str
    accuracy: float
    inference_time: float
    avg_work_units: float
    predictions: NDArray
    model: Any
    metadata: Dict[str, Any]
    scores: Optional[NDArray] = None


def run_baseline_full_rf(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    n_estimators: int = 100,
    random_state: int = 42,
    n_jobs: int = -1,
) -> BaselineResult:
    """Train and evaluate the full RandomForestClassifier baseline."""
    rf_full = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs
    )
    train_start = time.time()
    rf_full.fit(X_train, y_train)
    train_time = time.time() - train_start

    _warmup_trees(rf_full, X_test, max_trees=10)

    inf_start = time.time()
    y_pred_full = rf_full.predict(X_test)
    inf_time = time.time() - inf_start
    acc = accuracy_score(y_test, y_pred_full)
    scores = None
    if len(rf_full.classes_) == 2 and hasattr(rf_full, "predict_proba"):
        scores = rf_full.predict_proba(X_test)[:, 1]
    return BaselineResult(
        name="Baseline A - Full RF",
        accuracy=acc,
        inference_time=inf_time,
        avg_work_units=n_estimators,
        predictions=y_pred_full,
        model=rf_full,
        metadata={"train_time": train_time},
        scores=scores,
    )


def run_fixed_cascade_baseline(
    rf_model: RandomForestClassifier,
    X_test: NDArray,
    y_test: NDArray,
    checkpoints: Sequence[int],
    thresholds: Sequence[float],
) -> BaselineResult:
    """Run the fixed-cascade (static checkpoint) baseline."""
    n_trees = len(rf_model.estimators_)
    checkpoints = list(checkpoints)
    if not checkpoints:
        raise ValueError("At least one checkpoint is required.")
    if checkpoints[-1] != n_trees:
        raise ValueError("Final checkpoint must equal the total number of trees.")
    if len(checkpoints) != len(thresholds):
        raise ValueError("Checkpoints and thresholds must match in length.")

    mapper = ClassIndexMapper(rf_model.classes_)
    n_classes = len(rf_model.classes_)
    n_samples = X_test.shape[0]

    _warmup_trees(rf_model, X_test, max_trees=10)

    start_eval = time.time()
    votes = np.zeros((n_samples, n_classes), dtype=np.int32)
    final_preds = np.empty(n_samples, dtype=rf_model.classes_.dtype)
    trees_used = np.full(n_samples, n_trees, dtype=int)
    active = np.ones(n_samples, dtype=bool)
    checkpoints_info: List[Dict[str, float]] = []
    positive_scores = np.full(n_samples, np.nan, dtype=np.float64) if n_classes == 2 else None

    prev_checkpoint = 0
    for ckpt, tau in zip(checkpoints, thresholds):
        active_indices = np.where(active)[0]
        if not active_indices.size:
            break
        X_active = X_test[active_indices]
        for tree_idx in range(prev_checkpoint, ckpt):
            preds = rf_model.estimators_[tree_idx].predict(X_active)
            try:
                mapped = mapper.batch_to_indices(preds)
            except KeyError as err:
                raise RuntimeError(f"Unknown class label from tree #{tree_idx}: {err}") from err
            np.add.at(votes, (active_indices, mapped), 1)

        total_votes = ckpt
        confidences = votes[active_indices].max(axis=1) / np.maximum(total_votes, 1)
        confident_mask = confidences >= tau
        if np.any(confident_mask):
            exiting = active_indices[confident_mask]
            final_preds[exiting] = rf_model.classes_[np.argmax(votes[exiting], axis=1)]
            if positive_scores is not None:
                positive_scores[exiting] = votes[exiting, 1] / max(ckpt, 1)
            trees_used[exiting] = ckpt
            active[exiting] = False

        checkpoints_info.append(
            {"checkpoint": ckpt, "threshold": tau, "active_fraction": float(np.mean(active))}
        )
        prev_checkpoint = ckpt

    if np.any(active):
        remaining = np.where(active)[0]
        final_preds[remaining] = rf_model.classes_[np.argmax(votes[remaining], axis=1)]
        if positive_scores is not None:
            positive_scores[remaining] = votes[remaining, 1] / max(n_trees, 1)

    inf_time = time.time() - start_eval
    acc = accuracy_score(y_test, final_preds)
    avg_trees = float(np.mean(trees_used))
    return BaselineResult(
        name="Baseline B - Fixed Cascade",
        accuracy=acc,
        inference_time=inf_time,
        avg_work_units=avg_trees,
        predictions=final_preds,
        model=rf_model,
        metadata={"checkpoints": checkpoints, "thresholds": thresholds, "stage_stats": checkpoints_info},
        scores=positive_scores,
    )


def cascade_predict(
    stage1_model: RandomForestClassifier,
    stage2_model: RandomForestClassifier,
    X: NDArray,
    confidence_threshold: float = 0.8,
) -> Tuple[NDArray, float, NDArray, Optional[NDArray]]:
    """Two-stage cascade inference helper."""
    preds_s1 = stage1_model.predict_proba(X)
    max_probs = np.max(preds_s1, axis=1)
    final_preds = np.empty(X.shape[0], dtype=stage2_model.classes_.dtype)
    mask_hard = max_probs < confidence_threshold
    positive_scores = np.full(X.shape[0], np.nan, dtype=np.float64) if len(stage2_model.classes_) == 2 else None
    if np.any(~mask_hard):
        class_indices = np.argmax(preds_s1[~mask_hard], axis=1)
        final_preds[~mask_hard] = stage1_model.classes_[class_indices]
        if positive_scores is not None:
            positive_scores[~mask_hard] = preds_s1[~mask_hard, 1]
    if np.any(mask_hard):
        if hasattr(stage2_model, "predict_proba"):
            preds_s2 = stage2_model.predict_proba(X[mask_hard])
            class_indices = np.argmax(preds_s2, axis=1)
            final_preds[mask_hard] = stage2_model.classes_[class_indices]
            if positive_scores is not None:
                positive_scores[mask_hard] = preds_s2[:, 1]
        else:
            final_preds[mask_hard] = stage2_model.predict(X[mask_hard])
    avg_trees = stage1_model.n_estimators + (np.mean(mask_hard) * stage2_model.n_estimators)
    return final_preds, avg_trees, mask_hard, positive_scores


def run_cascade_baseline(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    stage2_model: RandomForestClassifier,
    stage1_trees: int = 10,
    threshold: float = 0.9,
    random_state: int = 42,
) -> BaselineResult:
    """Train and evaluate the two-stage cascade baseline."""
    rf_stage1 = RandomForestClassifier(
        n_estimators=stage1_trees, random_state=random_state, n_jobs=-1
    )
    rf_stage1.fit(X_train, y_train)
    start_time = time.time()
    y_pred, avg_trees, mask_hard, scores = cascade_predict(rf_stage1, stage2_model, X_test, threshold)
    inf_time = time.time() - start_time
    acc = accuracy_score(y_test, y_pred)
    return BaselineResult(
        name="Cascade RF (Two-Stage)",
        accuracy=acc,
        inference_time=inf_time,
        avg_work_units=avg_trees,
        predictions=y_pred,
        model=(rf_stage1, stage2_model),
        metadata={"stage1_trees": stage1_trees, "threshold": threshold, "hard_fraction": float(np.mean(mask_hard))},
        scores=scores,
    )


class QuickScorerTree:
    """Bitmask representation of a single decision tree."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator
        self.tree_ = estimator.tree_
        self.classes_ = estimator.classes_
        self.n_classes = len(self.classes_)
        self.n_leaves = int(np.sum(self.tree_.children_left == -1))
        self.leaf_values: List[np.ndarray] = [
            np.zeros(self.n_classes, dtype=float) for _ in range(self.n_leaves)
        ]
        self.feature_checks: Dict[int, List[Tuple[float, np.ndarray, np.ndarray]]] = defaultdict(list)
        self.fallbacks = 0
        self._build_masks()

    def _build_masks(self) -> None:
        tree_ = self.tree_
        children_left = tree_.children_left
        children_right = tree_.children_right
        feature = tree_.feature
        threshold = tree_.threshold
        node_masks: Dict[int, np.ndarray] = {}
        leaf_cursor = 0

        def build(node_id: int) -> np.ndarray:
            nonlocal leaf_cursor
            if children_left[node_id] == -1:
                mask = np.zeros(self.n_leaves, dtype=bool)
                mask[leaf_cursor] = True
                self.leaf_values[leaf_cursor] = tree_.value[node_id][0]
                node_masks[node_id] = mask
                leaf_cursor += 1
                return mask
            left_mask = build(children_left[node_id])
            right_mask = build(children_right[node_id])
            node_masks[node_id] = np.logical_or(left_mask, right_mask)
            feat = feature[node_id]
            if feat >= 0:
                thr = threshold[node_id]
                self.feature_checks[feat].append((thr, node_masks[children_left[node_id]], node_masks[children_right[node_id]]))
            return node_masks[node_id]

        build(0)

    def _predict_single(self, x: np.ndarray) -> Any:
        mask = np.ones(len(self.leaf_values), dtype=bool)
        for feat in sorted(self.feature_checks.keys()):
            value = x[feat]
            for thr, left_mask, right_mask in self.feature_checks[feat]:
                mask &= left_mask if value <= thr else right_mask
                if not mask.any():
                    self.fallbacks += 1
                    return self.estimator.predict(x.reshape(1, -1))[0]
        valid = np.where(mask)[0]
        if len(valid) != 1:
            self.fallbacks += 1
            return self.estimator.predict(x.reshape(1, -1))[0]
        leaf_idx = valid[0]
        votes = self.leaf_values[leaf_idx]
        return self.classes_[np.argmax(votes)]

    def predict(self, X: NDArray) -> NDArray:
        preds = np.empty(X.shape[0], dtype=self.classes_.dtype)
        for i in range(X.shape[0]):
            preds[i] = self._predict_single(X[i])
        return preds


class QuickScorerForest:
    """Collection of QuickScorerTree objects with voting."""

    def __init__(self, rf_model: RandomForestClassifier) -> None:
        self.trees = [QuickScorerTree(tree) for tree in rf_model.estimators_]
        self.classes_ = rf_model.classes_
        self.class_mapper = ClassIndexMapper(self.classes_)
        self.last_fallback_rate = 0.0
        self.last_scores: Optional[NDArray] = None

    def predict(self, X: NDArray) -> NDArray:
        votes = np.zeros((X.shape[0], len(self.classes_)), dtype=np.float32)
        for tree in self.trees:
            preds = tree.predict(X)
            mapped = self.class_mapper.batch_to_indices(preds)
            for idx, class_idx in enumerate(mapped):
                votes[idx, class_idx] += 1
        total_fallbacks = sum(tree.fallbacks for tree in self.trees)
        self.last_fallback_rate = total_fallbacks / max(X.shape[0] * len(self.trees), 1)
        self.last_scores = votes[:, 1] / max(len(self.trees), 1) if len(self.classes_) == 2 else None
        return self.classes_[np.argmax(votes, axis=1)]


def run_quickscorer_baseline(
    rf_model: RandomForestClassifier,
    X_test: NDArray,
    y_test: NDArray,
) -> BaselineResult:
    """Evaluate the QuickScorer baseline."""
    qs_forest = QuickScorerForest(rf_model)
    start_time = time.time()
    preds = qs_forest.predict(X_test)
    inf_time = time.time() - start_time
    acc = accuracy_score(y_test, preds)
    return BaselineResult(
        name="Baseline C - QuickScorer",
        accuracy=acc,
        inference_time=inf_time,
        avg_work_units=len(rf_model.estimators_),
        predictions=preds,
        model=qs_forest,
        metadata={"description": "Bitmask traversal across features", "fallback_rate": qs_forest.last_fallback_rate},
        scores=qs_forest.last_scores,
    )


class EarlyExitMLP(nn.Module):
    """Three-block MLP with two early-exit heads."""

    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.exit1 = nn.Linear(hidden_dim, num_classes)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.exit2 = nn.Linear(hidden_dim, num_classes)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.exit_final = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h1 = self.relu(self.fc1(x))
        out1 = self.exit1(h1)
        h2 = self.relu(self.fc2(h1))
        out2 = self.exit2(h2)
        h3 = self.relu(self.fc3(h2))
        out_final = self.exit_final(h3)
        return out1, out2, out_final


def train_early_exit_network(
    X_train: NDArray,
    y_train: NDArray,
    num_classes: int,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 0.001,
) -> Tuple[EarlyExitMLP, torch.device]:
    """Train the branchy network."""
    # Ensure labels are in [0, num_classes-1] range
    y_train_remapped = y_train - y_train.min()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EarlyExitMLP(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train_remapped, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out1, out2, out_final = model(bx)
            loss = criterion(out_final, by) + 0.3 * criterion(out2, by) + 0.2 * criterion(out1, by)
            loss.backward()
            optimizer.step()
    return model, device


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    return -(probs * torch.log(probs + eps)).sum(dim=1)


def early_exit_inference(
    model: EarlyExitMLP,
    X: torch.Tensor,
    thresholds: Tuple[float, float],
) -> Tuple[NDArray, float, Dict[str, float], Optional[NDArray]]:
    """Apply early-exit policy and report exit fractions."""
    model.eval()
    with torch.no_grad():
        out1, out2, out_final = model(X)
        prob1 = torch.softmax(out1, dim=1)
        prob2 = torch.softmax(out2, dim=1)
        prob_final = torch.softmax(out_final, dim=1)
        ent1 = _entropy(prob1)
        ent2 = _entropy(prob2)
        pred1 = torch.argmax(prob1, dim=1)
        pred2 = torch.argmax(prob2, dim=1)
        pred_final = torch.argmax(prob_final, dim=1)
        mask1 = ent1 < thresholds[0]
        mask2 = (~mask1) & (ent2 < thresholds[1])
        final_preds = pred_final.clone()
        final_preds[mask2] = pred2[mask2]
        final_preds[mask1] = pred1[mask1]
        scores = None
        if prob_final.shape[1] == 2:
            scores_t = prob_final[:, 1].clone()
            scores_t[mask2] = prob2[mask2, 1]
            scores_t[mask1] = prob1[mask1, 1]
            scores = scores_t.cpu().numpy()
        depth = np.full(X.shape[0], 3.0)
        depth[mask2.cpu().numpy()] = 2.0
        depth[mask1.cpu().numpy()] = 1.0
        exit1_frac = float(mask1.float().mean().item())
        exit2_frac = float(mask2.float().mean().item())
        metadata = {
            "exit1_fraction": exit1_frac,
            "exit2_fraction": exit2_frac,
            "final_fraction": max(0.0, 1.0 - exit1_frac - exit2_frac),
        }
        return final_preds.cpu().numpy(), float(np.mean(depth)), metadata, scores


def run_early_exit_baseline(
    X_train: NDArray,
    y_train: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    num_classes: int,
    thresholds: Tuple[float, float] = (0.5, 0.3),
    epochs: int = 20,
) -> Dict[str, BaselineResult]:
    """Train the branchy network and evaluate both full and lazy passes."""
    # Ensure labels are in [0, num_classes-1] range
    label_offset = y_train.min()
    y_test_remapped = y_test - label_offset

    model, device = train_early_exit_network(X_train, y_train, num_classes=num_classes, epochs=epochs)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    start_full = time.time()
    _, _, out_final = model(X_test_t)
    prob_final = torch.softmax(out_final, dim=1)
    pred_final = torch.argmax(out_final, dim=1).cpu().numpy()
    time_full = time.time() - start_full
    acc_full = accuracy_score(y_test_remapped, pred_final)
    full_result = BaselineResult(
        name="Baseline D - Full BranchyNet",
        accuracy=acc_full,
        inference_time=time_full,
        avg_work_units=3.0,
        predictions=pred_final + label_offset,
        model=model,
        metadata={"exit": "final_only"},
        scores=prob_final[:, 1].detach().cpu().numpy() if prob_final.shape[1] == 2 else None,
    )
    start_lazy = time.time()
    y_pred_lazy, avg_depth, exit_meta, lazy_scores = early_exit_inference(model, X_test_t, thresholds)
    time_lazy = time.time() - start_lazy
    acc_lazy = accuracy_score(y_test_remapped, y_pred_lazy)
    lazy_result = BaselineResult(
        name="Baseline D - Early Exit BranchyNet",
        accuracy=acc_lazy,
        inference_time=time_lazy,
        avg_work_units=avg_depth,
        predictions=y_pred_lazy + label_offset,
        model=model,
        metadata={"thresholds": thresholds, **exit_meta},
        scores=lazy_scores,
    )
    return {"full": full_result, "early_exit": lazy_result}
