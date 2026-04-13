"""
Lazy evaluation algorithms for tree ensembles.

This module contains two estimators:
* LazyRF  - Bayesian stopping for RandomForest classifiers.
* LazyGBM - Residual-bounded stopping plus prototype margin heuristics for GradientBoostingClassifier.

The implementations preserve the original research behavior while adding
extensive documentation, validations, and type hints for maintainability.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import betainc
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    import numba  # type: ignore

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - numba optional
    NUMBA_AVAILABLE = False

    class numba:  # pragma: no cover - fallback shim
        """Minimal shim replicating the numba.jit decorator interface."""

        @staticmethod
        def jit(*args, **kwargs):
            def decorator(func):
                return func

            return decorator


@numba.jit(nopython=True)
def _gaussian_prob_scalar(alpha_best: float, alpha_second: float, total: float) -> float:
    """Gaussian approximation to P(best > second) for Dirichlet counts."""
    mu_diff = (alpha_best - alpha_second) / total
    var_best = alpha_best * (total - alpha_best)
    var_second = alpha_second * (total - alpha_second)
    cov = -alpha_best * alpha_second
    numerator = var_best + var_second - 2.0 * cov
    denom = total * total * (total + 1.0)
    sigma_diff = np.sqrt(numerator / denom)
    z = mu_diff / (sigma_diff + 1e-12)
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (z + 0.044715 * z * z * z)))


@numba.jit(nopython=True)
def _gaussian_prob_vectorized(
    alpha_best: np.ndarray, alpha_second: np.ndarray, total: np.ndarray
) -> np.ndarray:
    """Vectorized gaussian approximation helper."""
    probs = np.empty(len(alpha_best), dtype=np.float64)
    for i in range(len(alpha_best)):
        probs[i] = _gaussian_prob_scalar(alpha_best[i], alpha_second[i], total[i])
    return probs


@numba.jit(nopython=True)
def _margin_stopping_numba(
    top1_scores: np.ndarray, top2_scores: np.ndarray, remaining_bound: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Margin based multi-class stopping rule (numba accelerated)."""
    margins = top1_scores - top2_scores
    unstoppable = margins > 2.0 * remaining_bound
    return margins, unstoppable


def _to_numpy(x: ArrayLike) -> np.ndarray:
    array = np.asarray(x)
    if array.ndim == 0:
        raise ValueError("Expected array-like input, received scalar.")
    return array


def _ensure_2d(X: ArrayLike) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X.shape}.")
    return X


def _approx_norm_cdf(z_scores: np.ndarray) -> np.ndarray:
    """Fast tanh-based approximation of norm.cdf used when numba unavailable."""
    return 0.5 * (1.0 + np.tanh(0.7978845608 * (z_scores + 0.044715 * z_scores**3)))


class LazyRF(BaseEstimator, ClassifierMixin):
    """
    Lazy Random Forest inference.

    Stop checks are evaluated only at fixed block boundaries.
    """

    STOP_REASON_LABELS = {
        0: "posterior_threshold",
        1: "full_evaluation",
    }

    def __init__(
        self,
        base_estimator: BaseEstimator,
        threshold: float = 0.95,
        min_trees: int = 10,
        block_size: int = 10,
        mc_samples: int = 256,
        random_state: Optional[int] = None,
    ) -> None:
        if not hasattr(base_estimator, "estimators_"):
            raise ValueError("Base estimator must be a fitted RandomForestClassifier.")

        self.base_estimator = base_estimator
        self.threshold = float(threshold)
        self.min_trees = int(min_trees)
        self.block_size = max(1, int(block_size))
        self.mc_samples = int(mc_samples)
        self.rng = np.random.default_rng(random_state)

        self.n_classes_: int = base_estimator.n_classes_
        self.classes_: np.ndarray = base_estimator.classes_
        self._class_dtype = self.classes_.dtype
        self.estimators_: Sequence = base_estimator.estimators_

        self._class_to_idx: Dict[Any, int] = {}
        for idx, label in enumerate(self.classes_):
            normalized = self._normalize_label(label)
            self._class_to_idx[normalized] = idx
            coerced = self._coerce_label(normalized)
            self._class_to_idx.setdefault(coerced, idx)

    @staticmethod
    def _normalize_label(label: Any) -> Any:
        """Return a hashable python scalar for numpy inputs."""
        value = label.item() if isinstance(label, np.generic) else label
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    def _coerce_label(self, label: Any) -> Any:
        """Cast label values to the RF class dtype when possible."""
        try:
            return self._class_dtype.type(label)
        except Exception:
            return label

    def _label_to_index(self, label: Any) -> int:
        """Map arbitrary label to class index, handling float/int mismatches."""
        normalized = self._normalize_label(label)
        if normalized in self._class_to_idx:
            return self._class_to_idx[normalized]
        if isinstance(normalized, (int, np.integer)) and 0 <= normalized < self.n_classes_:
            idx = int(normalized)
            self._class_to_idx[normalized] = idx
            return idx
        coerced = self._coerce_label(normalized)
        if coerced in self._class_to_idx:
            idx = self._class_to_idx[coerced]
            self._class_to_idx[normalized] = idx
            return idx
        matches = np.where(self.classes_ == normalized)[0]
        if matches.size:
            idx = int(matches[0])
            self._class_to_idx[normalized] = idx
            return idx
        matches = np.where(self.classes_ == label)[0]
        if matches.size:
            return int(matches[0])
        raise KeyError(f"Unknown class label encountered: {label}")

    def _sample_posterior_probs(
        self, alphas_active: np.ndarray, best_idx: np.ndarray
    ) -> np.ndarray:
        total_counts = np.sum(alphas_active, axis=1)
        probs = np.zeros(len(alphas_active))
        if self.n_classes_ == 2:
            best_counts = alphas_active[np.arange(len(alphas_active)), best_idx]
            other_counts = total_counts - best_counts
            return 1.0 - betainc(best_counts, other_counts, 0.5)

        use_exact = total_counts < 30
        if np.any(use_exact):
            exact_idx = np.where(use_exact)[0]
            sample_count = max(self.mc_samples, self.n_classes_ * 32)
            for idx in exact_idx:
                alpha_vec = alphas_active[idx]
                samples = self.rng.dirichlet(alpha_vec, size=sample_count)
                probs[idx] = np.mean(np.argmax(samples, axis=1) == best_idx[idx])

        if np.any(~use_exact):
            approx_idx = np.where(~use_exact)[0]
            current_alphas = alphas_active[approx_idx]
            totals = total_counts[approx_idx]
            row_idx = np.arange(len(approx_idx))
            alpha_best = current_alphas[row_idx, best_idx[approx_idx]]
            masked = current_alphas.copy()
            masked[row_idx, best_idx[approx_idx]] = -1
            runner_up_idx = np.argmax(masked, axis=1)
            alpha_second = current_alphas[row_idx, runner_up_idx]
            if NUMBA_AVAILABLE:
                probs[approx_idx] = _gaussian_prob_vectorized(alpha_best, alpha_second, totals)
            else:
                mu_diff = (alpha_best - alpha_second) / totals
                var_best = alpha_best * (totals - alpha_best)
                var_second = alpha_second * (totals - alpha_second)
                cov = -alpha_best * alpha_second
                numerator = var_best + var_second - 2 * cov
                denom = totals**2 * (totals + 1)
                sigma = np.sqrt(numerator / denom)
                z_scores = mu_diff / (sigma + 1e-12)
                probs[approx_idx] = _approx_norm_cdf(z_scores)
        return probs

    def _predict_lazy_internal(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Core LazyRF inference loop returning per-sample diagnostics."""
        n_samples = X.shape[0]
        n_trees = len(self.estimators_)
        alphas = np.ones((n_samples, self.n_classes_), dtype=np.float64)
        final_preds = np.full(n_samples, -1, dtype=int)
        trees_used = np.zeros(n_samples, dtype=int)
        stop_scores = np.full(n_samples, np.nan, dtype=np.float64)
        stop_reasons = np.full(n_samples, -1, dtype=np.int8)
        active_mask = np.ones(n_samples, dtype=bool)

        def accumulate_block(start_idx: int, end_idx: int, sample_idx: np.ndarray) -> None:
            if sample_idx.size == 0:
                return
            X_block = X[sample_idx]
            for tree_idx in range(start_idx, end_idx):
                tree = self.estimators_[tree_idx]
                preds = tree.predict(X_block)
                mapped = np.searchsorted(self.classes_, preds)
                valid = mapped >= 0
                if np.any(valid):
                    np.add.at(alphas, (sample_idx[valid], mapped[valid]), 1)

        def apply_stopping(active_indices: np.ndarray, evaluated: int) -> None:
            if active_indices.size == 0:
                return
            alphas_active = alphas[active_indices]
            best_idx = np.argmax(alphas_active, axis=1)
            probs = self._sample_posterior_probs(alphas_active, best_idx)
            confident = (probs > self.threshold) & (evaluated >= self.min_trees)
            exiting = active_indices[confident]
            if exiting.size:
                final_preds[exiting] = best_idx[confident]
                trees_used[exiting] = evaluated
                stop_scores[exiting] = probs[confident]
                stop_reasons[exiting] = 0
                active_mask[exiting] = False

        for start in range(0, n_trees, self.block_size):
            end = min(start + self.block_size, n_trees)
            active_idx = np.where(active_mask)[0]
            if not active_idx.size:
                break
            accumulate_block(start, end, active_idx)
            if end >= self.min_trees:
                apply_stopping(active_idx, end)

        if np.any(active_mask):
            leftover = np.where(active_mask)[0]
            final_preds[leftover] = np.argmax(alphas[leftover], axis=1)
            trees_used[leftover] = n_trees
            stop_reasons[leftover] = 1

        missing_score = np.isnan(stop_scores)
        if np.any(missing_score):
            missing_idx = np.where(missing_score)[0]
            alphas_missing = alphas[missing_idx]
            best_idx = np.argmax(alphas_missing, axis=1)
            stop_scores[missing_idx] = self._sample_posterior_probs(alphas_missing, best_idx)

        class_posteriors = alphas / np.sum(alphas, axis=1, keepdims=True)

        return (
            self.classes_[final_preds],
            float(np.mean(trees_used)),
            trees_used,
            stop_scores,
            stop_reasons,
            class_posteriors,
        )

    def predict_lazy(self, X: ArrayLike) -> Tuple[np.ndarray, float]:
        """Return lazy predictions along with average evaluated trees."""
        X = _ensure_2d(X)
        preds, avg_trees, _, _, _, _ = self._predict_lazy_internal(X)
        return preds, avg_trees

    def predict_lazy_with_details(self, X: ArrayLike) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
        """Return lazy predictions + average work + per-sample stopping diagnostics."""
        X = _ensure_2d(X)
        preds, avg_trees, trees_used, stop_scores, stop_reasons, class_posteriors = self._predict_lazy_internal(X)
        details = {
            "trees_used": trees_used,
            "stop_scores": stop_scores,
            "stop_reasons": stop_reasons,
            "class_posteriors": class_posteriors,
        }
        return preds, avg_trees, details


class LazyGBM(BaseEstimator, ClassifierMixin):
    """Lazy evaluation for GradientBoostingClassifier."""

    STOP_REASON_LABELS = {
        0: "certificate",
        1: "ratio_heuristic",
        2: "flip_score",
        3: "late_margin",
        4: "full_evaluation",
    }

    def __init__(
        self,
        base_estimator: BaseEstimator,
        spr_threshold: float = 4.6,
        min_trees: int = 10,
        block_size: int = 1,
        enable_ratio_heuristic: bool = True,
        ratio_threshold: float = 1.5,
        enable_flip_heuristic: bool = True,
        multiclass_flip_scale: float = 2.0,
        enable_late_margin_fallback: bool = True,
        late_margin_fraction: float = 0.8,
    ) -> None:
        if not hasattr(base_estimator, "estimators_"):
            raise ValueError("Base estimator must be a fitted GradientBoostingClassifier.")

        self.base_estimator = base_estimator
        self.spr_threshold = spr_threshold
        self.min_trees = int(min_trees)
        self.block_size = max(1, int(block_size))
        self.enable_ratio_heuristic = bool(enable_ratio_heuristic)
        self.ratio_threshold = float(ratio_threshold)
        self.enable_flip_heuristic = bool(enable_flip_heuristic)
        self.multiclass_flip_scale = float(multiclass_flip_scale)
        self.enable_late_margin_fallback = bool(enable_late_margin_fallback)
        self.late_margin_fraction = float(late_margin_fraction)

        self.estimators_ = base_estimator.estimators_
        self.init_estimator = base_estimator.init_
        self.learning_rate = base_estimator.learning_rate
        self.classes_ = base_estimator.classes_
        self.n_classes_ = len(self.classes_)

        self._precompute_bounds()

    def _precompute_bounds(self) -> None:
        w_max: list[float] = []
        n_stages, n_class_cols = self.estimators_.shape
        for stage in range(n_stages):
            stage_max = 0.0
            for class_idx in range(n_class_cols):
                tree = self.estimators_[stage, class_idx]
                leaf_values = tree.tree_.value.squeeze() * self.learning_rate
                stage_max = max(stage_max, np.max(np.abs(leaf_values)))
            w_max.append(stage_max)

        self.w_max = np.array(w_max)
        self.residual_bounds = np.zeros(n_stages)
        self.residual_range_sq = np.zeros(n_stages)

        future_sum = 0.0
        future_range_sq = 0.0
        for idx in range(n_stages - 1, -1, -1):
            self.residual_bounds[idx] = future_sum
            self.residual_range_sq[idx] = future_range_sq
            future_sum += self.w_max[idx]
            future_range_sq += self.w_max[idx] ** 2

        self.flip_prob_threshold = np.exp(-self.spr_threshold)

    def _init_raw_predictions(self, X: np.ndarray) -> np.ndarray:
        is_multiclass = self.n_classes_ > 2
        constant = None
        if hasattr(self.init_estimator, "priors"):
            constant = self.init_estimator.priors
        elif hasattr(self.init_estimator, "prior"):
            constant = self.init_estimator.prior

        if constant is not None:
            if is_multiclass:
                return np.tile(constant, (X.shape[0], 1)).astype(np.float64)
            val = constant.item() if np.ndim(constant) else constant
            return np.full(X.shape[0], val, dtype=np.float64)

        raw_pred = self.base_estimator._raw_predict_init(X).astype(np.float64)
        if not is_multiclass:
            if raw_pred.ndim > 1 and raw_pred.shape[1] == 1:
                raw_pred = raw_pred.ravel()
        else:
            if raw_pred.ndim != 2 or raw_pred.shape[1] != self.n_classes_:
                raise ValueError(f"Unexpected init shape: {raw_pred.shape}")
        return raw_pred

    def _predict_lazy_internal(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Core LazyGBM loop returning per-sample stopping diagnostics."""
        X = _ensure_2d(X)
        n_samples = X.shape[0]
        n_stages = self.estimators_.shape[0]
        is_multiclass = self.n_classes_ > 2

        raw_predictions = self._init_raw_predictions(X)
        trees_used = np.zeros(n_samples, dtype=int)
        active_mask = np.ones(n_samples, dtype=bool)
        final_preds = np.empty(n_samples, dtype=self.classes_.dtype)
        stop_reasons = np.full(n_samples, -1, dtype=np.int8)
        margins_at_stop = np.full(n_samples, np.nan, dtype=np.float64)
        flip_scores = np.full(n_samples, np.nan, dtype=np.float64)

        if not is_multiclass:
            all_tree_preds = np.array([stage[0].predict(X) * self.learning_rate for stage in self.estimators_])

        for start in range(0, n_stages, self.block_size):
            end = min(start + self.block_size, n_stages)
            active_indices = np.where(active_mask)[0]
            if not active_indices.size:
                break
            X_active = X[active_indices]

            if is_multiclass:
                for stage in range(start, end):
                    contributions = np.zeros((len(active_indices), self.n_classes_))
                    for cls in range(self.n_classes_):
                        contributions[:, cls] = self.estimators_[stage, cls].predict(X_active)
                    raw_predictions[active_indices] += contributions * self.learning_rate
            else:
                block = all_tree_preds[start:end]
                raw_predictions[active_indices] += np.sum(block[:, active_indices], axis=0)

            step_idx = end - 1
            remaining_bound = self.residual_bounds[step_idx]
            remaining_range_sq = self.residual_range_sq[step_idx]

            if is_multiclass:
                current_logits = raw_predictions[active_indices]
                partitioned = np.partition(current_logits, -2, axis=1)
                top1 = partitioned[:, -1]
                top2 = partitioned[:, -2]
                if NUMBA_AVAILABLE:
                    margins, unstoppable = _margin_stopping_numba(top1, top2, remaining_bound)
                else:
                    margins = top1 - top2
                    unstoppable = margins > 2 * remaining_bound
                reason_codes = np.full(len(active_indices), -1, dtype=np.int8)
                final_mask = unstoppable.copy()
                reason_codes[unstoppable] = 0
                if self.enable_ratio_heuristic and remaining_bound > 0:
                    conf_ratio = margins / (2 * remaining_bound + 1e-9)
                    ratio_mask = (~final_mask) & (conf_ratio > self.ratio_threshold)
                    final_mask |= ratio_mask
                    reason_codes[ratio_mask] = 1
                if end >= self.min_trees and self.enable_flip_heuristic:
                    unresolved = ~final_mask
                    if np.any(unresolved):
                        if remaining_range_sq > 0:
                            prob_flip = np.exp(-(margins[unresolved] ** 2) / (2 * (remaining_range_sq + 1e-12)))
                            flip_scores[active_indices[unresolved]] = prob_flip
                            prob_stop = prob_flip < self.flip_prob_threshold * self.multiclass_flip_scale
                            idx = np.where(unresolved)[0][prob_stop]
                            final_mask[idx] = True
                            reason_codes[idx] = 2
                        else:
                            final_mask[unresolved] = True
                            reason_codes[unresolved] = 2
                late_threshold = int(self.late_margin_fraction * n_stages)
                if self.enable_late_margin_fallback and end > late_threshold:
                    late_mask = (~final_mask) & (margins > 0)
                    final_mask |= late_mask
                    reason_codes[late_mask] = 3
                newly_done = active_indices[final_mask]
                if newly_done.size:
                    best = np.argmax(raw_predictions[newly_done], axis=1)
                    final_preds[newly_done] = self.classes_[best]
                    trees_used[newly_done] = end
                    stop_reasons[newly_done] = reason_codes[final_mask]
                    margins_at_stop[newly_done] = margins[final_mask]
                    active_mask[newly_done] = False
            else:
                margins = raw_predictions[active_indices]
                unstoppable_pos = margins - remaining_bound > 0
                unstoppable_neg = margins + remaining_bound < 0
                reason_codes = np.full(len(active_indices), -1, dtype=np.int8)
                final_mask = unstoppable_pos | unstoppable_neg
                reason_codes[final_mask] = 0
                if end >= self.min_trees and self.enable_flip_heuristic:
                    unresolved = ~final_mask
                    if np.any(unresolved):
                        if remaining_range_sq > 0:
                            prob_flip = np.exp(-(np.abs(margins[unresolved]) ** 2) / (2 * (remaining_range_sq + 1e-12)))
                            flip_scores[active_indices[unresolved]] = prob_flip
                            idx = np.where(unresolved)[0][prob_flip < self.flip_prob_threshold]
                            final_mask[idx] = True
                            reason_codes[idx] = 2
                        else:
                            final_mask[unresolved] = True
                            reason_codes[unresolved] = 2
                newly_done = active_indices[final_mask]
                if newly_done.size:
                    final_preds[newly_done] = np.where(
                        raw_predictions[newly_done] >= 0, self.classes_[1], self.classes_[0]
                    )
                    trees_used[newly_done] = end
                    stop_reasons[newly_done] = reason_codes[final_mask]
                    margins_at_stop[newly_done] = margins[final_mask]
                    active_mask[newly_done] = False

            if not np.any(active_mask):
                break

        if np.any(active_mask):
            remaining = np.where(active_mask)[0]
            if is_multiclass:
                best = np.argmax(raw_predictions[remaining], axis=1)
                final_preds[remaining] = self.classes_[best]
                partitioned = np.partition(raw_predictions[remaining], -2, axis=1)
                margins_at_stop[remaining] = partitioned[:, -1] - partitioned[:, -2]
            else:
                final_preds[remaining] = np.where(
                    raw_predictions[remaining] >= 0, self.classes_[1], self.classes_[0]
                )
                margins_at_stop[remaining] = raw_predictions[remaining]
            trees_used[remaining] = n_stages
            stop_reasons[remaining] = 4

        return final_preds, float(np.mean(trees_used)), trees_used, stop_reasons, {
            "margins_at_stop": margins_at_stop,
            "flip_scores": flip_scores,
        }

    def predict_lazy(self, X: ArrayLike) -> Tuple[np.ndarray, float]:
        """Run lazy evaluation and return predictions + average depth."""
        preds, avg_trees, _, _, _ = self._predict_lazy_internal(_ensure_2d(X))
        return preds, avg_trees

    def predict_lazy_with_details(self, X: ArrayLike) -> Tuple[np.ndarray, float, Dict[str, np.ndarray]]:
        """Return lazy predictions + average depth + per-sample stopping diagnostics."""
        preds, avg_trees, trees_used, stop_reasons, aux = self._predict_lazy_internal(_ensure_2d(X))
        details = {
            "trees_used": trees_used,
            "stop_reasons": stop_reasons,
            "margins_at_stop": aux["margins_at_stop"],
            "flip_scores": aux["flip_scores"],
        }
        return preds, avg_trees, details
