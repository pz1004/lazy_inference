"""
Lazy evaluation algorithms for tree ensembles.

This module contains two estimators:
* LazyRF  - Bayesian stopping for RandomForest classifiers.
* LazyGBM - Residual-bounded / SPRT stopping for GradientBoostingClassifier.

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

    Parameters mirror the original implementation. See predict_lazy for usage.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        threshold: float = 0.99,
        min_trees: int = 10,
        block_size: int = 10,
        fast_start_trees: Optional[int] = None,
        mc_samples: int = 256,
        random_state: Optional[int] = None,
        hybrid_mode: bool = False,
    ) -> None:
        if not hasattr(base_estimator, "estimators_"):
            raise ValueError("Base estimator must be a fitted RandomForestClassifier.")

        self.base_estimator = base_estimator
        self.threshold = float(threshold)
        self.min_trees = int(min_trees)
        self.block_size = max(1, int(block_size))
        self.fast_start_trees = (
            int(fast_start_trees) if fast_start_trees is not None else self.min_trees
        )
        self.mc_samples = int(mc_samples)
        self.rng = np.random.default_rng(random_state)
        self.hybrid_mode = bool(hybrid_mode)

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

    def predict_lazy(self, X: ArrayLike) -> Tuple[np.ndarray, float]:
        """Return lazy predictions along with average evaluated trees."""
        X = _ensure_2d(X)
        n_samples = X.shape[0]
        n_trees = len(self.estimators_)
        alphas = np.ones((n_samples, self.n_classes_), dtype=np.float64)
        final_preds = np.full(n_samples, -1, dtype=int)
        trees_used = np.zeros(n_samples, dtype=int)
        active_mask = np.ones(n_samples, dtype=bool)

        def accumulate_block(start_idx: int, end_idx: int, sample_idx: np.ndarray) -> None:
            if sample_idx.size == 0:
                return
            X_block = X[sample_idx]
            for tree_idx in range(start_idx, end_idx):
                tree = self.estimators_[tree_idx]
                preds = tree.predict(X_block)
                mapped = np.array([self._label_to_index(pred) for pred in preds])
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
                active_mask[exiting] = False

        trees_evaluated = 0
        hybrid_cutoff = 0
        if self.hybrid_mode:
            hybrid_cutoff = max(self.min_trees, n_trees // 2)
            for tree in self.estimators_[:hybrid_cutoff]:
                preds = tree.predict(X)
                mapped = np.array([self._label_to_index(pred) for pred in preds])
                valid = mapped >= 0
                if np.any(valid):
                    np.add.at(alphas, (np.arange(n_samples)[valid], mapped[valid]), 1)
            trees_evaluated = hybrid_cutoff
            apply_stopping(np.arange(n_samples), trees_evaluated)

        fast_start = max(hybrid_cutoff, min(self.fast_start_trees, n_trees))
        if fast_start > hybrid_cutoff:
            initial_idx = np.arange(n_samples)
            accumulate_block(hybrid_cutoff, fast_start, initial_idx)
            trees_evaluated = fast_start
            apply_stopping(initial_idx, trees_evaluated)

        for start in range(max(fast_start, trees_evaluated), n_trees, self.block_size):
            end = min(start + self.block_size, n_trees)
            active_idx = np.where(active_mask)[0]
            if not active_idx.size:
                break
            accumulate_block(start, end, active_idx)
            trees_evaluated = end
            apply_stopping(active_idx, trees_evaluated)

        if np.any(active_mask):
            leftover = np.where(active_mask)[0]
            final_preds[leftover] = np.argmax(alphas[leftover], axis=1)
            trees_used[leftover] = n_trees

        return self.classes_[final_preds], float(np.mean(trees_used))


class LazyGBM(BaseEstimator, ClassifierMixin):
    """Lazy evaluation for GradientBoostingClassifier."""

    def __init__(
        self,
        base_estimator: BaseEstimator,
        spr_threshold: float = 4.6,
        min_trees: int = 10,
        block_size: int = 1,
    ) -> None:
        if not hasattr(base_estimator, "estimators_"):
            raise ValueError("Base estimator must be a fitted GradientBoostingClassifier.")

        self.base_estimator = base_estimator
        self.spr_threshold = spr_threshold
        self.min_trees = int(min_trees)
        self.block_size = max(1, int(block_size))

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

    def predict_lazy(self, X: ArrayLike) -> Tuple[np.ndarray, float]:
        """Run lazy evaluation and return predictions + average depth."""
        X = _ensure_2d(X)
        n_samples = X.shape[0]
        n_stages = self.estimators_.shape[0]
        is_multiclass = self.n_classes_ > 2

        raw_predictions = self._init_raw_predictions(X)
        trees_used = np.zeros(n_samples, dtype=int)
        active_mask = np.ones(n_samples, dtype=bool)
        final_preds = np.empty(n_samples, dtype=self.classes_.dtype)

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
                final_mask = unstoppable.copy()
                if remaining_bound > 0:
                    conf_ratio = margins / (2 * remaining_bound + 1e-9)
                    final_mask |= conf_ratio > 1.5
                if end >= self.min_trees:
                    unresolved = ~final_mask
                    if np.any(unresolved):
                        if remaining_range_sq > 0:
                            prob_flip = np.exp(-(margins[unresolved] ** 2) / (2 * (remaining_range_sq + 1e-12)))
                            prob_stop = prob_flip < self.flip_prob_threshold * 2.0
                            idx = np.where(unresolved)[0][prob_stop]
                            final_mask[idx] = True
                        else:
                            final_mask[unresolved] = True
                if end > int(0.8 * n_stages):
                    final_mask |= margins > 0
                newly_done = active_indices[final_mask]
                if newly_done.size:
                    best = np.argmax(raw_predictions[newly_done], axis=1)
                    final_preds[newly_done] = self.classes_[best]
                    trees_used[newly_done] = end
                    active_mask[newly_done] = False
            else:
                margins = raw_predictions[active_indices]
                unstoppable_pos = margins - remaining_bound > 0
                unstoppable_neg = margins + remaining_bound < 0
                final_mask = unstoppable_pos | unstoppable_neg
                if end >= self.min_trees:
                    unresolved = ~final_mask
                    if np.any(unresolved):
                        if remaining_range_sq > 0:
                            prob_flip = np.exp(-(np.abs(margins[unresolved]) ** 2) / (2 * (remaining_range_sq + 1e-12)))
                            idx = np.where(unresolved)[0][prob_flip < self.flip_prob_threshold]
                            final_mask[idx] = True
                        else:
                            final_mask[unresolved] = True
                newly_done = active_indices[final_mask]
                if newly_done.size:
                    final_preds[newly_done] = np.where(
                        raw_predictions[newly_done] >= 0, self.classes_[1], self.classes_[0]
                    )
                    trees_used[newly_done] = end
                    active_mask[newly_done] = False

            if not np.any(active_mask):
                break

        if np.any(active_mask):
            remaining = np.where(active_mask)[0]
            if is_multiclass:
                best = np.argmax(raw_predictions[remaining], axis=1)
                final_preds[remaining] = self.classes_[best]
            else:
                final_preds[remaining] = np.where(
                    raw_predictions[remaining] >= 0, self.classes_[1], self.classes_[0]
                )
            trees_used[remaining] = n_stages

        return final_preds, float(np.mean(trees_used))
