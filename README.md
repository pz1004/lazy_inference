# Instance-Adaptive Lazy Inference for Tree Ensembles

This repository provides inference-time wrappers for pre-trained tree ensembles (Random Forests and Gradient Boosted Decision Trees) that adaptively stop evaluation early when the prediction is sufficiently stable, reducing computational cost without modifying the trained model.

## Overview

Tree ensembles evaluate all components for every input, incurring fixed computational costs regardless of instance difficulty. However, many inputs are "easy" and can be classified reliably after evaluating only a fraction of the ensemble.

This implementation provides two lazy inference wrappers:

- **LazyRF**: For Random Forests - uses a Bayesian posterior predictive model to determine when the majority vote is unlikely to change
- **LazyGBM**: For Gradient Boosting - uses residual-capacity certificates and concentration-inspired bounds for early stopping

### Key Features

- **Model-preserving**: Works as inference-time wrappers around pre-trained scikit-learn models
- **Instance-adaptive**: Stops early on easy inputs, uses full evaluation for hard inputs
- **Worst-case guarantee**: Always bounded by full model evaluation
- **Configurable trade-offs**: Adjustable thresholds to control accuracy-efficiency trade-off

## Methods

### LazyRF (Lazy Random Forest)

LazyRF models the remaining forest votes using a Dirichlet-multinomial posterior predictive distribution and computes a stability surrogate that estimates the probability that the current leading class remains the final argmax.

**Key parameters:**
- `threshold` (α_stop): Confidence threshold for early stopping (default: 0.95)
- `min_trees`: Minimum trees to evaluate before checking stopping condition (default: 10)
- `block_size`: Number of trees to evaluate between stopping checks (default: 10)

### LazyGBM (Lazy Gradient Boosting)

LazyGBM exploits the additive structure of boosting to derive stopping conditions:

1. **Lossless mode**: Uses a deterministic residual-capacity certificate - stops when the current margin cannot be overturned by remaining stages
2. **Near-lossless mode**: Uses a concentration-inspired stability score for tunable early stopping

**Key parameters:**
- `spr_threshold` (γ): Stability threshold for near-lossless stopping (default: 3.0)
- `min_trees`: Minimum stages before checking stopping condition (default: 10)

## Installation

```bash
# Clone the repository
git clone https://github.com/pz1004/lazy_inference.git
cd lazy_inference

# Install dependencies
pip install numpy scipy scikit-learn torch torchvision pandas
# Optional: pip install numba  # For JIT-accelerated computations
```

## Usage

### Basic Usage

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lazy_evaluation import LazyRF, LazyGBM

# Train a standard Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Wrap with LazyRF for efficient inference
lazy_rf = LazyRF(rf, threshold=0.95, min_trees=10)
predictions, avg_trees_used = lazy_rf.predict_lazy(X_test)
print(f"Average trees evaluated: {avg_trees_used:.1f} / 100")

# Train a standard Gradient Boosting Classifier
gbm = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbm.fit(X_train, y_train)

# Wrap with LazyGBM for efficient inference
lazy_gbm = LazyGBM(gbm, spr_threshold=3.0, min_trees=10)
predictions, avg_stages_used = lazy_gbm.predict_lazy(X_test)
print(f"Average stages evaluated: {avg_stages_used:.1f} / 100")
```

### Running Experiments

```bash
# Run experiments on all datasets
python run_experiments.py

# Run on specific datasets
python run_experiments.py --datasets mnist covertype

# Run specific methods only
python run_experiments.py --methods lazy_rf lazy_gbm full_rf full_gbm

# Threshold sweep for LazyRF
python run_experiments.py --lazy-rf-thresholds 0.90 0.95 0.97 0.99

# Threshold sweep for LazyGBM
python run_experiments.py --lazy-gbm-thresholds 2.0 3.0 4.0

# Save results to JSON
python run_experiments.py --output-json results/experiment_results.json
```

### Multi-seed Experiments

```bash
# Run experiments across multiple random seeds
python run_multi_experiments.py --seeds 30 --output-json results/multi_seed_results.json
```

## Datasets

The experiments use four benchmark classification datasets:

| Dataset | Type | Classes | Features | Instances | Difficulty |
|---------|------|---------|----------|-----------|------------|
| MNIST | Image | 10 | 784 | 70,000 | Low (easy digits exit early) |
| Covertype | Tabular | 7 | 54 | 581,000 | Medium (complex boundaries) |
| Higgs | Physics | 2 | 28 | 11,000,000 | High (noise, requires depth) |
| Credit Card | Anomaly | 2 | 30 | 284,000 | Variable (fraud is hard, normal is easy) |

Dataset paths can be configured via command-line arguments or environment variables:
- `--higgs-path` or `HIGGS_PATH`
- `--credit-path` or `CREDIT_CARD_PATH`

## Results Summary

### LazyRF Performance

LazyRF achieves substantial work reductions while maintaining accuracy close to the full Random Forest:

| Dataset | Full RF Accuracy | LazyRF Accuracy | Avg. Trees Used | Work Reduction |
|---------|------------------|-----------------|-----------------|----------------|
| Covertype | 0.9271 | 0.9267 | 19.16 | 81% |
| Credit Card | 0.9995 | 0.9995 | 10.02 | 90% |
| Higgs | 0.7218 | 0.7205 | 39.60 | 60% |
| MNIST | 0.9695 | 0.9694 | 17.50 | 83% |

### LazyGBM Performance

LazyGBM provides more modest, dataset-dependent reductions:

| Dataset | Full GBM Accuracy | LazyGBM Accuracy | Avg. Stages Used | Work Reduction |
|---------|-------------------|------------------|------------------|----------------|
| Covertype | 0.7721 | 0.7672 | 81.00 | 19% |
| Credit Card | 0.9987 | 0.9987 | 31.67 | 68% |
| Higgs | 0.7120 | 0.7120 | 80.29 | 20% |
| MNIST | 0.9459 | 0.9419 | 81.00 | 19% |

## Project Structure

```
lazy_inference/
├── lazy_evaluation.py      # Core LazyRF and LazyGBM implementations
├── baselines.py            # Baseline methods (cascades, QuickScorer, BranchyNet)
├── run_experiments.py      # Single-seed experiment runner
├── run_multi_experiments.py # Multi-seed experiment runner
├── table_figures.py        # Result visualization utilities
└── README.md
```

## Baselines

The repository includes several baseline methods for comparison:

- **Full RF/GBM**: Standard full-ensemble evaluation
- **Fixed Cascade**: Static checkpoint-based early exit
- **Two-Stage Cascade**: Two-stage RF with confidence-based routing
- **QuickScorer**: Bitmask-based tree traversal (constant-factor speedup)
- **BranchyNet**: Early-exit neural network baseline

## Configuration Options

### run_experiments.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--datasets` | all | Datasets to evaluate |
| `--methods` | all | Methods to run |
| `--rf-trees` | 100 | Number of trees in Random Forest |
| `--gbm-trees` | 100 | Number of stages in GBM |
| `--lazy-rf-threshold` | 0.95 | LazyRF confidence threshold |
| `--lazy-rf-min-trees` | 10 | Minimum trees before LazyRF can stop |
| `--lazy-gbm-threshold` | 3.0 | LazyGBM stability threshold |
| `--lazy-gbm-min-trees` | 10 | Minimum stages before LazyGBM can stop |
| `--random-state` | 42 | Random seed |
| `--output-json` | None | Path to save JSON results |

## Notes

- **Work units** are an implementation-agnostic proxy for computational cost (trees for RF, stages for GBM). Wall-clock speedups may vary based on implementation details.
- LazyRF's stability surrogate uses a Gaussian approximation to the Dirichlet-multinomial tails - it should be interpreted as a practical stopping statistic rather than a guaranteed probability bound.
- LazyGBM's near-lossless mode uses an auxiliary-model assumption; the threshold should be interpreted as an aggressiveness knob rather than a calibrated probability.

## License

This project is provided for research and educational purposes.

## Acknowledgments

This implementation uses:
- [scikit-learn](https://scikit-learn.org/) for base ensemble models
- [PyTorch](https://pytorch.org/) for neural network baselines
- [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/) for numerical computations
- Optional [Numba](https://numba.pydata.org/) for JIT acceleration
