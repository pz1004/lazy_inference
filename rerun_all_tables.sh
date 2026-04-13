#!/usr/bin/env bash
set -euo pipefail

export MPLBACKEND=Agg

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$SCRIPT_DIR"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT="${ROOT:-results/full_tables_runs30_${STAMP}}"
LOG="${LOG:-logs/full_tables_runs30_${STAMP}}"

mkdir -p "$ROOT" "$LOG"

TABLE_FIGURES_SCRIPT=""
for candidate in \
  "$REPO/table_figures.py" \
  "$REPO/results/table_figures.py" \
  "$REPO/results_old/table_figures.py"
do
  if [[ -f "$candidate" ]]; then
    TABLE_FIGURES_SCRIPT="$candidate"
    break
  fi
done

if [[ -z "$TABLE_FIGURES_SCRIPT" ]]; then
  echo "[ERROR] Could not find table_figures.py in expected locations." >&2
  echo "[ERROR] Checked: $REPO/results/table_figures.py and $REPO/results_old/table_figures.py" >&2
  exit 1
fi

run_chunked () {
  local name="$1"
  shift

  for start in 42 52 62; do
    echo "[START] ${name} seed chunk ${start}..$((start+9))"
    RUN_EXPERIMENTS_LOG="$LOG/${name}_${start}.log" \
      python run_multi_experiments.py \
        --runs 10 \
        --random-state "$start" \
        "$@" \
        --result-dir "$ROOT/${name}_chunk_${start}" \
        --no-display \
        > "$LOG/${name}_${start}.out" 2>&1
    echo "[DONE] ${name} seed chunk ${start}..$((start+9))"
  done

  NAME="$name" ROOT="$ROOT" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

root = Path(os.environ["ROOT"])
name = os.environ["NAME"]
frames = []

for start in (42, 52, 62):
    p = root / f"{name}_chunk_{start}" / "raw_results.csv"
    df = pd.read_csv(p)
    df["seed"] = start + df["run"].astype(int)
    df["run"] = df["seed"] - 42
    frames.append(df)

out = pd.concat(frames, ignore_index=True)
runs = sorted(out["run"].unique().tolist())
if runs != list(range(30)):
    raise SystemExit(f"{name}: expected run=0..29, got {runs}")

out_dir = root / name
out_dir.mkdir(parents=True, exist_ok=True)
out.to_csv(out_dir / "raw_results.csv", index=False)
out.to_json(out_dir / "raw_results.json", orient="records", lines=True)
print(f"[DONE] {name}: {len(out)} rows -> {out_dir / 'raw_results.csv'}")
PY
}

COMMON_RF_GBM_DEFAULTS=(
  --early-exit-epochs 5
  --lazy-rf-threshold 0.95
  --lazy-rf-min-trees 10
  --lazy-rf-block-size 10
  --lazy-gbm-threshold 3.0
  --lazy-gbm-min-trees 10
  --lazy-gbm-block-size 1
  --lazy-gbm-variant current
  --lazy-gbm-ratio-threshold 1.5
  --lazy-gbm-flip-scale 2.0
  --lazy-gbm-late-margin-fraction 0.8
)

run_chunked main \
  --datasets mnist covertype higgs credit \
  --methods full_rf fixed_cascade cascade lazy_rf quickscorer branchynet full_gbm lazy_gbm \
  "${COMMON_RF_GBM_DEFAULTS[@]}"

run_chunked sweep \
  --datasets mnist covertype higgs credit \
  --methods lazy_rf lazy_gbm \
  --lazy-rf-thresholds 0.90 0.95 0.97 0.99 \
  --lazy-gbm-thresholds 2.0 3.0 4.0 \
  "${COMMON_RF_GBM_DEFAULTS[@]}"

for B in 1 5 10 20; do
  run_chunked "rf_block_B${B}" \
    --datasets covertype higgs \
    --methods full_rf lazy_rf \
    --lazy-rf-threshold 0.95 \
    --lazy-rf-min-trees 10 \
    --lazy-rf-block-size "$B"
done

for M in 0 5 20; do
  run_chunked "rf_tmin_M${M}" \
    --datasets covertype higgs \
    --methods full_rf lazy_rf \
    --lazy-rf-threshold 0.95 \
    --lazy-rf-min-trees "$M" \
    --lazy-rf-block-size 10
done

for variant in certificate_only certificate_plus_flip current; do
  run_chunked "gbm_variant_${variant}" \
    --datasets covertype higgs \
    --methods full_gbm lazy_gbm \
    --lazy-gbm-threshold 3.0 \
    --lazy-gbm-min-trees 10 \
    --lazy-gbm-block-size 1 \
    --lazy-gbm-variant "$variant" \
    --lazy-gbm-ratio-threshold 1.5 \
    --lazy-gbm-flip-scale 2.0 \
    --lazy-gbm-late-margin-fraction 0.8
done

for M in 0 20; do
  run_chunked "gbm_tmin_M${M}" \
    --datasets covertype higgs \
    --methods full_gbm lazy_gbm \
    --lazy-gbm-threshold 3.0 \
    --lazy-gbm-min-trees "$M" \
    --lazy-gbm-block-size 1 \
    --lazy-gbm-variant current \
    --lazy-gbm-ratio-threshold 1.5 \
    --lazy-gbm-flip-scale 2.0 \
    --lazy-gbm-late-margin-fraction 0.8
done

mkdir -p "$ROOT/calibration" "$ROOT/figures"
for ds in mnist covertype higgs credit; do
  echo "[START] calibration ${ds}"
  python analysis/calibrate_lazyrf.py \
    --dataset "$ds" \
    --runs 30 \
    --start-seed 42 \
    --threshold 0.95 \
    --min-trees 10 \
    --calibration-samples 20000 \
    --output-dir "$ROOT/calibration" \
    --figures-dir "$ROOT/figures" \
    > "$LOG/calibration_${ds}.out" 2>&1
done

mkdir -p "$ROOT/table_figures_input"
cp "$ROOT/main/raw_results.csv" "$ROOT/table_figures_input/raw_results_perf.csv"
cp "$ROOT/sweep/raw_results.csv" "$ROOT/table_figures_input/raw_results_sweep.csv"
(
  cd "$ROOT/table_figures_input"
  python "$TABLE_FIGURES_SCRIPT"
)

python analysis/build_revision_tables.py \
  --main-csv "$ROOT/main/raw_results.csv" \
  --rf-ablation-glob "$ROOT/rf_*/raw_results.csv" \
  --gbm-ablation-glob "$ROOT/gbm_*/raw_results.csv" \
  --calibration-glob "$ROOT/calibration/lazyrf_calibration_summary_runs_*.csv" \
  --tables-dir "$ROOT/revision_tables"

ROOT="$ROOT" python - <<'PY'
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

root = Path(os.environ["ROOT"])
df = pd.read_csv(root / "main" / "raw_results.csv")
credit = df[df["dataset"].eq("Credit Card") & df["auroc"].notna() & df["auprc"].notna()].copy()

order = [
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

rows = []
for method in order:
    g = credit[credit["method"].eq(method)]
    if g.empty:
        continue
    n = len(g)
    crit = t.ppf(0.975, n - 1) if n > 1 else 0.0
    rows.append({
        "Method": method,
        "AUROC (mean ± 95% CI)": f"{g['auroc'].mean():.4f} $\\pm$ {crit*g['auroc'].std(ddof=1)/np.sqrt(n):.4f}",
        "AUPRC (mean ± 95% CI)": f"{g['auprc'].mean():.4f} $\\pm$ {crit*g['auprc'].std(ddof=1)/np.sqrt(n):.4f}",
    })

out = pd.DataFrame(rows)
out_dir = root / "credit_auc_table"
out_dir.mkdir(parents=True, exist_ok=True)
out.to_csv(out_dir / "credit_card_auc.csv", index=False)
tex = out.to_latex(index=False, escape=False, caption="Credit Card Fraud score-metric audit over 30 seeds.", label="tab:credit_card_auc")
(out_dir / "credit_card_auc.tex").write_text(tex)
print(f"[DONE] Credit Card AUROC/AUPRC table -> {out_dir}")
PY

echo "[DONE] All regenerated outputs are under: $ROOT"
echo "[DONE] Logs are under: $LOG"
