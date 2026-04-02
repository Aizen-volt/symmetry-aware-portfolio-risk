#!/usr/bin/env bash
set -euo pipefail

# ═══════════════════════════════════════════════════════════════════
# Master script: run all experiments for the revised paper
# ═══════════════════════════════════════════════════════════════════
#
# Prerequisites:
#   - Python 3.10+ with packages from requirements.txt
#   - GPU recommended (CPU works but ~10x slower)
#   - Fama-French data in data/raw/
#
# Total estimated time:
#   - GPU (RTX 3060+): ~4-6 hours
#   - CPU only: ~24-48 hours
#
# ═══════════════════════════════════════════════════════════════════

SEEDS="42 123 456 789 1024"
DATASETS="industry12 industry49"
PORTFOLIOS="equal_weight vol_scaled"
ALPHAS="0.05 0.01"
MODELS="flatten_mlp set_attention"
N_PERM=100

echo "=========================================="
echo "Step 1: Install dependencies"
echo "=========================================="
pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Step 2: Download Fama-French data"
echo "=========================================="
python -m src.data.download_ff_data

echo ""
echo "=========================================="
echo "Step 3: Run neural model experiments (5 seeds × 2 datasets × 2 portfolios × 2 alphas × 2 models = 80 runs)"
echo "=========================================="

TOTAL=0
for dataset in $DATASETS; do
for portfolio in $PORTFOLIOS; do
for alpha in $ALPHAS; do
for model in $MODELS; do
for seed in $SEEDS; do
    TOTAL=$((TOTAL + 1))
    RUN_NAME="${dataset}__${portfolio}__${model}__alpha_${alpha}__seed_${seed}"

    if [ -f "artifacts/${RUN_NAME}/metrics.json" ]; then
        echo "[${TOTAL}/80] SKIP (exists): ${RUN_NAME}"
        continue
    fi

    echo ""
    echo "[${TOTAL}/80] TRAINING: ${RUN_NAME}"
    python -m src.experiments.run_experiment \
        --dataset "$dataset" \
        --portfolio "$portfolio" \
        --model "$model" \
        --alpha "$alpha" \
        --seed "$seed" \
        --max_epochs 50

    CKPT="artifacts/${RUN_NAME}/model.pt"
    if [ -f "$CKPT" ]; then
        echo "[${TOTAL}/80] PERMUTATION TEST: ${RUN_NAME}"
        python -m src.experiments.run_permutation_test \
            --dataset "$dataset" \
            --portfolio "$portfolio" \
            --model "$model" \
            --checkpoint "$CKPT" \
            --n_perm $N_PERM \
            --seed "$seed"
    fi

done
done
done
done
done

echo ""
echo "=========================================="
echo "Step 4: Run Historical Simulation baselines"
echo "=========================================="
python -m src.experiments.run_historical_simulation

echo ""
echo "=========================================="
echo "Step 5: Aggregate results"
echo "=========================================="
python -m src.experiments.aggregate_results

echo ""
echo "=========================================="
echo "Step 6: Generate figures"
echo "=========================================="
python -m src.experiments.plot_results

echo ""
echo "=========================================="
echo "DONE. Outputs:"
echo "  artifacts/multi_seed_summary.csv"
echo "  artifacts/table_forecasting.tex"
echo "  artifacts/table_permutation.tex"
echo "  paper/figures/*.pdf"
echo "=========================================="
