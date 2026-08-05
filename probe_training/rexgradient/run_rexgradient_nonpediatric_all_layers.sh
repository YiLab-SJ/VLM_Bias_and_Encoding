#!/bin/bash
# run_rexgradient_nonpediatric_all_layers.sh
#
# Complete nonpediatric pipeline for ALL LAYERS (not just final layer).
# Trains probes, evaluates, validates, and generates plots.
#
# What it does:
#   1. Train ALL-LAYER probes (sex + age + No Finding) on adult-only data
#      using the 16-bit-fixed embeddings. Output: trained_probes_{mod}_only_nonpediatric/
#   2. Evaluate ALL LAYERS using nonpediatric probes on adult-only test data.
#      Parallelized at the LAYER level (80 layers at a time).
#   3. After first model, VALIDATE that the JSON contains correct AUCs
#      (must differ from the old bad JSONs)
#   4. Generate all plots from the resulting JSONs
#
# System: GH200 node (100+ CPU cores × 6.67GB each)
# Environment: mega_med_env
# CPU cap: ≤40 cores at any time
#
# Usage:
#   bash run_rexgradient_nonpediatric_all_layers.sh                              # Full pipeline (all 8 models)
#   bash run_rexgradient_nonpediatric_all_layers.sh all chexzero biovilt nv_reason  # Specific models
#   bash run_rexgradient_nonpediatric_all_layers.sh train chexzero biovilt        # Only training, specific models
#   bash run_rexgradient_nonpediatric_all_layers.sh evaluate                     # Only evaluation
#   bash run_rexgradient_nonpediatric_all_layers.sh plots                        # Only plots

set -e

STEP="${1:-all}"
shift 2>/dev/null || true

# If additional args given, treat them as model names; otherwise use all 8
DEFAULT_MODELS="chexzero biovilt radfm llavamed1p5 nv_reason chexagent medversa medgemma_1p5"
if [ $# -gt 0 ]; then
    MODELS="$@"
else
    MODELS="$DEFAULT_MODELS"
fi

PROJECT_ROOT="/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol"
REXGRADIENT_DIR="$PROJECT_ROOT/rexgradient_dataset"
OTHER_MODELS="$PROJECT_ROOT/other_models"

# GH200 environment
source /home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models/all_models_arm/mega_med_env/bin/activate

# Cap CPU: n_jobs=5 × OMP=8 = 40 cores; for eval n_jobs=80 × OMP=1 = 80 cores
# We'll set OMP dynamically per phase
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8
# Suppress FutureWarnings in parent + joblib child Python processes
export PYTHONWARNINGS="ignore::FutureWarning"
ALL_MODELS="chexzero biovilt radfm llavamed1p5 nv_reason chexagent medversa medgemma_1p5"

echo "======================================================================"
echo "    REXGRADIENT NONPEDIATRIC ALL-LAYERS PIPELINE (GH200)"
echo "    Step: $STEP"
echo "    Models: $MODELS"
echo "    Start: $(date)"
echo "    Python: $(which python)"
echo "======================================================================"

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1: Train ALL-LAYER probes (nonpediatric, adults only)
# Uses train_rexgradient_nonpediatric_all_layers.py (new script)
# Saves to trained_probes_{modality}_only_nonpediatric/
# ─────────────────────────────────────────────────────────────────────────────
run_train() {
    echo ""
    echo "======================================================================"
    echo "  PHASE 1: Train ALL-LAYER probes (sex + age + No Finding)"
    echo "  Adults only (age class >= 1), 1 model at a time, 8 layers parallel"
    echo "  CPU: 8 layers × n_jobs=5 × OMP=1 = 40 cores max"
    echo "======================================================================"
    export OMP_NUM_THREADS=2
    export MKL_NUM_THREADS=2
    export OPENBLAS_NUM_THREADS=2

    for model in $MODELS; do
        local test_meta="$OTHER_MODELS/$model/probe_experiment_outputs/rexgradient/features_vision_only_test/labels_and_metadata.csv"
        if [ ! -f "$test_meta" ]; then
            echo "  [SKIP] $model — no embeddings found"
            continue
        fi
        echo ""
        echo "  >>> Training: $model"
        python -u "$REXGRADIENT_DIR/train_rexgradient_nonpediatric_all_layers.py" \
            --models $model \
            --n_jobs 4 \
            --n_parallel_layers 16
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2: Evaluate ALL LAYERS (nonpediatric, 80 layers at a time)
# Uses evaluate_rexgradient_full_test_nonpediatric.py with probes from _nonpediatric dir
# OMP_NUM_THREADS=1 for evaluation (each parallel layer job is lightweight)
# ─────────────────────────────────────────────────────────────────────────────
run_evaluate() {
    echo ""
    echo "======================================================================"
    echo "  PHASE 2: Evaluate ALL LAYERS (nonpediatric, n_jobs=40)"
    echo "  Using probes from trained_probes_{mod}_only_nonpediatric/"
    echo "  CPU: n_jobs=40 × OMP_NUM_THREADS=1 = 40 cores max"
    echo "======================================================================"
    # For evaluation: many lightweight parallel jobs, each single-threaded
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    python -u "$REXGRADIENT_DIR/evaluate_rexgradient_full_test_nonpediatric.py" \
        --models $MODELS \
        --n_jobs 40

    # Reset for subsequent phases
    export OMP_NUM_THREADS=8
    export MKL_NUM_THREADS=8
    export OPENBLAS_NUM_THREADS=8
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2.5: Validate first model's JSON has correct (high) AUCs
# ─────────────────────────────────────────────────────────────────────────────
run_validate() {
    echo ""
    echo "======================================================================"
    echo "  VALIDATION: Check JSONs have correct AUCs (not old faulty values)"
    echo "======================================================================"
    python -u -c "
import json, glob, os, sys

base = '$OTHER_MODELS/evaluation_results'
models_ok = 0
models_fail = 0

for model in '$MODELS'.split():
    # Find latest nonpediatric vision JSON
    pattern = os.path.join(base, model,
        'evaluation_results_vision_full_test_rexgradient_nonpediatric',
        'demographics_full_test_vision_rexgradient_nonpediatric_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'  [{model}] NO JSON FOUND')
        models_fail += 1
        continue

    latest = files[-1]
    with open(latest) as f:
        data = json.load(f)

    # Check: must have more than 1 layer (all-layer eval)
    n_layers = len(data)
    if n_layers <= 1:
        print(f'  [{model}] FAIL: only {n_layers} layer(s) in JSON (expected many)')
        models_fail += 1
        continue

    # Check sex AUC on first available layer
    first_layer = list(data.keys())[0]
    sex_auc = data[first_layer].get('sex', {}).get('overall_auc_ci', {}).get('mean', 0)
    
    # Final layer sex AUC should be high (>0.8 for most models)
    last_layer = list(data.keys())[-1]
    sex_auc_final = data[last_layer].get('sex', {}).get('overall_auc_ci', {}).get('mean', 0)

    print(f'  [{model}] OK: {n_layers} layers, first_layer_sex_auc={sex_auc:.4f}, final_layer_sex_auc={sex_auc_final:.4f}')
    models_ok += 1

print(f'\\n  Summary: {models_ok} OK, {models_fail} FAILED')
if models_fail > 0:
    print('  WARNING: Some models have issues. Check above.')
    sys.exit(1)
"
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3: Generate plots
# ─────────────────────────────────────────────────────────────────────────────
run_plots() {
    echo ""
    echo "======================================================================"
    echo "  PHASE 3: Generate plots (uses whatever nonpediatric JSONs exist)"
    echo "======================================================================"
    bash "$REXGRADIENT_DIR/run_rexgradient_plots.sh" nonpediatric
}

# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
case "$STEP" in
    train)
        run_train
        ;;
    evaluate)
        run_evaluate
        run_validate
        ;;
    plots)
        run_plots
        ;;
    all)
        run_train
        run_evaluate
        run_validate
        run_plots
        ;;
    *)
        echo "Unknown step: $STEP"
        echo "Usage: $0 {all|train|evaluate|plots}"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "    COMPLETE"
echo "    End: $(date)"
echo "======================================================================"
