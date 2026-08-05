#!/bin/bash
# run_rexgradient_nonpediatric_text_train.sh
#
# Train ALL-LAYER text probes (sex + age + No Finding) on nonpediatric data.
# Uses train_rexgradient_nonpediatric_all_layers.py with --modalities text.
#
# Current state of trained_probes_text_only_nonpediatric/ per model:
#   - Only final-layer sex + age probes exist (4 files each)
#   - No No_Finding probes at all
#   - No intermediate layer probes
#
# This script will:
#   - Train sex + age + No_Finding for ALL intermediate text layers
#   - Train No_Finding for the final layer (sex + age already exist, will be skipped)
#
# Skip logic: per-probe check (not per-directory). If probe_X_sex.joblib exists,
# sex training is skipped for that layer. Only skips a layer entirely if ALL 3 exist.
#
# System: GH200 node
# Environment: mega_med_env
# CPU: 8 layers parallel × n_jobs=5 × OMP=1 = 40 cores max
#
# Usage:
#   bash run_rexgradient_nonpediatric_text_train.sh
#   bash run_rexgradient_nonpediatric_text_train.sh chexzero biovilt nv_reason

set -e

# If args given, treat as model names; otherwise all 8
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

# 8 parallel layers × n_jobs=5 inside GridSearchCV × OMP=1 = 40 cores
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PYTHONWARNINGS="ignore::FutureWarning"

echo "======================================================================"
echo "    REXGRADIENT TEXT PROBE TRAINING (nonpediatric, all layers)"
echo "    Models: $MODELS"
echo "    Modalities: text"
echo "    Probes: sex + age (4-class) + No Finding"
echo "    Parallelism: 8 layers × n_jobs=5 × OMP=1 = 40 cores"
echo "    Output: trained_probes_text_only_nonpediatric/"
echo "    Start: $(date)"
echo "======================================================================"

for model in $MODELS; do
    test_meta="$OTHER_MODELS/$model/probe_experiment_outputs/rexgradient/features_text_only_train/labels_and_metadata.csv"
    if [ ! -f "$test_meta" ]; then
        echo "  [SKIP] $model — no text train metadata found"
        continue
    fi
    echo ""
    echo "  >>> Training text probes: $model"
    python -u "$REXGRADIENT_DIR/train_rexgradient_nonpediatric_all_layers.py" \
        --models $model \
        --modalities text \
        --n_jobs 5 \
        --n_parallel_layers 8
done

echo ""
echo "======================================================================"
echo "    TEXT PROBE TRAINING COMPLETE"
echo "    End: $(date)"
echo "======================================================================"
