#!/bin/bash
# train_medversa_probes_pipeline.sh
# Master script to train VISION probes for MedVersa using universal scripts.
# Note: Text probes are disabled as only vision embeddings are currently available.

set -e

DATASET="${1:-MIMIC-CXR-JPG}"
MAX_PARALLEL=14
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- Define Universal Script Paths ---
UNIVERSAL_DIR="/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models/all_models"
DEMO_PY="$UNIVERSAL_DIR/universal_train_demographic_probe.py"
DISEASE_PY="$UNIVERSAL_DIR/universal_train_disease_probe.py"

# --- MedVersa specific VISION layers ---
# Swin-Base architecture: depths=[2, 2, 18, 2] + Embed + FinalNorm + MedVersa Projections
VISION_LAYERS=("Vis_Embed")

# Stage 0 (2 blocks)
for i in {1..2}; do VISION_LAYERS+=("Vis_S0_B$i"); done
# Stage 1 (2 blocks)
for i in {1..2}; do VISION_LAYERS+=("Vis_S1_B$i"); done
# Stage 2 (18 blocks)
for i in {1..18}; do VISION_LAYERS+=("Vis_S2_B$i"); done
# Stage 3 (2 blocks)
for i in {1..2}; do VISION_LAYERS+=("Vis_S3_B$i"); done

# Final layers
VISION_LAYERS+=("Vis_FinalNorm" "Vis_LNVision" "Vis_Projected")

ATTRIBUTES=("sex" "age" "ethnicity")

# --- Directory Setup ---
FEATURES_ROOT="$SCRIPT_DIR/probe_experiment_outputs/$DATASET"
LOG_DIR="$SCRIPT_DIR/parallel_logs_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "  MedVersa VISION Probe Training Pipeline - Dataset: $DATASET"
echo "  Max parallel jobs: $MAX_PARALLEL (Each using 2 cores = 28 cores total)"
echo "  Logs: $LOG_DIR"
echo "======================================================================"
echo ""

# Helper function to run the universal demographic script
run_demographic_jobs() {
    local MODALITY=$1 # "vision" or "text"
    local LAYERS=("${!2}")
    
    local TRAIN_DIR="$FEATURES_ROOT/features_${MODALITY}_only_train"
    local VAL_DIR="$FEATURES_ROOT/features_${MODALITY}_only_val"
    local MODELS_OUT="$FEATURES_ROOT/trained_probes_${MODALITY}_only"
    local RESULTS_OUT="$FEATURES_ROOT/results_${MODALITY}_only_gridsearch"

    echo ">>> Starting $MODALITY Demographic Probes"
    for attribute in "${ATTRIBUTES[@]}"; do
        for layer in "${LAYERS[@]}"; do
            if [[ $(jobs -p | wc -l) -ge $MAX_PARALLEL ]]; then
                wait -n
            fi
            LOG_FILE="$LOG_DIR/log_${MODALITY}_demo_${layer}_${attribute}.txt"
            # -u forces real-time unbuffered logging
            python -u "$DEMO_PY" \
                --train_dir "$TRAIN_DIR" \
                --val_dir "$VAL_DIR" \
                --models_out_dir "$MODELS_OUT" \
                --results_out_dir "$RESULTS_OUT" \
                --layer_name "$layer" \
                --attribute "$attribute" > "$LOG_FILE" 2>&1 &
            echo "  -> Launched: Demographic $MODALITY | $layer | $attribute"
        done
    done
    wait
    echo "  --- $MODALITY Demographic Probes DONE ---"
}

# Helper function to run the universal disease script
run_disease_jobs() {
    local MODALITY=$1 # "vision" or "text"
    local LAYERS=("${!2}")
    
    local TRAIN_DIR="$FEATURES_ROOT/features_${MODALITY}_only_train"
    local VAL_DIR="$FEATURES_ROOT/features_${MODALITY}_only_val"
    local MODELS_OUT="$FEATURES_ROOT/trained_probes_${MODALITY}_diseases"
    local RESULTS_OUT="$FEATURES_ROOT/results_${MODALITY}_only_disease_gridsearch"

    echo ">>> Starting $MODALITY Disease Probes"
    for layer in "${LAYERS[@]}"; do
        if [[ $(jobs -p | wc -l) -ge $MAX_PARALLEL ]]; then
            wait -n
        fi
        LOG_FILE="$LOG_DIR/log_${MODALITY}_disease_${layer}.txt"
        # -u forces real-time unbuffered logging
        python -u "$DISEASE_PY" \
            --train_dir "$TRAIN_DIR" \
            --val_dir "$VAL_DIR" \
            --models_out_dir "$MODELS_OUT" \
            --results_out_dir "$RESULTS_OUT" \
            --layer_name "$layer" > "$LOG_FILE" 2>&1 &
        echo "  -> Launched: Disease $MODALITY | $layer"
    done
    wait
    echo "  --- $MODALITY Disease Probes DONE ---"
}

# --- Execute VISION Only ---
run_demographic_jobs "vision" VISION_LAYERS[@]
run_disease_jobs "vision" VISION_LAYERS[@]

echo ""
echo "======================================================================"
echo "  All MedVersa Vision Training COMPLETE"
echo "======================================================================"