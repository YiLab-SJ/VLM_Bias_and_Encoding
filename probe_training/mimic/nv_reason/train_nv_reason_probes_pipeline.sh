#!/bin/bash
# train_nvreason_probes_pipeline.sh
# Master script to train all NV-Reason-CXR-3B probes using universal scripts.

set -e

DATASET="${1:-MIMIC-CXR-JPG}"
MAX_PARALLEL=11
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# --- Define Universal Script Paths ---
UNIVERSAL_DIR="/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models/all_models"
DEMO_PY="$UNIVERSAL_DIR/universal_train_demographic_probe.py"
DISEASE_PY="$UNIVERSAL_DIR/universal_train_disease_probe.py"

# --- NV-Reason specific layers ---
# Vision Encoder: 32 blocks (0 to 31)
VISION_LAYERS=()
for i in {0..31}; do VISION_LAYERS+=("Vis_Block_$i"); done

# Text Decoder: Embed + 36 blocks (0 to 35) + FinalNorm
TEXT_LAYERS=("Txt_Embed")
for i in {0..35}; do TEXT_LAYERS+=("Txt_Block_$i"); done
TEXT_LAYERS+=("Txt_FinalNorm")

ATTRIBUTES=("sex" "age" "ethnicity")

# --- Directory Setup ---
FEATURES_ROOT="$SCRIPT_DIR/probe_experiment_outputs/$DATASET"
LOG_DIR="$SCRIPT_DIR/parallel_logs_training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "  NV-Reason Probe Training Pipeline (Universal) - Dataset: $DATASET"
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

# --- Execute ---
# run_demographic_jobs "vision" VISION_LAYERS[@]
# run_demographic_jobs "text" TEXT_LAYERS[@]

run_disease_jobs "vision" VISION_LAYERS[@]
run_disease_jobs "text" TEXT_LAYERS[@]

echo ""
echo "======================================================================"
echo "  All NV-Reason Training COMPLETE"
echo "======================================================================"