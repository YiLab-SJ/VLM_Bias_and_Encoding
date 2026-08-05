#!/bin/bash

# run_chexzero_train_probes.sh
# Trains CheXzero demographic probes using the universal training script.
# Maintains a constant pool of parallel jobs per attribute.
#
# Usage: bash run_chexzero_train_probes.sh [DATASET] [MAX_PARALLEL] [MODALITY]
#   Default dataset: MIMIC-CXR-JPG
#   Default max parallel: 4
#   MODALITY: vision | text | both (default: both)

set -e

DATASET="${1:-MIMIC-CXR-JPG}"
MAX_PARALLEL="${2:-4}"
MODALITY="${3:-both}"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Universal training script
UNIVERSAL_DEMO_PY="/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models/all_models/universal_train_demographic_probe.py"

# Feature directories
FEATURES_ROOT="$SCRIPT_DIR/probe_experiment_outputs/$DATASET"

VISION_TRAIN_DIR="$FEATURES_ROOT/features_vision_only_train"
VISION_VAL_DIR="$FEATURES_ROOT/features_vision_only_val"
VISION_MODELS_DIR="$FEATURES_ROOT/trained_probes_vision_only"
VISION_RESULTS_DIR="$FEATURES_ROOT/results_vision_only_gridsearch"

TEXT_TRAIN_DIR="$FEATURES_ROOT/features_text_only_train"
TEXT_VAL_DIR="$FEATURES_ROOT/features_text_only_val"
TEXT_MODELS_DIR="$FEATURES_ROOT/trained_probes_text_only"
TEXT_RESULTS_DIR="$FEATURES_ROOT/results_text_only_gridsearch"

VISION_LAYERS=(
    "Vis_Block_1" "Vis_Block_2" "Vis_Block_3" "Vis_Block_4"
    "Vis_Block_5" "Vis_Block_6" "Vis_Block_7" "Vis_Block_8"
    "Vis_Block_9" "Vis_Block_10" "Vis_Block_11" "Vis_Block_12"
    "image_embedding_final"
)

TEXT_LAYERS=(
    "Txt_TokenEmbed"
    "Txt_Block_1" "Txt_Block_2" "Txt_Block_3" "Txt_Block_4"
    "Txt_Block_5" "Txt_Block_6" "Txt_Block_7" "Txt_Block_8"
    "Txt_Block_9" "Txt_Block_10" "Txt_Block_11" "Txt_Block_12"
    "text_embedding_final"
)

# Only sex, age, ethnicity (NO sex_ethnicity)
ATTRIBUTES=("sex" "age" "ethnicity")

LOG_DIR="$SCRIPT_DIR/parallel_logs_probes_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "======================================================================"
echo "  CheXzero Demographic Probe Training - Dataset: $DATASET"
echo "  Max parallel jobs: $MAX_PARALLEL  Modality: $MODALITY"
echo "  Using universal training script"
echo "  Logs: $LOG_DIR"
echo "======================================================================"
echo "Start: $(date)"
echo ""

if [[ "$MODALITY" == "vision" || "$MODALITY" == "both" ]]; then
# --- VISION PROBES ---
echo "============================================================"
echo ">>> PHASE 1: VISION Demographic Probes"
echo "============================================================"

for attribute in "${ATTRIBUTES[@]}"; do
    echo ""
    echo "--- VISION attribute: $attribute ---"
    for layer in "${VISION_LAYERS[@]}"; do
        if [[ $(jobs -p | wc -l) -ge $MAX_PARALLEL ]]; then
            wait -n
        fi
        LOG_FILE="$LOG_DIR/log_vision_${layer}_${attribute}.txt"
        python -u "$UNIVERSAL_DEMO_PY" \
            --train_dir "$VISION_TRAIN_DIR" \
            --val_dir "$VISION_VAL_DIR" \
            --models_out_dir "$VISION_MODELS_DIR" \
            --results_out_dir "$VISION_RESULTS_DIR" \
            --layer_name "$layer" \
            --attribute "$attribute" > "$LOG_FILE" 2>&1 &
        echo "  -> Launched: $layer / $attribute (PID: $!)"
    done
    echo "  Waiting for vision $attribute jobs..."
    wait
    echo "  --- Vision $attribute DONE ---"
done
fi  # end vision

if [[ "$MODALITY" == "text" || "$MODALITY" == "both" ]]; then
# --- TEXT PROBES ---
echo ""
echo "============================================================"
echo ">>> PHASE 2: TEXT Demographic Probes"
echo "============================================================"

for attribute in "${ATTRIBUTES[@]}"; do
    echo ""
    echo "--- TEXT attribute: $attribute ---"
    for layer in "${TEXT_LAYERS[@]}"; do
        if [[ $(jobs -p | wc -l) -ge $MAX_PARALLEL ]]; then
            wait -n
        fi
        LOG_FILE="$LOG_DIR/log_text_${layer}_${attribute}.txt"
        python -u "$UNIVERSAL_DEMO_PY" \
            --train_dir "$TEXT_TRAIN_DIR" \
            --val_dir "$TEXT_VAL_DIR" \
            --models_out_dir "$TEXT_MODELS_DIR" \
            --results_out_dir "$TEXT_RESULTS_DIR" \
            --layer_name "$layer" \
            --attribute "$attribute" > "$LOG_FILE" 2>&1 &
        echo "  -> Launched: $layer / $attribute (PID: $!)"
    done
    echo "  Waiting for text $attribute jobs..."
    wait
    echo "  --- Text $attribute DONE ---"
done
fi  # end text

echo ""
echo "======================================================================"
echo "  CheXzero Demographic Probe Training COMPLETE ($MODALITY)"
echo "======================================================================"
echo "End: $(date)"
