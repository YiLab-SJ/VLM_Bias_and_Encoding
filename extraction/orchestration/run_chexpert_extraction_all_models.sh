#!/bin/bash
# run_chexpert_extraction_all_models.sh
#
# Master script to run CheXpert feature extraction across all 8 models.
# Optimized for 5 A100-80GB GPUs and 28 CPU cores (16 GB RAM each).
#
# Each model is run sequentially (different conda envs), but within each
# model, 5 parallel GPU jobs are launched (3 vision + 2 text), then the
# remaining text job runs on a freed GPU.
#
# Usage:
#   bash run_chexpert_extraction_all_models.sh                  # Run ALL models
#   bash run_chexpert_extraction_all_models.sh medversa          # Run single model
#   bash run_chexpert_extraction_all_models.sh medversa chexzero # Run specific models
#
# Hardware: 5x A100-80GB, 28 CPU cores, ~448 GB system RAM
# Each model gets exclusive access to all 5 GPUs while it runs.

set -e

DATASET="chexpert"
NUM_GPUS=5
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OTHER_MODELS_DIR="$SCRIPT_ROOT/other_models"
MASTER_LOG_DIR="$OTHER_MODELS_DIR/chexpert_extraction_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

if [ "$DATASET" != "chexpert" ]; then
    echo "ERROR: This script is hard-wired for CheXpert extraction only."
    exit 1
fi

# Default: all 8 models
ALL_MODELS="radfm llavamed1p5 nv_reason chexagent medversa chexzero"

if [ $# -gt 0 ]; then
    TARGET_MODELS="$@"
else
    TARGET_MODELS="$ALL_MODELS"
fi

echo "======================================================================"
echo "    CHEXPERT FEATURE EXTRACTION - ALL MODELS"
echo "    Dataset: $DATASET"
echo "    GPUs: $NUM_GPUS x A100-80GB"
echo "    Target models: $TARGET_MODELS"
echo "    Log dir: $MASTER_LOG_DIR"
echo "======================================================================"
echo "Start: $(date)"
echo ""

# Clear parent CUDA restrictions
unset CUDA_VISIBLE_DEVICES

# ============================================================================
# GENERIC EXTRACTION FUNCTION
#
# Runs 6 extraction jobs (3 vision splits + 3 text splits) across 5 GPUs:
#   Phase A (5 parallel): GPU 0-2 = vision train/val/test
#                          GPU 3-4 = text train/val
#   Phase B (1 remaining): GPU 0   = text test
#
# Arguments:
#   $1 = model directory name (under other_models/)
#   $2 = conda environment name
#   $3 = image extraction python script filename
#   $4 = text extraction python script filename
#   $5 = GPU assignment method: "cuda_vis" or "gpu_id"
#        "cuda_vis" = set CUDA_VISIBLE_DEVICES (most models)
#        "gpu_id"   = pass --gpu_id N argument (llavamed only)
# ============================================================================
extract_model() {
    local MODEL_DIR=$1
    local CONDA_ENV=$2
    local IMG_SCRIPT=$3
    local TXT_SCRIPT=$4
    local GPU_METHOD=${5:-cuda_vis}

    local MODEL_PATH="$OTHER_MODELS_DIR/$MODEL_DIR"
    local LOG_DIR="$MASTER_LOG_DIR/$MODEL_DIR"
    local FEATURES_ROOT="$MODEL_PATH/probe_experiment_outputs/$DATASET"
    local MIMIC_ROOT="$MODEL_PATH/probe_experiment_outputs/MIMIC-CXR-JPG"
    mkdir -p "$LOG_DIR"

    if [[ "$FEATURES_ROOT" != *"/probe_experiment_outputs/chexpert" ]]; then
        echo "ERROR: Refusing to write to unexpected CheXpert output root: $FEATURES_ROOT"
        return 1
    fi

    if [ -e "$FEATURES_ROOT" ]; then
        local FEATURES_REALPATH
        FEATURES_REALPATH=$(realpath "$FEATURES_ROOT") || return 1
        if [[ "$FEATURES_REALPATH" == *"/probe_experiment_outputs/MIMIC-CXR-JPG"* ]]; then
            echo "ERROR: CheXpert output root resolves into the MIMIC output tree: $FEATURES_REALPATH"
            return 1
        fi
    fi

    if [ -e "$MIMIC_ROOT" ]; then
        local MIMIC_REALPATH
        MIMIC_REALPATH=$(realpath "$MIMIC_ROOT") || return 1
        echo "    MIMIC outputs preserved at: $MIMIC_REALPATH"
    fi

    echo ""
    echo "============================================================"
    echo ">>> [$MODEL_DIR] Starting CheXpert extraction"
    echo "    Conda env: $CONDA_ENV"
    echo "    Scripts: $IMG_SCRIPT / $TXT_SCRIPT"
    echo "    GPU method: $GPU_METHOD"
    echo "    Output: $FEATURES_ROOT"
    echo "============================================================"

    (
        # Activate conda environment
        module load conda3 2>/dev/null || true
        source activate "$CONDA_ENV" 2>/dev/null || conda activate "$CONDA_ENV" 2>/dev/null || {
            echo "ERROR: Could not activate conda env '$CONDA_ENV'"
            exit 1
        }
        echo "Activated: $CONDA_ENV ($(which python))"
        cd "$MODEL_PATH"

        # Helper to build GPU-prefixed command
        gpu_cmd() {
            local GPU_ID=$1
            shift
            if [ "$GPU_METHOD" = "gpu_id" ]; then
                echo "$@ --gpu_id $GPU_ID"
            else
                echo "CUDA_VISIBLE_DEVICES=$GPU_ID $@"
            fi
        }

        # Track PIDs
        declare -a PIDS=()
        declare -a JOB_NAMES=()

        launch() {
            local GPU_ID=$1
            local JOB_NAME=$2
            local OUTPUT_DIR=$3
            shift 3
            local CMD="$@"

            if [[ "$OUTPUT_DIR" != "$FEATURES_ROOT"/* ]]; then
                echo "  [FAIL] Refusing to launch $JOB_NAME outside CheXpert root: $OUTPUT_DIR"
                return 1
            fi

            if [ -e "$OUTPUT_DIR" ]; then
                local OUTPUT_REALPATH
                OUTPUT_REALPATH=$(realpath "$OUTPUT_DIR") || return 1
                if [[ "$OUTPUT_REALPATH" == *"/probe_experiment_outputs/MIMIC-CXR-JPG"* ]]; then
                    echo "  [FAIL] Refusing to launch $JOB_NAME because output resolves into MIMIC tree: $OUTPUT_REALPATH"
                    return 1
                fi
            fi

            # Skip if output already exists
            if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A "$OUTPUT_DIR" 2>/dev/null)" ]; then
                echo "  [SKIP] $JOB_NAME — output already populated."
                return
            fi

            local FULL_CMD=$(gpu_cmd $GPU_ID $CMD)
            local LOG="$LOG_DIR/${JOB_NAME}.log"
            echo "  [GPU $GPU_ID] LAUNCH: $JOB_NAME"
            eval $FULL_CMD > "$LOG" 2>&1 &
            PIDS+=($!)
            JOB_NAMES+=("$JOB_NAME")
        }

        wait_all() {
            local FAIL=0
            for i in "${!PIDS[@]}"; do
                if ! wait ${PIDS[$i]}; then
                    echo "  [FAIL] ${JOB_NAMES[$i]} (PID ${PIDS[$i]}) failed. See: $LOG_DIR/${JOB_NAMES[$i]}.log"
                    FAIL=1
                else
                    echo "  [OK]   ${JOB_NAMES[$i]} completed."
                fi
            done
            PIDS=()
            JOB_NAMES=()
            return $FAIL
        }

        # =============================================================
        # Phase A: 5 parallel jobs on GPUs 0-4
        #   GPU 0: Vision TRAIN
        #   GPU 1: Vision VAL
        #   GPU 2: Vision TEST
        #   GPU 3: Text   TRAIN
        #   GPU 4: Text   VAL
        # =============================================================
        echo "  --- Phase A: 5 parallel jobs (3 vision + 2 text) ---"

        launch 0 "vision_train" "$FEATURES_ROOT/features_vision_only_train" \
            python "$IMG_SCRIPT" --dataset_folder_name "$DATASET" --split_value 0

        launch 1 "vision_val" "$FEATURES_ROOT/features_vision_only_val" \
            python "$IMG_SCRIPT" --dataset_folder_name "$DATASET" --split_value 1

        launch 2 "vision_test" "$FEATURES_ROOT/features_vision_only_test" \
            python "$IMG_SCRIPT" --dataset_folder_name "$DATASET" --split_value 2

        launch 3 "text_train" "$FEATURES_ROOT/features_text_only_train" \
            python "$TXT_SCRIPT" --dataset_folder_name "$DATASET" --split_value 0

        launch 4 "text_val" "$FEATURES_ROOT/features_text_only_val" \
            python "$TXT_SCRIPT" --dataset_folder_name "$DATASET" --split_value 1

        wait_all
        PHASE_A_STATUS=$?

        # =============================================================
        # Phase B: Remaining text test on GPU 0
        # =============================================================
        echo "  --- Phase B: Remaining text test ---"

        launch 0 "text_test" "$FEATURES_ROOT/features_text_only_test" \
            python "$TXT_SCRIPT" --dataset_folder_name "$DATASET" --split_value 2

        wait_all
        PHASE_B_STATUS=$?

        if [ $PHASE_A_STATUS -ne 0 ] || [ $PHASE_B_STATUS -ne 0 ]; then
            exit 1
        fi
    )
    return $?
}

# ============================================================================
# MODEL CONFIGURATIONS & EXECUTION
# ============================================================================
# Model configs: MODEL_DIR  CONDA_ENV  IMG_SCRIPT  TXT_SCRIPT  GPU_METHOD
# ============================================================================

FAILED=0
SUCCEEDED=0
SKIPPED=0

for MODEL in $TARGET_MODELS; do
    case $MODEL in
        medgemma_1p5)
            extract_model "medgemma_1p5" "medgemma1p5" \
                "medgemma_extract_image_layers.py" "medgemma_extract_text_layers.py" "cuda_vis"
            ;;
        biovilt)
            extract_model "biovilt" "monai_cxr" \
                "biovilt_extract_image_layers.py" "biovilt_extract_text_layers.py" "cuda_vis"
            ;;
        radfm)
            extract_model "radfm" "radfm_env" \
                "radfm_extract_image_layers.py" "radfm_extract_text_layers.py" "cuda_vis"
            ;;
        llavamed1p5)
            extract_model "llavamed1p5" "llava-med" \
                "llavamed_extract_image_layers.py" "llavamed_extract_text_layers.py" "gpu_id"
            ;;
        nv_reason)
            extract_model "nv_reason" "nvreason_env" \
                "nvreason_extract_image_layers.py" "nvreason_extract_text_layers.py" "cuda_vis"
            ;;
        chexagent)
            extract_model "chexagent" "chexagent" \
                "chexagent_extract_image_layers.py" "chexagent_extract_text_layers.py" "cuda_vis"
            ;;
        medversa)
            extract_model "medversa" "medversa_env" \
                "medversa_extract_image_layers.py" "medversa_extract_text_layers.py" "cuda_vis"
            ;;
        chexzero)
            extract_model "chexzero" "embeddings" \
                "chexzero_extract_image_layers.py" "chexzero_extract_text_layers.py" "cuda_vis"
            ;;
        *)
            echo "[SKIP] Unknown model: $MODEL"
            SKIPPED=$((SKIPPED + 1))
            continue
            ;;
    esac

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[OK]   $MODEL completed successfully."
        SUCCEEDED=$((SUCCEEDED + 1))
    else
        echo "[FAIL] $MODEL failed with exit code $EXIT_CODE."
        echo "       Check logs in: $MASTER_LOG_DIR/$MODEL/"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "======================================================================"
echo "    CHEXPERT EXTRACTION SUMMARY"
echo "    Succeeded: $SUCCEEDED"
echo "    Failed:    $FAILED"
echo "    Skipped:   $SKIPPED"
echo "    End: $(date)"
echo "======================================================================"

exit $FAILED
