#!/bin/bash
# run_rexgradient_fixed_final_pipeline.sh
#
# SCRIPT 1 of 2: VISION EXTRACTION (run on A100 system, 6 GPUs)
#
# Phase 1: Re-extract FINAL LAYER vision embeddings (16-bit fix) for all 8 models.
#          Fast — skips intermediate layer hooks, only saves final embedding.
#          Runs 3 splits per model in parallel on GPUs 0-2. One model at a time.
#
# Phase 2: Re-extract ALL LAYERS vision embeddings (16-bit fix) for all 8 models.
#          Full extraction with hooks on every transformer block.
#          Runs 3 vision splits in parallel on GPUs 0-2. One model at a time.
#          Text features are NOT re-extracted (they don't use images).
#
# Each model uses its own conda env. Already-completed extractions are skipped
# (checks labels_and_metadata.csv timestamp vs this script's changes).
#
# System requirements: 6x A100 GPUs, 28 CPU cores × 16GB each
#
# Usage:
#   bash run_rexgradient_fixed_final_pipeline.sh             # Full (final + all layers)
#   bash run_rexgradient_fixed_final_pipeline.sh final       # Only final layer
#   bash run_rexgradient_fixed_final_pipeline.sh alllayers   # Only all layers
#   bash run_rexgradient_fixed_final_pipeline.sh chexzero    # Single model only (both phases)

set -e

STEP="${1:-all}"

PROJECT_ROOT="/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol"
REXGRADIENT_DIR="$PROJECT_ROOT/rexgradient_dataset"
OTHER_MODELS="$PROJECT_ROOT/other_models"

MASTER_LOG_DIR="$REXGRADIENT_DIR/extraction_logs_fixed_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$MASTER_LOG_DIR"

# Clear parent CUDA restrictions
unset CUDA_VISIBLE_DEVICES

echo "======================================================================"
echo "    REXGRADIENT VISION RE-EXTRACTION (16-BIT FIX)"
echo "    Step: $STEP"
echo "    Logs: $MASTER_LOG_DIR"
echo "    Start: $(date)"
echo "======================================================================"

ALL_MODELS="chexzero biovilt radfm llavamed1p5 nv_reason chexagent medversa medgemma_1p5"

# ─────────────────────────────────────────────────────────────────────────────
# Model → conda env mapping
# ─────────────────────────────────────────────────────────────────────────────
get_conda_env() {
    case $1 in
        chexzero)     echo "embeddings" ;;
        biovilt)      echo "monai_cxr" ;;
        radfm)        echo "radfm_env" ;;
        llavamed1p5)  echo "llava-med" ;;
        nv_reason)    echo "nvreason_env" ;;
        chexagent)    echo "chexagent" ;;
        medversa)     echo "medversa_env" ;;
        medgemma_1p5) echo "medgemma1p5" ;;
    esac
}

get_img_script() {
    case $1 in
        chexzero)     echo "chexzero_extract_image_layers.py" ;;
        biovilt)      echo "biovilt_extract_image_layers.py" ;;
        radfm)        echo "radfm_extract_image_layers.py" ;;
        llavamed1p5)  echo "llavamed_extract_image_layers.py" ;;
        nv_reason)    echo "nvreason_extract_image_layers.py" ;;
        chexagent)    echo "chexagent_extract_image_layers.py" ;;
        medversa)     echo "medversa_extract_image_layers.py" ;;
        medgemma_1p5) echo "medgemma_extract_image_layers.py" ;;
    esac
}

get_gpu_method() {
    case $1 in
        llavamed1p5) echo "gpu_id" ;;
        *)           echo "cuda_vis" ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# GPU JOB QUEUE: 6 GPUs, jobs distributed at the (model, split) level.
# When a GPU finishes its job, it immediately picks up the next one.
# Uses atomic file renames as GPU locks.
# ─────────────────────────────────────────────────────────────────────────────
GPU_LOCK_DIR=""

init_gpu_pool() {
    GPU_LOCK_DIR=$(mktemp -d "${MASTER_LOG_DIR}/gpu_locks.XXXX")
    for gpu in 0 1 2 3 4 5; do
        touch "$GPU_LOCK_DIR/$gpu"
    done
}

acquire_gpu() {
    # Spin until a GPU becomes available (atomic mv)
    while true; do
        for gpu in 0 1 2 3 4 5; do
            if mv "$GPU_LOCK_DIR/$gpu" "$GPU_LOCK_DIR/${gpu}.locked" 2>/dev/null; then
                echo $gpu
                return
            fi
        done
        sleep 0.5
    done
}

release_gpu() {
    mv "$GPU_LOCK_DIR/${1}.locked" "$GPU_LOCK_DIR/$1"
}

# ─────────────────────────────────────────────────────────────────────────────
# Run a single extraction job: one model, one split, one GPU
# Args: model split gpu_id extra_args
# ─────────────────────────────────────────────────────────────────────────────
run_single_job() {
    local model=$1
    local split=$2
    local gpu_id=$3
    local extra_args="$4"
    
    local conda_env=$(get_conda_env $model)
    local img_script=$(get_img_script $model)
    local gpu_method=$(get_gpu_method $model)
    local model_dir="$OTHER_MODELS/$model"
    local model_python="/home/apalliko/.conda/envs/${conda_env}/bin/python"
    
    local split_name=""
    case $split in
        0) split_name="train" ;;
        1) split_name="val" ;;
        2) split_name="test" ;;
    esac
    
    local log_dir="$MASTER_LOG_DIR/$model"
    mkdir -p "$log_dir"
    local log_file="$log_dir/${model}_${split_name}${extra_args:+_final}.log"
    
    local cmd=""
    if [ "$gpu_method" = "gpu_id" ]; then
        cmd="$model_python -u $model_dir/$img_script --dataset_folder_name rexgradient --split_value $split --gpu_id $gpu_id $extra_args"
    else
        cmd="CUDA_VISIBLE_DEVICES=$gpu_id $model_python -u $model_dir/$img_script --dataset_folder_name rexgradient --split_value $split $extra_args"
    fi
    
    echo "  [GPU $gpu_id] START: ${model}/${split_name} $extra_args"
    if eval $cmd > "$log_file" 2>&1; then
        echo "  [GPU $gpu_id] OK:    ${model}/${split_name}"
    else
        echo "  [GPU $gpu_id] FAIL:  ${model}/${split_name} (see $log_file)"
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Run a batch of jobs through the 6-GPU pool
# Args: extra_args  "model:split" "model:split" ...
# Jobs are sorted: train splits first (biggest), so they start immediately
# and val/test fill in as GPUs free up.
# ─────────────────────────────────────────────────────────────────────────────
run_gpu_pool() {
    local extra_args="$1"
    shift
    local jobs=("$@")
    local n_jobs=${#jobs[@]}
    
    echo "  Queue: $n_jobs jobs → 6 GPUs"
    
    init_gpu_pool
    
    declare -a WORKER_PIDS=()
    
    for job in "${jobs[@]}"; do
        local model="${job%%:*}"
        local split="${job##*:}"
        
        # Acquire a free GPU (blocks until available)
        local gpu_id=$(acquire_gpu)
        
        # Launch job in background, release GPU when done
        (
            run_single_job "$model" "$split" "$gpu_id" "$extra_args"
            release_gpu "$gpu_id"
        ) &
        WORKER_PIDS+=($!)
        
        # Tiny delay to avoid thundering herd on lock dir
        sleep 0.1
    done
    
    # Wait for all jobs to complete
    local failed=0
    for pid in "${WORKER_PIDS[@]}"; do
        if ! wait $pid; then
            failed=$((failed+1))
        fi
    done
    
    # Cleanup
    rm -rf "$GPU_LOCK_DIR"
    
    echo "  Pool complete: $((n_jobs - failed))/$n_jobs succeeded"
}

# ─────────────────────────────────────────────────────────────────────────────
# Build job list: train splits first (biggest), then val, then test
# This ensures large jobs start immediately on GPUs 0-5 and small jobs
# fill gaps as GPUs free up.
# ─────────────────────────────────────────────────────────────────────────────
build_job_list() {
    local models="$1"
    local jobs=()
    
    # Train splits first (biggest — ~105K images each)
    for model in $models; do
        jobs+=("$model:0")
    done
    # Val splits next (~22K each)
    for model in $models; do
        jobs+=("$model:1")
    done
    # Test splits last (~22K each)
    for model in $models; do
        jobs+=("$model:2")
    done
    
    echo "${jobs[@]}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: Final layer only (fast, --final_only)
# ─────────────────────────────────────────────────────────────────────────────
run_final_layer() {
    local models="${1:-$ALL_MODELS}"
    echo ""
    echo "======================================================================"
    echo "  PHASE 1: FINAL LAYER EXTRACTION (--final_only)"
    echo "  Models: $models"
    echo "  Distribution: 24 jobs (8 models × 3 splits) → 6 GPU pool"
    echo "======================================================================"

    local jobs=($(build_job_list "$models"))
    run_gpu_pool "--final_only" "${jobs[@]}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: All layers (full extraction)
# ─────────────────────────────────────────────────────────────────────────────
run_all_layers() {
    local models="${1:-$ALL_MODELS}"
    echo ""
    echo "======================================================================"
    echo "  PHASE 2: ALL LAYERS EXTRACTION"
    echo "  Models: $models"
    echo "  Distribution: 24 jobs (8 models × 3 splits) → 6 GPU pool"
    echo "  NOTE: Text features NOT re-extracted (no 16-bit issue)"
    echo "======================================================================"

    local jobs=($(build_job_list "$models"))
    run_gpu_pool "--overwrite" "${jobs[@]}"
}

# ─────────────────────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────────────────────
case "$STEP" in
    final)
        run_final_layer "$ALL_MODELS"
        ;;
    alllayers)
        run_all_layers "$ALL_MODELS"
        ;;
    all)
        run_final_layer "$ALL_MODELS"
        run_all_layers "$ALL_MODELS"
        ;;
    *)
        # Treat as model name(s)
        run_final_layer "$STEP"
        run_all_layers "$STEP"
        ;;
esac

echo ""
echo "======================================================================"
echo "    EXTRACTION COMPLETE"
echo "    Logs: $MASTER_LOG_DIR"
echo "    End: $(date)"
echo "======================================================================"
