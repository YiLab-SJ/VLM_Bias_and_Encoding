#!/bin/bash

# This script runs text-only layer extraction for MIMIC-CXR-JPG for all splits (0, 1, 2) in parallel.
# It will exit immediately if any of the extraction jobs fail.

echo "--- Starting TEXT-ONLY Layer-wise Feature Extraction for MIMIC-CXR-JPG (All Splits in Parallel) ---"
echo "Current Date and Time: $(date)"
echo ""

# --- Configuration ---
DATASET_FOLDER_NAME="MIMIC-CXR-JPG"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/script_4a_extract_layerwise_features_text_only.py"

# --- Launch Jobs in Parallel ---
# Launch job for TRAIN split (0) in the background
echo "==> Launching extraction for $DATASET_FOLDER_NAME - Split 0 (TRAIN)..."
python "$PYTHON_SCRIPT_PATH" --dataset_folder_name "$DATASET_FOLDER_NAME" --split_value 0 &
PID_TRAIN=$! # Store the Process ID

# Launch job for VALIDATION split (1) in the background
echo "==> Launching extraction for $DATASET_FOLDER_NAME - Split 1 (VALIDATION)..."
python "$PYTHON_SCRIPT_PATH" --dataset_folder_name "$DATASET_FOLDER_NAME" --split_value 1 &
PID_VAL=$! # Store the Process ID

# Launch job for TEST split (2) in the background
echo "==> Launching extraction for $DATASET_FOLDER_NAME - Split 2 (TEST)..."
python "$PYTHON_SCRIPT_PATH" --dataset_folder_name "$DATASET_FOLDER_NAME" --split_value 2 &
PID_TEST=$! # Store the Process ID

echo ""
echo "All jobs launched in the background. Waiting for completion..."
echo "PIDs: Train=$PID_TRAIN, Validation=$PID_VAL, Test=$PID_TEST"
echo ""

# --- Wait for Each Job and Check Its Exit Status ---

# Wait for the TRAIN job
wait $PID_TRAIN
EXIT_STATUS_TRAIN=$?
if [ $EXIT_STATUS_TRAIN -ne 0 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: TRAIN job (PID $PID_TRAIN) failed with exit status $EXIT_STATUS_TRAIN. Stopping all."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    # Optional: kill other jobs if one fails
    kill $PID_VAL $PID_TEST 2>/dev/null
    exit 1
else
    echo "Job for TRAIN split completed successfully."
fi

# Wait for the VALIDATION job
wait $PID_VAL
EXIT_STATUS_VAL=$?
if [ $EXIT_STATUS_VAL -ne 0 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: VALIDATION job (PID $PID_VAL) failed with exit status $EXIT_STATUS_VAL. Stopping all."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    kill $PID_TEST 2>/dev/null
    exit 1
else
    echo "Job for VALIDATION split completed successfully."
fi

# Wait for the TEST job
wait $PID_TEST
EXIT_STATUS_TEST=$?
if [ $EXIT_STATUS_TEST -ne 0 ]; then
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "ERROR: TEST job (PID $PID_TEST) failed with exit status $EXIT_STATUS_TEST. Stopping all."
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit 1
else
    echo "Job for TEST split completed successfully."
fi

echo ""
echo "--- All TEXT-ONLY Layer Extraction tasks for MIMIC-CXR-JPG finished successfully. ---"
echo "Completion Date and Time: $(date)"