#!/bin/bash

# This script runs the vision-only feature extraction sequentially for all
# data splits (train, validation, and test).

echo "--- Starting Vision-Only Feature Extraction for All Splits ---"
echo ""

# Loop through the split values 0, 1, and 2
for split_num in 0 1 2
do
  echo "=========================================================="
  echo "--- Running vision feature extraction for split: $split_num ---"
  echo "=========================================================="
  
  # Call the new python script, passing the current split number
  python /home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/script4b_extract_image_layers.py --split_value $split_num
  
  echo ""
  echo "--- Finished split: $split_num ---"
  echo ""
done

echo "=========================================================="
echo "All splits have been processed."
echo "--- Vision Feature Extraction Complete ---"