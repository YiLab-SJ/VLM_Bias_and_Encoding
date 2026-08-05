#!/usr/bin/env python3
"""
build_lmm_csvs_from_jsons.py

Converts evaluation JSON files into the CSV format required by
linear_mixed_model_analysis.py for MIMIC-CXR, CheXpert, and RexGradient.

For each dataset and model, produces two CSVs:
  - demographic_{model}.csv: columns [layer, level, metric, attribute, value]
  - disease_conditioned_{model}.csv: columns [layer, disease, condition, fpr]

These CSVs are consumed by the LMM analysis script.

Usage:
    python build_lmm_csvs_from_jsons.py
    python build_lmm_csvs_from_jsons.py --dataset mimic
    python build_lmm_csvs_from_jsons.py --dataset chexpert
    python build_lmm_csvs_from_jsons.py --dataset rexgradient
"""

import os
import sys
import json
import glob
import argparse
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models"

# Where evaluation JSONs live per dataset:
# MIMIC-CXR: evaluation_results_optimized_thresholds/{model}/
# CheXpert:  evaluation_results/{model}/
# RexGradient: evaluation_results/{model}/
EVAL_PATHS = {
    "mimic": os.path.join(BASE_DIR, "evaluation_results_optimized_thresholds"),
    "chexpert": os.path.join(BASE_DIR, "evaluation_results"),
    "rexgradient": os.path.join(BASE_DIR, "evaluation_results"),
    "rexgradient_nonpediatric": os.path.join(BASE_DIR, "evaluation_results"),
}

# Output directory for generated CSVs
OUTPUT_BASE = os.path.join(BASE_DIR, "all_models", "lmm_input_csvs")

MODELS = ["medgemma_1p5", "biovilt", "radfm", "llavamed1p5", "nv_reason", "chexagent", "medversa", "chexzero"]

# JSON subfolder naming per dataset:
# MIMIC demographics:  evaluation_results_vision_full_test/demographics_full_test_vision_MIMIC-CXR-JPG_*.json
# MIMIC disease (NF):  evaluation_results_vision_disease_full_test/diseases_full_test_vision_MIMIC-CXR-JPG_*.json
# CheXpert demographics: evaluation_results_vision_full_test_chexpert/demographics_full_test_vision_chexpert_*.json
# CheXpert NF:           evaluation_results_vision_nofinding_full_test_chexpert/nofinding_full_test_vision_chexpert_*.json
# RexGradient demographics: evaluation_results_vision_full_test_rexgradient/demographics_full_test_vision_rexgradient_*.json
# RexGradient NF:           evaluation_results_vision_nofinding_full_test_rexgradient/nofinding_full_test_vision_rexgradient_*.json

DATASET_CONFIG = {
    "mimic": {
        "demo_folder": "evaluation_results_vision_full_test",
        "demo_prefix": "demographics_full_test_vision_MIMIC-CXR-JPG_",
        "nf_folder": "evaluation_results_vision_disease_full_test",
        "nf_prefix": "diseases_full_test_vision_MIMIC-CXR-JPG_",
        "nf_key": "No Finding",  # Key name in disease JSON
        "demo_attrs": ["sex", "age", "ethnicity"],
    },
    "chexpert": {
        "demo_folder": "evaluation_results_vision_full_test_chexpert",
        "demo_prefix": "demographics_full_test_vision_chexpert_",
        "nf_folder": "evaluation_results_vision_nofinding_full_test_chexpert",
        "nf_prefix": "nofinding_full_test_vision_chexpert_",
        "nf_key": "No_Finding",  # Key name in NF-specific JSON
        "demo_attrs": ["sex", "age", "ethnicity"],
    },
    "rexgradient": {
        "demo_folder": "evaluation_results_vision_full_test_rexgradient",
        "demo_prefix": "demographics_full_test_vision_rexgradient_",
        "nf_folder": "evaluation_results_vision_nofinding_full_test_rexgradient",
        "nf_prefix": "nofinding_full_test_vision_rexgradient_",
        "nf_key": "No_Finding",
        "demo_attrs": ["sex", "age"],  # RexGradient has no ethnicity
    },
    "rexgradient_nonpediatric": {
        "demo_folder": "evaluation_results_vision_full_test_rexgradient_nonpediatric",
        "demo_prefix": "demographics_full_test_vision_rexgradient_nonpediatric_",
        "nf_folder": "evaluation_results_vision_nofinding_full_test_rexgradient_nonpediatric",
        "nf_prefix": "nofinding_full_test_vision_rexgradient_nonpediatric_",
        "nf_key": "No_Finding",
        "demo_attrs": ["sex", "age"],
    },
}


def get_latest_json(directory, prefix):
    """Get the most recent JSON file matching a prefix in a directory."""
    pattern = os.path.join(directory, f"{prefix}*.json")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def build_demographic_csv(json_path, attrs):
    """
    Convert demographics JSON to CSV with columns:
    [layer, level, metric, attribute, value]
    
    The LMM script filters: level == "OVERALL" and metric == "auc"
    """
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for layer_name, layer_data in data.items():
        for attr in attrs:
            if attr not in layer_data:
                continue
            auc_ci = layer_data[attr].get('overall_auc_ci', {})
            mean_auc = auc_ci.get('mean')
            if mean_auc is None:
                continue
            rows.append({
                'layer': layer_name,
                'level': 'OVERALL',
                'metric': 'auc',
                'attribute': attr,
                'value': mean_auc,
            })
    return pd.DataFrame(rows)


def build_disease_csv_from_disease_json(json_path):
    """
    For MIMIC: Convert disease full-test JSON (has all 14 diseases) to CSV.
    Columns: [layer, disease, condition, fpr]
    
    Extracts FPR from conditioned metrics for No Finding.
    """
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for layer_name, layer_data in data.items():
        nf_data = layer_data.get('No Finding', {})
        conditioned = nf_data.get('conditioned', {})
        for condition_key, cond_data in conditioned.items():
            fpr = cond_data.get('metrics', {}).get('fpr')
            if fpr is not None:
                rows.append({
                    'layer': layer_name,
                    'disease': 'No Finding',
                    'condition': condition_key,
                    'fpr': fpr,
                })
    return pd.DataFrame(rows)


def build_disease_csv_from_nofinding_json(json_path, nf_key="No_Finding"):
    """
    For CheXpert/RexGradient: Convert NF-specific JSON to CSV.
    Columns: [layer, disease, condition, fpr]
    """
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for layer_name, layer_data in data.items():
        nf_data = layer_data.get(nf_key, {})
        conditioned = nf_data.get('conditioned', {})
        for condition_key, cond_data in conditioned.items():
            fpr = cond_data.get('metrics', {}).get('fpr')
            if fpr is not None:
                rows.append({
                    'layer': layer_name,
                    'disease': 'No Finding',
                    'condition': condition_key,
                    'fpr': fpr,
                })
    return pd.DataFrame(rows)


def process_dataset(dataset_name):
    """Process all models for a given dataset, producing CSVs."""
    config = DATASET_CONFIG[dataset_name]
    eval_base = EVAL_PATHS[dataset_name]
    out_dir = os.path.join(OUTPUT_BASE, dataset_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Processing: {dataset_name.upper()}")
    print(f"  Source: {eval_base}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    for model in MODELS:
        model_eval_dir = os.path.join(eval_base, model)
        if not os.path.isdir(model_eval_dir):
            print(f"  [{model}] SKIP - directory not found: {model_eval_dir}")
            continue

        # --- Demographics CSV ---
        demo_dir = os.path.join(model_eval_dir, config["demo_folder"])
        demo_json = get_latest_json(demo_dir, config["demo_prefix"])

        if demo_json is None:
            print(f"  [{model}] SKIP demographics - no JSON in {demo_dir}")
            dem_df = pd.DataFrame()
        else:
            dem_df = build_demographic_csv(demo_json, config["demo_attrs"])
            dem_path = os.path.join(out_dir, f"demographic_{model}.csv")
            dem_df.to_csv(dem_path, index=False)
            print(f"  [{model}] Demographics: {len(dem_df)} rows -> {dem_path}")

        # --- Disease/NF CSV ---
        nf_dir = os.path.join(model_eval_dir, config["nf_folder"])
        nf_json = get_latest_json(nf_dir, config["nf_prefix"])

        if nf_json is None:
            print(f"  [{model}] SKIP disease - no JSON in {nf_dir}")
            continue

        if dataset_name == "mimic":
            dis_df = build_disease_csv_from_disease_json(nf_json)
        else:
            dis_df = build_disease_csv_from_nofinding_json(nf_json, config["nf_key"])

        dis_path = os.path.join(out_dir, f"disease_conditioned_{model}.csv")
        dis_df.to_csv(dis_path, index=False)
        print(f"  [{model}] Disease/NF: {len(dis_df)} rows -> {dis_path}")

    print(f"\n  Done: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Build LMM input CSVs from evaluation JSONs")
    parser.add_argument('--dataset', choices=['mimic', 'chexpert', 'rexgradient', 'rexgradient_nonpediatric', 'all'],
                        default='all', help='Which dataset to process')
    args = parser.parse_args()

    datasets = ['mimic', 'chexpert', 'rexgradient'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        process_dataset(ds)

    print(f"\n{'='*70}")
    print(f"  ALL DONE. CSVs written to: {OUTPUT_BASE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
