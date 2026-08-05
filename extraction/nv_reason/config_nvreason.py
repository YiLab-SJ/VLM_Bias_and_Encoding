# config_nvreason.py
# Configuration for the NV-Reason-CXR-3B linear probing pipeline.
#
# Architecture (Qwen2.5-VL based):
# Vision Encoder: 32 transformer blocks
# Language Model: 36 Qwen2 decoder blocks

import os

# =============================================================================
# --- 1. CORE PATHS & MODEL CONFIGURATION ---
# =============================================================================
MAIN_PROJECT_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol"

# NV-Reason specific paths
NVREASON_ROOT_DIR = os.path.join(MAIN_PROJECT_ROOT, "other_models", "nv_reason")
NVREASON_MODEL_PATH = os.path.join(NVREASON_ROOT_DIR, "NV-Reason-CXR-3B")

# Shared metadata
METADATA_ATTR_LR_INPUT_ROOT_DIR = os.path.join(MAIN_PROJECT_ROOT, "generated_probe_metadata_from_raw_v4")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE_EXPERIMENT_OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, "probe_experiment_outputs")

# =============================================================================
# --- 2. GENERAL EXPERIMENT PARAMETERS ---
# =============================================================================
RANDOM_STATE = 42
N_BOOTSTRAP_SAMPLES = 1000
OPTIMIZED_BATCH_SIZE = 1 # Required for Qwen-VL dynamic resolution and complex padding

# =============================================================================
# --- 3. DATASET-SPECIFIC CONFIGURATIONS ---
# =============================================================================
PUBLIC_DATASETS_ROOT = "/research/groups/pyigrp/projects/opendatashare/common/public_datasets"
MIMIC_REPORTS_BASE_DIR = MAIN_PROJECT_ROOT
CHEXPERT_REPORTS_CSV_PATH = "/home/apalliko/ondemand/Demographic_classifiers/df_chexpert_plus_240401.csv"

DATASET_CONFIGS = {
    'MIMIC-CXR-JPG': {
        'metadata_attr_lr_file': os.path.join(METADATA_ATTR_LR_INPUT_ROOT_DIR, 'MIMIC-CXR-JPG', 'foundation_fair_meta', 'metadata_attr_lr.csv'),
        'base_image_dir': os.path.join(PUBLIC_DATASETS_ROOT, 'physionet.org/files/mimic-cxr-jpg/2.1.0/'),
        'attributes_to_probe': ['sex', 'age', 'ethnicity'],
        'balance_cols': ['sex', 'ethnicity', 'age'],
        'logic_key': 'MIMIC'
    },
    'chexpert': {
        'metadata_attr_lr_file': os.path.join(METADATA_ATTR_LR_INPUT_ROOT_DIR, 'chexpert', 'foundation_fair_meta', 'metadata_attr_lr.csv'),
        'base_image_dir': os.path.join(PUBLIC_DATASETS_ROOT, 'chexpert-small/CheXpert-v1.0-small/'),
        'attributes_to_probe': ['sex', 'age', 'ethnicity'],
        'balance_cols': ['sex', 'ethnicity', 'age'],
        'logic_key': 'CheXpert'
    },
    'rexgradient': {
        'metadata_attr_lr_file': '/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/rexgradient_dataset/rexgradient_processed/metadata_attr_lr.csv',
        'base_image_dir': '/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/rexgradient_dataset/rexgradient_raw/deid_png/',
        'attributes_to_probe': ['sex', 'age'],
        'balance_cols': ['sex', 'age'],
        'logic_key': 'rexgradient'
    },
}

# =============================================================================
# --- 4. NV-REASON MODEL ARCHITECTURE INFORMATION ---
# =============================================================================
# Hooking the 32 vision blocks found during the probe
VISION_LAYER_NAMES = [f'Vis_Block_{i}' for i in range(32)]

# Hooking the 36 text blocks found during the probe + embedding/norm
TEXT_LAYER_NAMES = ['Txt_Embed'] + [f'Txt_Block_{i}' for i in range(36)] + ['Txt_FinalNorm']

ATTR_CLASS_NAMES_CONFIG = {
    "sex": {0: "Female", 1: "Male"},
    "age": {0: "Age_80+", 1: "Age_60-79", 2: "Age_40-59", 3: "Age_18-39", 4: "Age_0-17"},
    "ethnicity": {0: "White", 1: "Black", 2: "Asian", 3: "Others"}
}