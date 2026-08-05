# config_chexzero.py
# Configuration for the CheXzero linear probing pipeline.
#
# CheXzero = CLIP ViT-B/32 (vision) + CLIP Transformer (text), fine-tuned on CXR.
# Vision encoder: VisualTransformer with 12 ResidualAttentionBlock layers, width=768
#   - Patch size 32x32, input resolution 224 -> 7x7=49 patches + 1 CLS = 50 tokens
#   - proj: 768 -> 512 (applied to CLS token after ln_post)
# Text encoder: Transformer with 12 ResidualAttentionBlock layers, width=512
#   - CLIP SimpleTokenizer, context_length=77
#   - text_projection: 512 -> 512 (applied to EOT token after ln_final)

import os

# =============================================================================
# --- 1. CORE PATHS & MODEL CONFIGURATION ---
# =============================================================================

CHEXZERO_MODEL_FILENAME = "best_128_0.0002_original_15000_0.859.pt"
ALL_CHECKPOINTS_DIR = "/home/apalliko/ondemand/embedding_project/CheXzero/checkpoints/all_checkpoints/CheXzero_Models/"
CHEXZERO_MODEL_PATH = os.path.join(ALL_CHECKPOINTS_DIR, CHEXZERO_MODEL_FILENAME)

# CheXzero source code directory (needed for model.py, clip.py, simple_tokenizer.py)
CHEXZERO_SRC_DIR = "/home/apalliko/ondemand/embedding_project/CheXzero"

# Root of the main project (for accessing shared metadata CSVs)
MAIN_PROJECT_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol"

# Root directory for shared metadata CSVs (same as the original pipeline)
METADATA_ATTR_LR_INPUT_ROOT_DIR = os.path.join(MAIN_PROJECT_ROOT, "generated_probe_metadata_from_raw_v4")

# All outputs for this model go under other_models/chexzero/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROBE_EXPERIMENT_OUTPUT_ROOT_DIR = os.path.join(SCRIPT_DIR, "probe_experiment_outputs")

# =============================================================================
# --- 2. GENERAL EXPERIMENT PARAMETERS ---
# =============================================================================
RANDOM_STATE = 42
N_BOOTSTRAP_SAMPLES = 1000

# =============================================================================
# --- 3. DATASET-SPECIFIC CONFIGURATIONS ---
# =============================================================================
PUBLIC_DATASETS_ROOT = "/research/groups/pyigrp/projects/opendatashare/common/public_datasets"

# Paths for text reports (needed for text extraction)
CHEXPERT_REPORTS_CSV_PATH = "/home/apalliko/ondemand/Demographic_classifiers/df_chexpert_plus_240401.csv"
MIMIC_REPORTS_BASE_DIR = MAIN_PROJECT_ROOT

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
# --- 4. CHEXZERO MODEL ARCHITECTURE INFORMATION ---
# =============================================================================
# CheXzero Vision Encoder: CLIP ViT-B/32
#   - 12 ResidualAttentionBlock layers, hidden_size=768
#   - conv1: 3 -> 768, kernel 32x32, stride 32 (patch embedding)
#   - Input: 224x224 -> 7x7 grid -> 49 patches + 1 CLS = 50 tokens
#   - Each resblock processes (50, B, 768) in LND format
#   - After transformer: ln_post applied to CLS token -> (B, 768)
#   - proj: 768 -> 512 -> final image embedding (B, 512)
#
# CheXzero Text Encoder: CLIP Transformer
#   - 12 ResidualAttentionBlock layers, hidden_size=512
#   - token_embedding: 49408 -> 512
#   - positional_embedding: (77, 512)
#   - Each resblock processes (77, B, 512) in LND format
#   - After transformer: ln_final -> select EOT token -> text_projection (512 -> 512)
#   - Final text embedding: (B, 512)

# Layer names and dimensions for VISION hooks
VISION_LAYER_DIMS = {
    'Vis_Block_1': 768,
    'Vis_Block_2': 768,
    'Vis_Block_3': 768,
    'Vis_Block_4': 768,
    'Vis_Block_5': 768,
    'Vis_Block_6': 768,
    'Vis_Block_7': 768,
    'Vis_Block_8': 768,
    'Vis_Block_9': 768,
    'Vis_Block_10': 768,
    'Vis_Block_11': 768,
    'Vis_Block_12': 768,
    'image_embedding_final': 512,
}

# Layer names and dimensions for TEXT hooks
TEXT_LAYER_DIMS = {
    'Txt_TokenEmbed': 512,
    'Txt_Block_1': 512,
    'Txt_Block_2': 512,
    'Txt_Block_3': 512,
    'Txt_Block_4': 512,
    'Txt_Block_5': 512,
    'Txt_Block_6': 512,
    'Txt_Block_7': 512,
    'Txt_Block_8': 512,
    'Txt_Block_9': 512,
    'Txt_Block_10': 512,
    'Txt_Block_11': 512,
    'Txt_Block_12': 512,
    'text_embedding_final': 512,
}

# =============================================================================
# --- 5. IMAGE PREPROCESSING PARAMETERS ---
# =============================================================================
# CheXzero uses standard CLIP preprocessing:
#   Resize(224) -> CenterCrop(224) -> RGB -> ToTensor -> Normalize
CHEXZERO_IMAGE_SIZE = 224
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# =============================================================================
# --- 6. ATTRIBUTE & PROBING CONFIGURATIONS ---
# =============================================================================
ATTR_CLASS_NAMES_CONFIG = {
    "sex": {0: "Female", 1: "Male"},
    "age": {
        0: "Age_80+",
        1: "Age_60-79",
        2: "Age_40-59",
        3: "Age_18-39",
        4: "Age_0-17"
    },
    "ethnicity": {
        0: "White",
        1: "Black",
        2: "Asian",
        3: "Others"
    },
}

PROBE_TYPES = {
    "sex": "binary",
    "age": "multiclass",
    "ethnicity": "multiclass",
}

# Disease labels (14 CheXpert conditions)
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]
