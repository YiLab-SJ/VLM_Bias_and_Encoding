# config_llavamed.py
# Configuration for the LLaVA-Med 1.5 linear probing pipeline.
# This mirrors the structure of config_biovilt.py but is specific to LLaVA-Med 1.5.
import os

# =============================================================================
# --- 1. CORE PATHS & MODEL CONFIGURATION ---
# =============================================================================

# LLaVA-Med 1.5 model path (Mistral-7B backbone)
LLAVAMED_MODEL_PATH = "/home/apalliko/.cache/huggingface/hub/models--microsoft--llava-med-v1.5-mistral-7b/snapshots/91bb16c122001ddc9cf1fd36ce1dae09448943a2"

# LLaVA-Med repo (for model builder code)
LLAVAMED_REPO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LLaVA-Med")

# CLIP vision tower name (used by the model)
CLIP_VISION_TOWER = "openai/clip-vit-large-patch14-336"

# Root of the main project (for accessing shared metadata CSVs)
MAIN_PROJECT_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol"

# Root directory for shared metadata CSVs (same as the original pipeline)
METADATA_ATTR_LR_INPUT_ROOT_DIR = os.path.join(MAIN_PROJECT_ROOT, "generated_probe_metadata_from_raw_v4")

# All outputs for this model go under other_models/llavamed1p5/
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
MIMIC_REPORTS_BASE_DIR = MAIN_PROJECT_ROOT  # Reports are derived from image filenames relative to this

DATASET_CONFIGS = {
    'MIMIC-CXR-JPG': {
        'metadata_attr_lr_file': os.path.join(METADATA_ATTR_LR_INPUT_ROOT_DIR, 'MIMIC-CXR-JPG', 'foundation_fair_meta', 'metadata_attr_lr.csv'),
        'base_image_dir': os.path.join(PUBLIC_DATASETS_ROOT, 'physionet.org/files/mimic-cxr-jpg/2.1.0/'),
        'attributes_to_probe': ['sex', 'age', 'ethnicity', 'sex_ethnicity'],
        'balance_cols': ['sex', 'ethnicity', 'age'],
        'logic_key': 'MIMIC'
    },
    'chexpert': {
        'metadata_attr_lr_file': os.path.join(METADATA_ATTR_LR_INPUT_ROOT_DIR, 'chexpert', 'foundation_fair_meta', 'metadata_attr_lr.csv'),
        'base_image_dir': os.path.join(PUBLIC_DATASETS_ROOT, 'chexpert-small/CheXpert-v1.0-small/'),
        'attributes_to_probe': ['sex', 'age', 'ethnicity', 'sex_ethnicity'],
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
# --- 4. LLAVA-MED 1.5 MODEL ARCHITECTURE INFORMATION ---
# =============================================================================
# LLaVA-Med 1.5 Architecture:
#
# VISION ENCODER: CLIP ViT-L/14-336 (openai/clip-vit-large-patch14-336)
#   - 24 transformer encoder layers, hidden_size = 1024
#   - Image size: 336x336, patch size: 14 → 576 patches (+1 CLS token = 577)
#   - mm_vision_select_layer = -2 (layer 22, penultimate)
#   - mm_vision_select_feature = 'patch' (CLS token removed, 576 patch tokens)
#
# MM PROJECTOR: mlp2x_gelu
#   - Sequential(Linear(1024, 4096), GELU(), Linear(4096, 4096))
#   - Maps vision patch features to LLM embedding space
#   - Output: (B, 576, 4096) per image
#
# LANGUAGE MODEL: Mistral-7B-Instruct-v0.2
#   - 32 transformer decoder layers, hidden_size = 4096
#   - Vocabulary: 32000 tokens
#
# Layer extraction strategy:
#   VISION: Use CLIP ViT with output_hidden_states=True, mean-pool patch tokens.
#           Also pass through mm_projector and mean-pool.
#   TEXT:   Use Mistral with text-only input, output_hidden_states=True, mean-pool.

# Layer names and dimensions for VISION hooks/extraction
# We extract all 24 CLIP ViT layers + embeddings + mm_projector output.
# All features are mean-pooled over patch tokens to get a single vector.
VISION_LAYER_DIMS = {
    'Vis_CLIP_Embed': 1024,         # CLIP patch embeddings (before first layer) -> mean over patches -> (B, 1024)
}
for _i in range(1, 25):
    VISION_LAYER_DIMS[f'Vis_CLIP_Layer_{_i}'] = 1024  # CLIP encoder layer i output -> mean over patches -> (B, 1024)
VISION_LAYER_DIMS['Vis_MM_Projector'] = 4096  # mm_projector output -> mean over patches -> (B, 4096)

# Layer names and dimensions for TEXT extraction
# We extract Mistral embeddings + all 32 decoder layers.
# Features are mean-pooled over token positions.
TEXT_LAYER_DIMS = {
    'Txt_Mistral_Embed': 4096,      # Mistral token embeddings -> mean over tokens -> (B, 4096)
}
for _i in range(1, 33):
    TEXT_LAYER_DIMS[f'Txt_Mistral_Layer_{_i}'] = 4096  # Mistral decoder layer i output -> mean over tokens -> (B, 4096)

# =============================================================================
# --- 5. IMAGE PREPROCESSING PARAMETERS ---
# =============================================================================
# CLIP ViT-L/14-336 uses CLIPImageProcessor with:
#   - Resize to 336x336
#   - Center crop 336
#   - Normalize with CLIP mean/std
# We use CLIPImageProcessor.from_pretrained() which handles this automatically.

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
    "sex_ethnicity": {
        0: "Male_White", 1: "Female_White",
        2: "Male_Black", 3: "Female_Black",
        4: "Male_Asian", 5: "Female_Asian",
        6: "Male_Others", 7: "Female_Others"
    }
}

PROBE_TYPES = {
    "sex": "binary",
    "age": "multiclass",
    "ethnicity": "multiclass",
    "sex_ethnicity": "multiclass"
}

# Disease labels (14 CheXpert conditions)
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]
