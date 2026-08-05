# config_medgemma.py
# Configuration for the MedGemma 1.5 (4B-IT) linear probing pipeline.
# This mirrors the structure of config_biovilt.py but is specific to MedGemma 1.5.
#
# MedGemma 1.5 Architecture (Gemma3ForConditionalGeneration):
#   Vision Encoder:  SigLIP ViT (27 SiglipEncoderLayer blocks, hidden_size=1152)
#                    Image: 896x896, patch_size=14 -> 4096 patches
#   Multi-modal Projector: AvgPool2d(4,4) -> RMSNorm -> Linear(1152, 2560)
#                          Reduces 4096 patches -> 256 tokens, projects 1152 -> 2560
#   Language Model:  Gemma3 decoder (34 layers, hidden_size=2560, decoder-only)
#
# For probing:
#   - Vision: Hook SigLIP layers + embeddings, mean-pool patches -> (B, 1152)
#             Also capture projected embedding -> (B, 2560)
#   - Text:   Feed text-only through Gemma3 decoder, hook each layer,
#             mean-pool over non-padding tokens -> (B, 2560)

import os

# =============================================================================
# --- 1. CORE PATHS & MODEL CONFIGURATION ---
# =============================================================================

# MedGemma 1.5 model path (local HuggingFace cache)
MEDGEMMA_MODEL_PATH = "/home/apalliko/.cache/huggingface/hub/models--google--medgemma-1.5-4b-it/snapshots/e9792da5fb8ee651083d345ec4bce07c3c9f1641"

# Root of the main project (for accessing shared metadata CSVs)
MAIN_PROJECT_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol"

# Root directory for shared metadata CSVs (same as the original pipeline)
METADATA_ATTR_LR_INPUT_ROOT_DIR = os.path.join(MAIN_PROJECT_ROOT, "generated_probe_metadata_from_raw_v4")

# All outputs for this model go under other_models/medgemma_1p5/
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
# --- 4. MEDGEMMA 1.5 MODEL ARCHITECTURE INFORMATION ---
# =============================================================================
# MedGemma 1.5 Vision Encoder: SigLIP ViT
#   - SiglipVisionEmbeddings: patch_embed + position_embed -> (B, 4096, 1152)
#   - SiglipEncoder: 27 x SiglipEncoderLayer, each outputs (B, 4096, 1152)
#   - post_layernorm: LayerNorm -> (B, 4096, 1152)
#   - Multi-modal Projector:
#       - AvgPool2d(4,4): (B, 4096, 1152) -> (B, 256, 1152)
#       - mm_soft_emb_norm (RMSNorm): (B, 256, 1152)
#       - mm_input_projection_weight: (B, 256, 1152) @ (1152, 2560) -> (B, 256, 2560)
#
# MedGemma 1.5 Language Model: Gemma3 decoder
#   - embed_tokens: vocab -> (B, seq, 2560)
#   - 34 x Gemma3DecoderLayer, each outputs (B, seq, 2560)
#     (mix of sliding_attention and full_attention layers)
#   - norm (RMSNorm): (B, seq, 2560)

# Layer names and dimensions for VISION hooks
# We probe: SigLIP embeddings, all 27 encoder blocks, post-layernorm, and projected embedding.
# All intermediate features are mean-pooled across patches to a single vector.
VISION_LAYER_DIMS = {
    'Vis_Embed': 1152,              # SiglipVisionEmbeddings output -> mean-pool -> (B, 1152)
    'Vis_Block_1': 1152,            # SiglipEncoderLayer[0] output -> mean-pool -> (B, 1152)
    'Vis_Block_2': 1152,
    'Vis_Block_3': 1152,
    'Vis_Block_4': 1152,
    'Vis_Block_5': 1152,
    'Vis_Block_6': 1152,
    'Vis_Block_7': 1152,
    'Vis_Block_8': 1152,
    'Vis_Block_9': 1152,
    'Vis_Block_10': 1152,
    'Vis_Block_11': 1152,
    'Vis_Block_12': 1152,
    'Vis_Block_13': 1152,
    'Vis_Block_14': 1152,
    'Vis_Block_15': 1152,
    'Vis_Block_16': 1152,
    'Vis_Block_17': 1152,
    'Vis_Block_18': 1152,
    'Vis_Block_19': 1152,
    'Vis_Block_20': 1152,
    'Vis_Block_21': 1152,
    'Vis_Block_22': 1152,
    'Vis_Block_23': 1152,
    'Vis_Block_24': 1152,
    'Vis_Block_25': 1152,
    'Vis_Block_26': 1152,
    'Vis_Block_27': 1152,           # SiglipEncoderLayer[26] output -> mean-pool -> (B, 1152)
    'Vis_PostNorm': 1152,           # post_layernorm output -> mean-pool -> (B, 1152)
    'Vis_Projected': 2560,          # Multi-modal projector output -> mean-pool -> (B, 2560)
}

# Layer names and dimensions for TEXT hooks (Gemma3 decoder, text-only)
# We feed report text through the decoder without any image input.
# Mean-pool over non-padding tokens at each layer.
TEXT_LAYER_DIMS = {
    'Txt_Embed': 2560,              # embed_tokens output -> mean-pool -> (B, 2560)
    'Txt_Block_1': 2560,            # Gemma3DecoderLayer[0] output -> mean-pool -> (B, 2560)
    'Txt_Block_2': 2560,
    'Txt_Block_3': 2560,
    'Txt_Block_4': 2560,
    'Txt_Block_5': 2560,
    'Txt_Block_6': 2560,
    'Txt_Block_7': 2560,
    'Txt_Block_8': 2560,
    'Txt_Block_9': 2560,
    'Txt_Block_10': 2560,
    'Txt_Block_11': 2560,
    'Txt_Block_12': 2560,
    'Txt_Block_13': 2560,
    'Txt_Block_14': 2560,
    'Txt_Block_15': 2560,
    'Txt_Block_16': 2560,
    'Txt_Block_17': 2560,
    'Txt_Block_18': 2560,
    'Txt_Block_19': 2560,
    'Txt_Block_20': 2560,
    'Txt_Block_21': 2560,
    'Txt_Block_22': 2560,
    'Txt_Block_23': 2560,
    'Txt_Block_24': 2560,
    'Txt_Block_25': 2560,
    'Txt_Block_26': 2560,
    'Txt_Block_27': 2560,
    'Txt_Block_28': 2560,
    'Txt_Block_29': 2560,
    'Txt_Block_30': 2560,
    'Txt_Block_31': 2560,
    'Txt_Block_32': 2560,
    'Txt_Block_33': 2560,
    'Txt_Block_34': 2560,           # Gemma3DecoderLayer[33] output -> mean-pool -> (B, 2560)
    'Txt_FinalNorm': 2560,          # language_model.norm output -> mean-pool -> (B, 2560)
}

# =============================================================================
# --- 5. IMAGE PREPROCESSING PARAMETERS ---
# =============================================================================
# MedGemma 1.5 uses Gemma3ImageProcessor:
#   Resize to 896x896, rescale 1/255, normalize mean=0.5 std=0.5, convert to RGB
MEDGEMMA_IMAGE_SIZE = 896

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
    }
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
