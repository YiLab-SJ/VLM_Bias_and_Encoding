# config_chexagent.py
# Configuration for the CheXagent-2-3b linear probing pipeline.
# This mirrors the structure of config_biovilt.py but is specific to CheXagent.
#
# CheXagent-2-3b = XraySigLIP ViT-L/16 (vision) + Phi-2 3B (LLM text decoder)
# The vision encoder is a SigLIP ViT-L/16 model (24 layers, 1024 hidden).
# The text "encoder" is the Phi-2 decoder (32 layers, 2560 hidden).
# There is NO contrastive text encoder; the LLM decoder layers are probed.

import os

# =============================================================================
# --- 1. CORE PATHS & MODEL CONFIGURATION ---
# =============================================================================

# CheXagent HuggingFace model identifier (weights auto-downloaded)
CHEXAGENT_MODEL_NAME = "StanfordAIMI/CheXagent-2-3b"

# Root of the main project (for accessing shared metadata CSVs)
MAIN_PROJECT_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol"

# Root directory for shared metadata CSVs (same as the original pipeline)
METADATA_ATTR_LR_INPUT_ROOT_DIR = os.path.join(MAIN_PROJECT_ROOT, "generated_probe_metadata_from_raw_v4")

# All outputs for this model go under other_models/chexagent/
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
# --- 4. CHEXAGENT MODEL ARCHITECTURE INFORMATION ---
# =============================================================================
# CheXagent Vision Encoder: XraySigLIP ViT-L/16
#   - SigLIP ViT-L/16: 24 transformer layers, hidden_size=1024
#   - Image: 512x512 -> patch 16x16 -> 32x32 = 1024 patch tokens
#   - Each ViT layer output: (B, 1024, 1024) -> mean-pool -> (B, 1024)
#   - After ViT: CLIPModel resampler with:
#       attn_pool MLP: 1024 -> 10240 -> 2560
#       ln_post: LayerNorm(2560)
#       proj: Linear(2560, 2560)
#     Output: (B, 1024, 2560) -> mean-pool -> (B, 2560)
#   - image_embedding_final: the mean-pooled resampler output (2560-dim)
#
# CheXagent Text "Encoder": Phi-2 Decoder (used as text representation)
#   - 32 Phi decoder layers, hidden_size=2560
#   - embed_tokens: vocab -> 2560
#   - layers[0..31]: PhiDecoderLayer, output 2560-dim
#   - final_layernorm: LayerNorm(2560)
#   - For probing: we feed reports through the LLM and extract hidden states
#     at each layer, using the last non-padding token as the representation.
#   - text_embedding_final: final_layernorm output -> last token -> (B, 2560)

# Layer names and dimensions for VISION hooks
# We probe every ViT layer + the final resampled embedding.
VISION_LAYER_DIMS = {
    'Vis_SigLIP_Block_1': 1024,     # vision_model.encoder.layers[0] -> mean-pool -> (B, 1024)
    'Vis_SigLIP_Block_2': 1024,
    'Vis_SigLIP_Block_3': 1024,
    'Vis_SigLIP_Block_4': 1024,
    'Vis_SigLIP_Block_5': 1024,
    'Vis_SigLIP_Block_6': 1024,
    'Vis_SigLIP_Block_7': 1024,
    'Vis_SigLIP_Block_8': 1024,
    'Vis_SigLIP_Block_9': 1024,
    'Vis_SigLIP_Block_10': 1024,
    'Vis_SigLIP_Block_11': 1024,
    'Vis_SigLIP_Block_12': 1024,
    'Vis_SigLIP_Block_13': 1024,
    'Vis_SigLIP_Block_14': 1024,
    'Vis_SigLIP_Block_15': 1024,
    'Vis_SigLIP_Block_16': 1024,
    'Vis_SigLIP_Block_17': 1024,
    'Vis_SigLIP_Block_18': 1024,
    'Vis_SigLIP_Block_19': 1024,
    'Vis_SigLIP_Block_20': 1024,
    'Vis_SigLIP_Block_21': 1024,
    'Vis_SigLIP_Block_22': 1024,
    'Vis_SigLIP_Block_23': 1024,
    'Vis_SigLIP_Block_24': 1024,
    'image_embedding_final': 2560,  # Resampler output: mean-pool -> (B, 2560)
}

# Layer names and dimensions for TEXT (Phi-2 decoder) hooks
TEXT_LAYER_DIMS = {
    'Txt_TokenEmbed': 2560,         # model.embed_tokens output -> mean-pool -> (B, 2560)
    'Txt_PhiBlock_1': 2560,         # model.layers[0] output -> last token -> (B, 2560)
    'Txt_PhiBlock_2': 2560,
    'Txt_PhiBlock_3': 2560,
    'Txt_PhiBlock_4': 2560,
    'Txt_PhiBlock_5': 2560,
    'Txt_PhiBlock_6': 2560,
    'Txt_PhiBlock_7': 2560,
    'Txt_PhiBlock_8': 2560,
    'Txt_PhiBlock_9': 2560,
    'Txt_PhiBlock_10': 2560,
    'Txt_PhiBlock_11': 2560,
    'Txt_PhiBlock_12': 2560,
    'Txt_PhiBlock_13': 2560,
    'Txt_PhiBlock_14': 2560,
    'Txt_PhiBlock_15': 2560,
    'Txt_PhiBlock_16': 2560,
    'Txt_PhiBlock_17': 2560,
    'Txt_PhiBlock_18': 2560,
    'Txt_PhiBlock_19': 2560,
    'Txt_PhiBlock_20': 2560,
    'Txt_PhiBlock_21': 2560,
    'Txt_PhiBlock_22': 2560,
    'Txt_PhiBlock_23': 2560,
    'Txt_PhiBlock_24': 2560,
    'Txt_PhiBlock_25': 2560,
    'Txt_PhiBlock_26': 2560,
    'Txt_PhiBlock_27': 2560,
    'Txt_PhiBlock_28': 2560,
    'Txt_PhiBlock_29': 2560,
    'Txt_PhiBlock_30': 2560,
    'Txt_PhiBlock_31': 2560,
    'Txt_PhiBlock_32': 2560,
    'text_embedding_final': 2560,   # final_layernorm output -> last token -> (B, 2560)
}

# =============================================================================
# --- 5. IMAGE PREPROCESSING PARAMETERS ---
# =============================================================================
# CheXagent uses XraySigLIP's processor: Resize(512x512) + Normalize
# The CLIPModel.image_transform handles: Resize(512,512) -> ToTensor -> Normalize
CHEXAGENT_IMAGE_SIZE = 512

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
