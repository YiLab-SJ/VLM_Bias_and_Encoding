# config_medversa.py
# Configuration for the MedVersa linear probing pipeline.
# This mirrors the structure of config_medgemma.py / config_radfm.py but is
# specific to MedVersa (MedOmni architecture).
#
# MedVersa Architecture (MedOmni):
#   Vision Encoder:  Swin-Base (microsoft/swin-base-patch4-window7-224)
#                    4 stages with depths=[2,2,18,2] = 24 blocks total
#                    Dims per stage: [128, 256, 512, 1024]
#                    Input: 224x224, patch_size=4, window_size=7
#                    Final spatial resolution: 7x7 = 49 tokens at dim 1024
#   Vision LayerNorm: LayerNorm(1024)  (ln_vision_2d)
#   Vision Projection: Linear(1024, 4096)  (llama_proj_2d)
#   Language Model:  LLaMA-2-7B-chat with LoRA (r=16, q_proj + v_proj)
#                    32 decoder layers, hidden_size=4096
#
# For probing:
#   - Vision: Hook Swin-B layers (24 blocks + embed + final LN),
#             mean-pool spatial tokens -> single vector per sample.
#             Also capture ln_vision_2d and projected embedding.
#   - Text:   Feed text-only through LLaMA decoder, hook each layer,
#             mean-pool over non-padding tokens -> (B, 4096).

import os

# =============================================================================
# --- 1. CORE PATHS & MODEL CONFIGURATION ---
# =============================================================================

# MedVersa repo directory (contains config.json, model.safetensors, medomni/)
MEDVERSA_REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# Root of the main project (for accessing shared metadata CSVs)
MAIN_PROJECT_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol"

# Root directory for shared metadata CSVs (same as the original pipeline)
METADATA_ATTR_LR_INPUT_ROOT_DIR = os.path.join(MAIN_PROJECT_ROOT, "generated_probe_metadata_from_raw_v4")

# All outputs for this model go under other_models/medversa/
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
# --- 4. MEDVERSA MODEL ARCHITECTURE INFORMATION ---
# =============================================================================
# Swin-Base 2D Vision Encoder (microsoft/swin-base-patch4-window7-224)
#   - embeddings (SwinEmbeddings): patch_embed + pos_embed -> (B, 3136, 128)
#   - encoder.layers[0] (SwinStage 0): 2 SwinLayer blocks, dim=128
#     + downsample (SwinPatchMerging): 128 -> 256, 3136 -> 784 tokens
#   - encoder.layers[1] (SwinStage 1): 2 SwinLayer blocks, dim=256
#     + downsample (SwinPatchMerging): 256 -> 512, 784 -> 196 tokens
#   - encoder.layers[2] (SwinStage 2): 18 SwinLayer blocks, dim=512
#     + downsample (SwinPatchMerging): 512 -> 1024, 196 -> 49 tokens
#   - encoder.layers[3] (SwinStage 3): 2 SwinLayer blocks, dim=1024
#     (no downsample for last stage)
#   - layernorm: LayerNorm(1024) -> (B, 49, 1024)
#
# MedVersa-specific projections (on top of Swin output):
#   - ln_vision_2d: LayerNorm(1024) applied after spatial pooling
#   - llama_proj_2d: Linear(1024, 4096) maps vision features to LLaMA dim
#
# LLaMA-2-7B-chat Language Model (with LoRA):
#   - embed_tokens: Embedding(32005, 4096) (resized for special tokens)
#   - 32 x LlamaDecoderLayer, each outputs (B, seq, 4096)
#   - norm: LlamaRMSNorm(4096)

SWIN_STAGE_DEPTHS = [2, 2, 18, 2]   # 24 transformer blocks total
SWIN_STAGE_DIMS = [128, 256, 512, 1024]
MEDVERSA_IMAGE_SIZE = 224

# Image preprocessing (CLIP normalization used by MedVersa)
MEDVERSA_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
MEDVERSA_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# Layer names and dimensions for VISION hooks
# We probe: Swin embeddings, all 24 encoder blocks, Swin final layernorm,
# MedVersa's ln_vision_2d, and the llama_proj_2d projected embedding.
# All intermediate features are mean-pooled across spatial tokens.
VISION_LAYER_DIMS = {}
VISION_LAYER_DIMS['Vis_Embed'] = 128                # SwinEmbeddings output -> mean-pool -> (B, 128)
for stage_idx, (depth, dim) in enumerate(zip(SWIN_STAGE_DEPTHS, SWIN_STAGE_DIMS)):
    for block_idx in range(depth):
        VISION_LAYER_DIMS[f'Vis_S{stage_idx}_B{block_idx + 1}'] = dim
VISION_LAYER_DIMS['Vis_FinalNorm'] = 1024            # Swin layernorm -> mean-pool -> (B, 1024)
VISION_LAYER_DIMS['Vis_LNVision'] = 1024             # MedVersa ln_vision_2d -> mean-pool -> (B, 1024)
VISION_LAYER_DIMS['Vis_Projected'] = 4096            # llama_proj_2d output -> mean-pool -> (B, 4096)

# Layer names and dimensions for TEXT hooks (LLaMA-2-7B decoder, text-only)
# We feed report text through the decoder without any image input.
# Mean-pool over non-padding tokens at each layer.
LLAMA_NUM_LAYERS = 32
LLAMA_HIDDEN_SIZE = 4096

TEXT_LAYER_DIMS = {}
TEXT_LAYER_DIMS['Txt_Embed'] = LLAMA_HIDDEN_SIZE     # embed_tokens output -> mean-pool -> (B, 4096)
for i in range(LLAMA_NUM_LAYERS):
    TEXT_LAYER_DIMS[f'Txt_Block_{i + 1}'] = LLAMA_HIDDEN_SIZE
TEXT_LAYER_DIMS['Txt_FinalNorm'] = LLAMA_HIDDEN_SIZE  # LlamaRMSNorm output -> mean-pool -> (B, 4096)

# =============================================================================
# --- 5. ATTRIBUTE & PROBING CONFIGURATIONS ---
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
