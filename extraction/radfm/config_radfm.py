# config_radfm.py
# Configuration for the RadFM linear probing pipeline.
# This mirrors the structure of config_medgemma.py but is specific to RadFM.
#
# RadFM Architecture (MultiLLaMAForCausalLM):
#   Vision Encoder:  3D ViT (12 Transformer blocks, hidden_size=768)
#                    Image: 512x512x4, patch_size=32, frame_patch_size=4 -> 256 patches
#   Perceiver Resampler: 6-layer cross-attention, 32 latent queries (dim=768)
#   FC Projection:   Linear(768, 5120) -> maps vision features to LLaMA dim
#   Language Model:  LLaMA-13B (40 decoder layers, hidden_size=5120)
#
# For probing:
#   - Vision: Hook ViT layers + patch embedding, mean-pool patches -> (B, 768)
#             Also capture post-perceiver (B, 768) and projected embedding (B, 5120)
#   - Text:   Feed text-only through LLaMA decoder, hook each layer,
#             mean-pool over non-padding tokens -> (B, 5120)

import os

# =============================================================================
# --- 1. CORE PATHS & MODEL CONFIGURATION ---
# =============================================================================

# RadFM repo and checkpoint paths
RADFM_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
RADFM_CHECKPOINT_PATH = os.path.join(RADFM_REPO_DIR, "pytorch_model.bin")
RADFM_LANGUAGE_FILES_PATH = os.path.join(RADFM_REPO_DIR, "Quick_demo", "Language_files")

# Root of the main project (for accessing shared metadata CSVs)
MAIN_PROJECT_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol"

# Root directory for shared metadata CSVs (same as the original pipeline)
METADATA_ATTR_LR_INPUT_ROOT_DIR = os.path.join(MAIN_PROJECT_ROOT, "generated_probe_metadata_from_raw_v4")

# All outputs for this model go under other_models/radfm/
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
# --- 4. RADFM MODEL ARCHITECTURE INFORMATION ---
# =============================================================================
# RadFM Vision Encoder: 3D ViT
#   - to_patch_embedding: Rearrange + LayerNorm(12288) + Linear(12288, 768) + LayerNorm(768)
#   - pos_embedding: PositionEmbeddingLearned3d (row/col/dep embeddings, dim//3 each)
#   - Transformer: 12 blocks, each = PreNorm->Attention(768, heads=8, dim_head=64) + PreNorm->FFN(768, 2048)
#   For a 2D CXR (512x512, D=4): 256 patch tokens of dimension 768.
#
# Perceiver Resampler: 6-layer PerceiverAttention + FFN
#   - 32 learnable latent queries (dim=768)
#   - Compresses 256 patch tokens -> 32 latent tokens (each 768-d)
#
# FC Projection: Linear(768, 5120)
#   - Maps each of 32 latent tokens from vision dim to LLaMA hidden dim
#
# RadFM Language Model: LLaMA-13B
#   - embed_tokens: Embedding(32000, 5120)
#   - 40 x LlamaDecoderLayer, each outputs (B, seq, 5120)
#   - norm: LlamaRMSNorm(5120)

# Layer names and dimensions for VISION hooks
# We probe: patch embedding, all 12 ViT blocks, plus perceiver output and projected embedding.
# All intermediate features are mean-pooled across patches to a single vector.
VISION_LAYER_DIMS = {
    'Vis_PatchEmbed': 768,           # to_patch_embedding output -> mean-pool -> (B, 768)
    'Vis_Block_1': 768,              # Transformer block 0 -> mean-pool -> (B, 768)
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
    'Vis_Block_12': 768,             # Transformer block 11 -> mean-pool -> (B, 768)
    'Vis_Perceiver': 768,            # PerceiverResampler output -> mean-pool 32 latents -> (B, 768)
    'Vis_Projected': 5120,           # FC projection output -> mean-pool -> (B, 5120)
}

# Layer names and dimensions for TEXT hooks (LLaMA-13B decoder, text-only)
# We feed report text through the decoder without any image input.
# Mean-pool over non-padding tokens at each layer.
TEXT_LAYER_DIMS = {
    'Txt_Embed': 5120,               # embed_tokens output -> mean-pool -> (B, 5120)
    'Txt_Block_1': 5120,             # LlamaDecoderLayer[0] output -> mean-pool -> (B, 5120)
    'Txt_Block_2': 5120,
    'Txt_Block_3': 5120,
    'Txt_Block_4': 5120,
    'Txt_Block_5': 5120,
    'Txt_Block_6': 5120,
    'Txt_Block_7': 5120,
    'Txt_Block_8': 5120,
    'Txt_Block_9': 5120,
    'Txt_Block_10': 5120,
    'Txt_Block_11': 5120,
    'Txt_Block_12': 5120,
    'Txt_Block_13': 5120,
    'Txt_Block_14': 5120,
    'Txt_Block_15': 5120,
    'Txt_Block_16': 5120,
    'Txt_Block_17': 5120,
    'Txt_Block_18': 5120,
    'Txt_Block_19': 5120,
    'Txt_Block_20': 5120,
    'Txt_Block_21': 5120,
    'Txt_Block_22': 5120,
    'Txt_Block_23': 5120,
    'Txt_Block_24': 5120,
    'Txt_Block_25': 5120,
    'Txt_Block_26': 5120,
    'Txt_Block_27': 5120,
    'Txt_Block_28': 5120,
    'Txt_Block_29': 5120,
    'Txt_Block_30': 5120,
    'Txt_Block_31': 5120,
    'Txt_Block_32': 5120,
    'Txt_Block_33': 5120,
    'Txt_Block_34': 5120,
    'Txt_Block_35': 5120,
    'Txt_Block_36': 5120,
    'Txt_Block_37': 5120,
    'Txt_Block_38': 5120,
    'Txt_Block_39': 5120,
    'Txt_Block_40': 5120,            # LlamaDecoderLayer[39] output -> mean-pool -> (B, 5120)
    'Txt_FinalNorm': 5120,           # LlamaRMSNorm output -> mean-pool -> (B, 5120)
}

# =============================================================================
# --- 5. IMAGE PREPROCESSING PARAMETERS ---
# =============================================================================
# RadFM uses RandomResizedCrop to 512x512, ToTensor(), then unsqueeze to 3D with D=4.
# For reproducible feature extraction, we use a deterministic center-crop + resize.
RADFM_IMAGE_SIZE = 512
RADFM_DEPTH_SIZE = 4  # 2D images are interpolated to depth=4 for the 3D ViT

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
