# config_biovilt.py
# Configuration for the BioViL-T linear probing pipeline.
# This mirrors the structure of config_probe.py but is specific to BioViL-T.
import os

# =============================================================================
# --- 1. CORE PATHS & MODEL CONFIGURATION ---
# =============================================================================

# BioViL-T model paths
BIOVILT_MODEL_DIR = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/BioVilT/biovil_t_local"
BIOVILT_IMAGE_WEIGHTS_PATH = os.path.join(BIOVILT_MODEL_DIR, "biovil_t_image_model_proj_size_128.pt")

# Root of the main project (for accessing shared metadata CSVs)
MAIN_PROJECT_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol"

# Root directory for shared metadata CSVs (same as the original pipeline)
METADATA_ATTR_LR_INPUT_ROOT_DIR = os.path.join(MAIN_PROJECT_ROOT, "generated_probe_metadata_from_raw_v4")

# All outputs for this model go under other_models/biovilt/
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
# --- 4. BIOVIL-T MODEL ARCHITECTURE INFORMATION ---
# =============================================================================
# BioViL-T Vision Encoder: ResNet-50 backbone + backbone_to_vit Conv + MLP Projector
#   - ResNet-50 layer1..4 output dims: 256, 512, 1024, 2048
#   - backbone_to_vit conv: 2048 -> 256 (spatial features)
#   - NOTE: vit_pooler (3 transformer blocks) is ONLY used when a previous_image
#     is supplied (longitudinal/temporal mode). For single-image inference it is
#     bypassed entirely and a learned missing_previous_emb is used instead.
#   - img_embedding (pre-projection): 512-dim = avg_pool(cat(backbone_to_vit, missing_emb))
#   - projected_global_embedding (post-projection): 128-dim
#
# BioViL-T Text Encoder: CXR-BERT (12-layer BERT, hidden_size=768)
#   - bert.embeddings.word_embeddings: 768-dim
#   - bert.encoder.layer[0..11]: 768-dim each
#   - cls_projection_head: 768 -> 128-dim

# Layer names and dimensions for VISION hooks
# We probe: ResNet stages, backbone_to_vit projection, encoder-level pooled
# embedding, and the final projected embedding.  The ViT pooler blocks are NOT
# included because they do not fire during single-image inference.
# All intermediate features are global-average-pooled to a single vector.
VISION_LAYER_DIMS = {
    'Vis_ResNet_Layer1': 256,       # encoder.encoder.layer1 output -> GAP -> (B, 256)
    'Vis_ResNet_Layer2': 512,       # encoder.encoder.layer2 output -> GAP -> (B, 512)
    'Vis_ResNet_Layer3': 1024,      # encoder.encoder.layer3 output -> GAP -> (B, 1024)
    'Vis_ResNet_Layer4': 2048,      # encoder.encoder.layer4 output -> GAP -> (B, 2048)
    'Vis_BackboneToViT': 256,       # encoder.backbone_to_vit Conv2d(2048->256) -> GAP -> (B, 256)
    'img_embedding': 512,           # encoder output avg_pool(cat(backbone_to_vit, missing_emb)) -> (B, 512)
    'image_embedding_final': 128,   # projected_global_embedding: mean(projector(patch_fused)) -> (B, 128)
}

# Layer names and dimensions for TEXT hooks
TEXT_LAYER_DIMS = {
    'Txt_TokenEmbed': 768,          # bert.embeddings output -> mean -> (B, 768)
    'Txt_Block_1': 768,             # bert.encoder.layer.0 output -> CLS -> (B, 768)
    'Txt_Block_2': 768,
    'Txt_Block_3': 768,
    'Txt_Block_4': 768,
    'Txt_Block_5': 768,
    'Txt_Block_6': 768,
    'Txt_Block_7': 768,
    'Txt_Block_8': 768,
    'Txt_Block_9': 768,
    'Txt_Block_10': 768,
    'Txt_Block_11': 768,
    'Txt_Block_12': 768,
    'text_embedding_final': 128,    # cls_projection_head output: (B, 128)
}

# =============================================================================
# --- 5. IMAGE PREPROCESSING PARAMETERS ---
# =============================================================================
# BioViL-T uses: Resize(512) -> CenterCrop(480) -> ToTensor() -> ExpandChannels()
# The library's create_chest_xray_transform_for_inference handles this.
BIOVILT_IMAGE_RESIZE = 512
BIOVILT_IMAGE_CENTER_CROP = 480

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
    "ethnicity": "multiclass"
}

# Disease labels (14 CheXpert conditions)
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]
