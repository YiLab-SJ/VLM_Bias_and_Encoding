# llavamed_extract_image_layers.py
# Extracts layer-wise vision embeddings from LLaVA-Med 1.5's CLIP ViT-L/14-336
# vision encoder and the mm_projector (MLP2x_GELU).
#
# Architecture:
#   - CLIP ViT-L/14-336: 24 transformer layers, hidden_size=1024, 576 patches
#   - mm_projector: Linear(1024,4096) -> GELU -> Linear(4096,4096)
#
# Strategy:
#   - Load CLIP ViT + mm_projector weights from LLaVA-Med safetensors (no need
#     to load the full 7B LLM, saving ~13GB of GPU memory).
#   - Run CLIP with output_hidden_states=True to get all 25 hidden states.
#   - Mean-pool over patch tokens (excluding CLS) for each layer.
#   - Pass selected layer through mm_projector, mean-pool.
#
# Output format matches the original pipeline:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python llavamed_extract_image_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import re
import random
import argparse
import logging
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from tqdm import tqdm

# HuggingFace imports
from transformers import CLIPVisionModel, CLIPImageProcessor
from safetensors import safe_open

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_llavamed import (
    LLAVAMED_MODEL_PATH, CLIP_VISION_TOWER,
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    VISION_LAYER_DIMS
)

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# CLIP ViT-L is lightweight (~430M params), so larger batches are fine
OPTIMIZED_BATCH_SIZE = 64
OPTIMIZED_NUM_WORKERS = min(8, os.cpu_count() if os.cpu_count() else 1)


# =============================================================================
# --- Model Loading ---
# =============================================================================
def load_vision_components(model_path, clip_name, device):
    """
    Load CLIP ViT + mm_projector weights from LLaVA-Med safetensors.
    This avoids loading the full 7B Mistral LLM.
    """
    logging.info(f"Loading CLIP ViT from: {clip_name}")
    clip_model = CLIPVisionModel.from_pretrained(clip_name)
    image_processor = CLIPImageProcessor.from_pretrained(clip_name)

    # Load fine-tuned vision tower weights from LLaVA-Med checkpoint
    logging.info(f"Loading fine-tuned vision weights from: {model_path}")
    vt_state_dict = {}
    proj_state_dict = {}

    # Scan all safetensors files
    safetensor_files = sorted([
        f for f in os.listdir(model_path)
        if f.endswith('.safetensors')
    ])

    for sf_file in safetensor_files:
        sf_path = os.path.join(model_path, sf_file)
        with safe_open(sf_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if 'vision_tower.vision_tower.' in key:
                    new_key = key.replace('model.vision_tower.vision_tower.', '')
                    vt_state_dict[new_key] = f.get_tensor(key)
                elif 'mm_projector' in key:
                    new_key = key.replace('model.mm_projector.', '')
                    proj_state_dict[new_key] = f.get_tensor(key)

    # Apply fine-tuned vision weights (may be identical to original CLIP if frozen)
    if vt_state_dict:
        msg = clip_model.load_state_dict(vt_state_dict, strict=False)
        logging.info(f"Vision tower weights loaded: missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}")
    else:
        logging.info("No fine-tuned vision tower weights found; using original CLIP weights.")

    # Build mm_projector: mlp2x_gelu -> Sequential(Linear(1024, 4096), GELU, Linear(4096, 4096))
    mm_hidden_size = 1024   # CLIP hidden size
    hidden_size = 4096      # Mistral hidden size
    mm_projector = nn.Sequential(
        nn.Linear(mm_hidden_size, hidden_size),
        nn.GELU(),
        nn.Linear(hidden_size, hidden_size)
    )

    if proj_state_dict:
        msg = mm_projector.load_state_dict(proj_state_dict, strict=True)
        logging.info(f"mm_projector weights loaded: {msg}")
    else:
        logging.warning("WARNING: No mm_projector weights found!")

    clip_model.to(device=device, dtype=torch.float16).eval()
    mm_projector.to(device=device, dtype=torch.float16).eval()

    for param in clip_model.parameters():
        param.requires_grad = False
    for param in mm_projector.parameters():
        param.requires_grad = False

    logging.info(f"CLIP ViT + mm_projector loaded to {device} (fp16) and frozen.")
    return clip_model, mm_projector, image_processor


# =============================================================================
# --- Dataset ---
# =============================================================================
class LLaVAMedVisionDataset(Dataset):
    """Dataset for LLaVA-Med vision feature extraction using CLIP's image processor."""

    def __init__(self, metadata_csv_path, target_split_value, base_image_dir, image_processor):
        self.base_image_dir = base_image_dir
        self.image_processor = image_processor
        full_df = pd.read_csv(metadata_csv_path)
        self.metadata_df = full_df[full_df["split"] == target_split_value].reset_index(drop=True)

        if self.metadata_df.empty:
            logging.warning(f"No data found for split {target_split_value} in {metadata_csv_path}.")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        data_row = self.metadata_df.iloc[idx]
        relative_img_path = data_row['filename']

        try:
            full_img_path = os.path.join(self.base_image_dir, relative_img_path)
            img = Image.open(full_img_path)

            # Fix for 16-bit PNGs (RexGradient)
            if img.mode in ('I;16', 'I'):
                import numpy as _np
                arr = _np.array(img, dtype=_np.float32)
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = (arr - mn) / (mx - mn) * 255.0
                else:
                    arr = _np.zeros_like(arr)
                img = Image.fromarray(arr.astype(_np.uint8), mode='L')

            # CLIP expects RGB images
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Use CLIP's image processor (resize, center crop, normalize)
            pixel_values = self.image_processor(images=img, return_tensors='pt')['pixel_values'].squeeze(0)
            return {'image': pixel_values, 'labels': data_row.to_dict()}
        except Exception as e:
            return None


# =============================================================================
# --- Collate Function ---
# =============================================================================
def collate_fn_robust(batch_list):
    """Collate function that handles None samples from failed image loads."""
    batch_list = [item for item in batch_list if item is not None]
    if not batch_list:
        return None

    images = torch.stack([item['image'] for item in batch_list])

    list_of_label_dicts = [item['labels'] for item in batch_list]
    collated_labels = {}
    if list_of_label_dicts:
        for key in list_of_label_dicts[0].keys():
            values = [d[key] for d in list_of_label_dicts]
            if all(isinstance(v, (int, float, np.number)) for v in values):
                collated_labels[key] = torch.utils.data.dataloader.default_collate(values)
            else:
                collated_labels[key] = values

    return {'image': images, 'labels': collated_labels}


# =============================================================================
# --- Main Extraction Function ---
# =============================================================================
def extract_vision_layers_and_save(dataset, clip_model, mm_projector, device, output_dir):
    """Extract features from all CLIP ViT layers + mm_projector and save."""

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=collate_fn_robust,
        pin_memory=(device != "cpu")
    )

    # Config: which CLIP layer is used by LLaVA's vision tower (select_layer=-2 → index 22)
    select_layer_idx = -2  # mm_vision_select_layer from config

    for batch in tqdm(dataloader, desc="Extracting LLaVA-Med Vision Layers"):
        if batch is None:
            continue

        images = batch['image'].to(device, dtype=torch.float16, non_blocking=True)
        batch_labels = [dict(zip(batch['labels'], t)) for t in zip(*batch['labels'].values())]

        with torch.no_grad():
            # Forward pass through CLIP ViT with all hidden states
            outputs = clip_model(pixel_values=images, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of (B, 577, 1024) for 25 states (embed + 24 layers)

            # Extract each hidden state: mean-pool over patch tokens (exclude CLS at index 0)
            for layer_idx, hs in enumerate(hidden_states):
                # hs shape: (B, 577, 1024) — first token is CLS, rest are patches
                patch_features = hs[:, 1:, :]  # (B, 576, 1024)
                pooled = patch_features.mean(dim=1).float().cpu().numpy()  # (B, 1024)

                if layer_idx == 0:
                    layer_name = 'Vis_CLIP_Embed'
                else:
                    layer_name = f'Vis_CLIP_Layer_{layer_idx}'

                all_layer_features[layer_name].append(pooled)

            # mm_projector: takes the selected layer's patch features
            # LLaVA uses hidden_states[select_layer_idx] with CLS removed
            selected_features = hidden_states[select_layer_idx][:, 1:, :]  # (B, 576, 1024)
            projected = mm_projector(selected_features.to(dtype=torch.float16))  # (B, 576, 4096)
            proj_pooled = projected.mean(dim=1).float().cpu().numpy()  # (B, 4096)
            all_layer_features['Vis_MM_Projector'].append(proj_pooled)

            all_labels_and_metadata.extend(batch_labels)

    if not all_layer_features:
        logging.error("No features were extracted. Aborting save.")
        return

    # Save
    num_processed = len(all_labels_and_metadata)
    logging.info(f"Concatenating and saving features for {num_processed} samples across {len(all_layer_features)} layers...")

    for layer_name, feat_list in all_layer_features.items():
        try:
            concatenated = np.concatenate(feat_list, axis=0)
            if concatenated.shape[0] != num_processed:
                logging.warning(f"  WARNING for '{layer_name}': Feature count ({concatenated.shape[0]}) != label count ({num_processed}). Skipping.")
                continue
            filepath = os.path.join(output_dir, f"{layer_name}_embeddings.npy")
            np.save(filepath, concatenated)
            logging.info(f"  Saved {layer_name}: shape {concatenated.shape}")
        except ValueError as e:
            logging.error(f"Could not save '{layer_name}': {e}")

    labels_df = pd.DataFrame(all_labels_and_metadata)
    labels_csv_path = os.path.join(output_dir, "labels_and_metadata.csv")
    labels_df.to_csv(labels_csv_path, index=False)
    logging.info(f"Saved labels for {len(labels_df)} samples to {labels_csv_path}")


# =============================================================================
# --- Main ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract VISION layer-wise features from LLaVA-Med 1.5 (CLIP ViT + mm_projector).")
    parser.add_argument('--dataset_folder_name', type=str, default='MIMIC-CXR-JPG',
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to process.")
    parser.add_argument('--split_value', type=int, required=True, choices=[0, 1, 2],
                        help="Split: 0=train, 1=val, 2=test.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU device ID to use (default: 0).")
    parser.add_argument("--final_only", action="store_true", help="Ignored for compatibility")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting")
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu_id >= 0:
        DEVICE = f"cuda:{args.gpu_id}"
    else:
        DEVICE = "cpu"

    dset_cfg = DATASET_CONFIGS[args.dataset_folder_name]
    split_name_map = {0: "train", 1: "val", 2: "test"}
    split_name = split_name_map[args.split_value]

    logging.info(f"--- LLaVA-Med 1.5 VISION Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

    # Output directory
    features_root = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, args.dataset_folder_name)
    output_dir = os.path.join(features_root, f"features_vision_only_{split_name}")

    if os.path.exists(output_dir) and os.listdir(output_dir) and not getattr(args, "final_only", False) and not getattr(args, "overwrite", False):
        logging.error(f"Output directory already exists and is non-empty: {output_dir}")
        logging.error("Refusing to overwrite. Use --overwrite or --final_only.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output: {output_dir}")

    # Load CLIP ViT + mm_projector (memory-efficient: ~1.5GB, no 7B LLM needed)
    clip_model, mm_projector, image_processor = load_vision_components(
        LLAVAMED_MODEL_PATH, CLIP_VISION_TOWER, DEVICE
    )

    # Prepare dataset
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = LLaVAMedVisionDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        base_image_dir=dset_cfg['base_image_dir'],
        image_processor=image_processor
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} samples. Some may be skipped if images fail to load.")

    extract_vision_layers_and_save(dataset, clip_model, mm_projector, DEVICE, output_dir)
    logging.info("--- LLaVA-Med 1.5 Vision Extraction Finished ---")
