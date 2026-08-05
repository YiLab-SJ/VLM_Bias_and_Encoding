# chexzero_extract_image_layers.py
# Extracts layer-wise vision embeddings from CheXzero's CLIP ViT-B/32 encoder.
# Architecture: 12 ResidualAttentionBlock layers (hidden_size=768) + proj (768->512)
#
# Output format matches the pipeline convention:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python chexzero_extract_image_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import sys
import random
import argparse
import logging
from collections import OrderedDict, defaultdict
from functools import partial
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_chexzero import (
    CHEXZERO_MODEL_PATH, CHEXZERO_SRC_DIR,
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    VISION_LAYER_DIMS, CHEXZERO_IMAGE_SIZE, CLIP_MEAN, CLIP_STD
)

# --- CheXzero model imports ---
sys.path.insert(0, CHEXZERO_SRC_DIR)
from model import build_model

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

OPTIMIZED_BATCH_SIZE = 64
OPTIMIZED_NUM_WORKERS = min(8, os.cpu_count() if os.cpu_count() else 1)


# =============================================================================
# --- Model Loading ---
# =============================================================================
def load_chexzero_model(model_path, device):
    """Load the CheXzero model from a state_dict checkpoint."""
    logging.info(f"Loading CheXzero model from: {model_path}")

    state_dict = torch.load(model_path, map_location="cpu")
    model = build_model(state_dict)
    model.float()
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False

    logging.info(f"CheXzero model loaded to {device} and frozen.")
    logging.info(f"  Vision: VisualTransformer, {len(model.visual.transformer.resblocks)} layers, width=768")
    logging.info(f"  Vision proj: {model.visual.proj.shape}")
    logging.info(f"  Vision input_resolution: {model.visual.input_resolution}")

    return model


# =============================================================================
# --- Hook Function ---
# =============================================================================
def capture_vision_hook(module, input_val, output_val, layer_name, storage_dict):
    """
    Forward hook for CheXzero ViT ResidualAttentionBlock layers.
    CheXzero's ViT processes in LND (Length, Batch, Dim) format.
    Each resblock output: (SeqLen, B, 768).
    We permute to (B, SeqLen, 768) and mean-pool over SeqLen to get (B, 768).
    """
    tensor = output_val
    if isinstance(tensor, tuple):
        tensor = tensor[0]

    if not isinstance(tensor, torch.Tensor):
        return

    tensor = tensor.detach()

    # LND format: (SeqLen, B, Dim) -> permute to (B, SeqLen, Dim)
    if tensor.ndim == 3:
        tensor = tensor.permute(1, 0, 2)  # (B, SeqLen, Dim)
        features = tensor.mean(dim=1)      # (B, Dim) mean-pool over spatial+CLS
    elif tensor.ndim == 2:
        features = tensor
    else:
        return

    storage_dict[layer_name] = features.cpu().numpy()


# =============================================================================
# --- Dataset ---
# =============================================================================
class CheXzeroVisionDataset(Dataset):
    """Dataset for CheXzero vision feature extraction using CLIP preprocessing."""

    def __init__(self, metadata_csv_path, target_split_value, base_image_dir, image_size):
        self.base_image_dir = base_image_dir
        full_df = pd.read_csv(metadata_csv_path)
        self.metadata_df = full_df[full_df["split"] == target_split_value].reset_index(drop=True)

        self.img_transform = Compose([
            Resize(image_size, interpolation=Image.BICUBIC),
            CenterCrop(image_size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(CLIP_MEAN, CLIP_STD),
        ])

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
            # Fix for 16-bit PNGs (RexGradient): normalize to 8-bit before transforms
            if img.mode in ('I;16', 'I'):
                import numpy as _np
                arr = _np.array(img, dtype=_np.float32)
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = (arr - mn) / (mx - mn) * 255.0
                else:
                    arr = _np.zeros_like(arr)
                img = Image.fromarray(arr.astype(_np.uint8), mode='L')
            img_tensor = self.img_transform(img)
            return {'image': img_tensor, 'labels': data_row.to_dict()}
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
def extract_vision_layers_and_save(dataset, model, device, output_dir, final_only=False):
    """Extract features from CheXzero vision layers and save them."""

    logging.info("Registering hooks on CheXzero ViT layers...")

    # Define hook targets: ViT ResidualAttentionBlock layers
    vision_layers_to_probe = OrderedDict()
    if not final_only:
        for i, block in enumerate(model.visual.transformer.resblocks):
            vision_layers_to_probe[f'Vis_Block_{i+1}'] = block
        logging.info(f"Defined {len(vision_layers_to_probe)} intermediate ViT layers to hook.")
    else:
        logging.info("  final_only mode: skipping intermediate layer hooks.")

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=collate_fn_robust,
        pin_memory=(device != "cpu")
    )

    for batch in tqdm(dataloader, desc="Extracting CheXzero Vision Layers"):
        if batch is None:
            continue

        images = batch['image'].to(device, non_blocking=True)
        batch_labels = [dict(zip(batch['labels'], t)) for t in zip(*batch['labels'].values())]

        feature_storage_batch = {}
        with torch.no_grad():
            # Register hooks on ViT resblocks (only if not final_only)
            hook_handles = []
            if not final_only:
                for name, module in vision_layers_to_probe.items():
                    hook_handles.append(
                        module.register_forward_hook(
                            partial(capture_vision_hook, layer_name=name, storage_dict=feature_storage_batch)
                        )
                    )

            final_embedding = model.encode_image(images)  # (B, 512)

            for handle in hook_handles:
                handle.remove()

            if not final_only:
                for layer_name, features in feature_storage_batch.items():
                    all_layer_features[layer_name].append(features)

            all_layer_features['image_embedding_final'].append(final_embedding.cpu().numpy())
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
    parser = argparse.ArgumentParser(description="Extract VISION layer-wise features from CheXzero.")
    parser.add_argument('--dataset_folder_name', type=str, default='MIMIC-CXR-JPG',
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to process.")
    parser.add_argument('--split_value', type=int, required=True, choices=[0, 1, 2],
                        help="Split: 0=train, 1=val, 2=test.")
    parser.add_argument('--final_only', action='store_true',
                        help="Extract only the final image embedding (skip intermediate layers).")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing output directory.")
    args = parser.parse_args()

    dset_cfg = DATASET_CONFIGS[args.dataset_folder_name]
    split_name_map = {0: "train", 1: "val", 2: "test"}
    split_name = split_name_map[args.split_value]

    logging.info(f"--- CheXzero VISION Layer Extraction: {args.dataset_folder_name} - {split_name} ---")
    if args.final_only:
        logging.info("  MODE: final_only (only image_embedding_final)")

    # Output directory
    features_root = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, args.dataset_folder_name)
    output_dir = os.path.join(features_root, f"features_vision_only_{split_name}")

    if os.path.exists(output_dir) and os.listdir(output_dir) and not args.overwrite and not args.final_only:
        logging.error(f"Output directory already exists and is non-empty: {output_dir}")
        logging.error("Refusing to overwrite. Use --overwrite or --final_only to proceed.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output: {output_dir}")

    # Load model
    model = load_chexzero_model(CHEXZERO_MODEL_PATH, DEVICE)

    # Prepare dataset
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = CheXzeroVisionDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        base_image_dir=dset_cfg['base_image_dir'],
        image_size=CHEXZERO_IMAGE_SIZE
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} samples. Some may be skipped if images fail to load.")

    extract_vision_layers_and_save(dataset, model, DEVICE, output_dir, final_only=args.final_only)
    logging.info("--- CheXzero Vision Extraction Finished ---")
