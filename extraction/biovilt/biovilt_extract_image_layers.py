# biovilt_extract_image_layers.py
# Extracts layer-wise vision embeddings from the BioViL-T image encoder.
# Architecture: ResNet-50 backbone (4 stages) + backbone_to_vit Conv2d + MLP Projector
# NOTE: The ViT Pooler (temporal transformer) is NOT used in single-image mode.
#
# Output format matches the original CheXzero pipeline:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python biovilt_extract_image_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

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

# --- BioViL-T specific imports ---
from health_multimodal.image.model.model import ImageModel
from health_multimodal.image.model.encoder import MultiImageEncoder
from health_multimodal.image.model.modules import MLP
from health_multimodal.image.data.transforms import create_chest_xray_transform_for_inference

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_biovilt import (
    BIOVILT_IMAGE_WEIGHTS_PATH, BIOVILT_MODEL_DIR,
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    VISION_LAYER_DIMS, BIOVILT_IMAGE_RESIZE, BIOVILT_IMAGE_CENTER_CROP
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

OPTIMIZED_BATCH_SIZE = 64  # Lower than CheXzero since BioViL-T ResNet-50 uses more memory
OPTIMIZED_NUM_WORKERS = min(8, os.cpu_count() if os.cpu_count() else 1)


# =============================================================================
# --- Model Loading ---
# =============================================================================
def load_biovilt_image_model(weights_path, device):
    """Load the BioViL-T image model with pre-trained weights."""
    logging.info(f"Loading BioViL-T image model from: {weights_path}")

    # Reconstruct the architecture matching the checkpoint structure
    encoder = MultiImageEncoder(img_encoder_type="resnet50")
    projector = MLP(input_dim=512, output_dim=128, hidden_dim=128, use_1x1_convs=True)
    model = ImageModel(img_encoder_type="resnet50", joint_feature_size=128)
    model.encoder = encoder
    model.projector = projector

    # Load weights
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('state_dict', checkpoint)
    msg = model.load_state_dict(state_dict, strict=True)
    logging.info(f"Model weights loaded: {msg}")

    model.float().to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    logging.info(f"BioViL-T image model loaded to {device} and frozen.")
    return model


# =============================================================================
# --- Hook Function ---
# =============================================================================
def capture_vision_hook(module, input_val, output_val, layer_name, storage_dict):
    """
    Forward hook for BioViL-T vision layers.
    Handles different output shapes from ResNet stages and backbone_to_vit conv.
    """
    tensor = output_val[0] if isinstance(output_val, tuple) else output_val
    if not isinstance(tensor, torch.Tensor):
        return

    tensor = tensor.detach()

    # Both ResNet stages and backbone_to_vit output 4D: (B, C, H, W)
    if tensor.ndim == 4:
        features = F.adaptive_avg_pool2d(tensor, (1, 1)).flatten(start_dim=1)
    elif tensor.ndim == 2:
        features = tensor
    else:
        return

    storage_dict[layer_name] = features.cpu().numpy()


# =============================================================================
# --- Dataset ---
# =============================================================================
class BioViLTVisionDataset(Dataset):
    """Dataset for BioViL-T vision feature extraction using the library's own transforms."""

    def __init__(self, metadata_csv_path, target_split_value, base_image_dir):
        self.base_image_dir = base_image_dir
        full_df = pd.read_csv(metadata_csv_path)
        self.metadata_df = full_df[full_df["split"] == target_split_value].reset_index(drop=True)

        # Use BioViL-T's own image transform pipeline
        self.img_transform = create_chest_xray_transform_for_inference(
            resize=BIOVILT_IMAGE_RESIZE,
            center_crop_size=BIOVILT_IMAGE_CENTER_CROP
        )

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

            # BioViL-T expects grayscale -> 3-channel via ExpandChannels in the transform
            if img.mode != 'L':
                img = img.convert('L')

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
def extract_vision_layers_and_save(dataset, model, device, output_dir):
    """Extract features from all BioViL-T vision layers and save them."""

    logging.info("Registering hooks on BioViL-T vision layers...")

    # Define hook targets
    vision_layers_to_probe = OrderedDict()

    # ResNet backbone stages
    vision_layers_to_probe['Vis_ResNet_Layer1'] = model.encoder.encoder.layer1
    vision_layers_to_probe['Vis_ResNet_Layer2'] = model.encoder.encoder.layer2
    vision_layers_to_probe['Vis_ResNet_Layer3'] = model.encoder.encoder.layer3
    vision_layers_to_probe['Vis_ResNet_Layer4'] = model.encoder.encoder.layer4

    # backbone_to_vit Conv2d projection (2048 -> 256)
    vision_layers_to_probe['Vis_BackboneToViT'] = model.encoder.backbone_to_vit

    logging.info(f"Defined {len(vision_layers_to_probe)} intermediate vision layers to hook.")

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=collate_fn_robust,
        pin_memory=(device != "cpu")
    )

    for batch in tqdm(dataloader, desc="Extracting BioViL-T Vision Layers"):
        if batch is None:
            continue

        images = batch['image'].to(device, non_blocking=True)
        batch_labels = [dict(zip(batch['labels'], t)) for t in zip(*batch['labels'].values())]

        feature_storage_batch = {}
        with torch.no_grad():
            # Register hooks
            hook_handles = []
            for name, module in vision_layers_to_probe.items():
                hook_handles.append(
                    module.register_forward_hook(
                        partial(capture_vision_hook, layer_name=name, storage_dict=feature_storage_batch)
                    )
                )

            # Forward pass through the full image model
            output = model(images)

            # Remove hooks
            for handle in hook_handles:
                handle.remove()

            # Collect hooked features
            for layer_name, features in feature_storage_batch.items():
                all_layer_features[layer_name].append(features)

            # Collect the img_embedding (pre-projection, 512-dim)
            img_embedding = output.img_embedding.cpu().numpy()
            all_layer_features['img_embedding'].append(img_embedding)

            # Collect the final projected embedding (128-dim)
            final_embedding = output.projected_global_embedding.cpu().numpy()
            all_layer_features['image_embedding_final'].append(final_embedding)

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
    parser = argparse.ArgumentParser(description="Extract VISION layer-wise features from BioViL-T.")
    parser.add_argument('--dataset_folder_name', type=str, default='MIMIC-CXR-JPG',
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to process.")
    parser.add_argument('--split_value', type=int, required=True, choices=[0, 1, 2],
                        help="Split: 0=train, 1=val, 2=test.")
    parser.add_argument("--final_only", action="store_true", help="Ignored for compatibility")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting")
    args = parser.parse_args()

    dset_cfg = DATASET_CONFIGS[args.dataset_folder_name]
    split_name_map = {0: "train", 1: "val", 2: "test"}
    split_name = split_name_map[args.split_value]

    logging.info(f"--- BioViL-T VISION Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

    # Output directory
    features_root = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, args.dataset_folder_name)
    output_dir = os.path.join(features_root, f"features_vision_only_{split_name}")

    if os.path.exists(output_dir) and os.listdir(output_dir) and not getattr(args, "final_only", False) and not getattr(args, "overwrite", False):
        logging.error(f"Output directory already exists and is non-empty: {output_dir}")
        logging.error("Refusing to overwrite. Use --overwrite or --final_only.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output: {output_dir}")

    # Load model
    image_model = load_biovilt_image_model(BIOVILT_IMAGE_WEIGHTS_PATH, DEVICE)

    # Prepare dataset
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = BioViLTVisionDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        base_image_dir=dset_cfg['base_image_dir']
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} samples. Some may be skipped if images fail to load.")

    extract_vision_layers_and_save(dataset, image_model, DEVICE, output_dir)
    logging.info("--- BioViL-T Vision Extraction Finished ---")
