# chexagent_extract_image_layers.py
# Extracts layer-wise vision embeddings from CheXagent's XraySigLIP ViT-L/16 encoder.
# Architecture: SigLIP ViT-L/16 (24 layers, hidden_size=1024) + Resampler MLP (->2560)
#
# Output format matches the BioViL-T pipeline:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python chexagent_extract_image_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

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
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# --- CheXagent specific imports ---
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_chexagent import (
    CHEXAGENT_MODEL_NAME,
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    VISION_LAYER_DIMS, CHEXAGENT_IMAGE_SIZE
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

# CheXagent ViT-L/16 is large; use a moderate batch size
OPTIMIZED_BATCH_SIZE = 32
OPTIMIZED_NUM_WORKERS = min(8, os.cpu_count() if os.cpu_count() else 1)


# =============================================================================
# --- Model Loading ---
# =============================================================================
def load_chexagent_model(model_name, device):
    """Load the full CheXagent model and return the vision sub-model."""
    logging.info(f"Loading CheXagent model from: {model_name}")

    # Load the full CheXagent model (we only need the vision part)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # The vision encoder is at model.model.visual (a CLIPModel instance)
    visual_model = model.model.visual
    visual_model.to(device).eval()
    for param in visual_model.parameters():
        param.requires_grad = False

    logging.info(f"CheXagent vision encoder loaded to {device} and frozen.")
    logging.info(f"  ViT layers: {len(visual_model.model.encoder.layers)}")
    logging.info(f"  Hidden size: {visual_model.model.config.hidden_size}")
    logging.info(f"  Normalization: mean={visual_model.mean}, std={visual_model.std}")

    return visual_model


# =============================================================================
# --- Hook Function ---
# =============================================================================
def capture_vision_hook(module, input_val, output_val, layer_name, storage_dict):
    """
    Forward hook for CheXagent ViT layers.
    SigLIP ViT encoder layers output tuples: (hidden_states, ...).
    Each hidden_states is (B, SeqLen, Dim). We mean-pool over SeqLen.
    """
    if isinstance(output_val, tuple):
        tensor = output_val[0]
    else:
        tensor = output_val

    if not isinstance(tensor, torch.Tensor):
        return

    tensor = tensor.detach()

    # ViT layers output (B, SeqLen, Dim) -> mean pool to (B, Dim)
    if tensor.ndim == 3:
        features = tensor.mean(dim=1)
    elif tensor.ndim == 2:
        features = tensor
    elif tensor.ndim == 4:
        # Unlikely for ViT but handle just in case
        features = F.adaptive_avg_pool2d(tensor, (1, 1)).flatten(start_dim=1)
    else:
        return

    storage_dict[layer_name] = features.cpu().numpy()


# =============================================================================
# --- Dataset ---
# =============================================================================
class CheXagentVisionDataset(Dataset):
    """Dataset for CheXagent vision feature extraction using XraySigLIP's transforms."""

    def __init__(self, metadata_csv_path, target_split_value, base_image_dir, image_size,
                 norm_mean=None, norm_std=None):
        self.base_image_dir = base_image_dir
        full_df = pd.read_csv(metadata_csv_path)
        self.metadata_df = full_df[full_df["split"] == target_split_value].reset_index(drop=True)

        # CheXagent CLIPModel uses: Resize(512,512) + ToTensor + Normalize
        # Normalization values come from the XraySigLIP processor (default [0.5,0.5,0.5]).
        if norm_mean is None:
            norm_mean = [0.5, 0.5, 0.5]
        if norm_std is None:
            norm_std = [0.5, 0.5, 0.5]
        self.img_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
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

            # CheXagent expects RGB images
            if img.mode != 'RGB':
                img = img.convert('RGB')

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
def extract_vision_layers_and_save(dataset, visual_model, device, output_dir):
    """Extract features from all CheXagent vision layers and save them."""

    logging.info("Registering hooks on CheXagent ViT layers...")

    # Define hook targets: SigLIP ViT encoder layers
    vision_layers_to_probe = OrderedDict()
    for i, layer in enumerate(visual_model.model.encoder.layers):
        vision_layers_to_probe[f'Vis_SigLIP_Block_{i+1}'] = layer

    logging.info(f"Defined {len(vision_layers_to_probe)} intermediate ViT layers to hook.")

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=collate_fn_robust,
        pin_memory=(device != "cpu")
    )

    for batch in tqdm(dataloader, desc="Extracting CheXagent Vision Layers"):
        if batch is None:
            continue

        images = batch['image'].to(device, non_blocking=True)
        batch_labels = [dict(zip(batch['labels'], t)) for t in zip(*batch['labels'].values())]

        feature_storage_batch = {}
        with torch.no_grad():
            # Register hooks on ViT layers
            hook_handles = []
            for name, module in vision_layers_to_probe.items():
                hook_handles.append(
                    module.register_forward_hook(
                        partial(capture_vision_hook, layer_name=name, storage_dict=feature_storage_batch)
                    )
                )

            # Forward pass through the full vision model (CLIPModel.forward)
            # This runs: ViT -> resampler (attn_pool -> ln_post -> proj)
            final_output = visual_model(images)  # (B, 1024, 2560)

            # Remove hooks
            for handle in hook_handles:
                handle.remove()

            # Collect hooked ViT layer features
            for layer_name, features in feature_storage_batch.items():
                all_layer_features[layer_name].append(features)

            # Collect the final resampled embedding (2560-dim)
            # final_output shape: (B, num_patches, 2560) -> mean-pool -> (B, 2560)
            final_embedding = final_output.mean(dim=1).cpu().numpy()
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
    parser = argparse.ArgumentParser(description="Extract VISION layer-wise features from CheXagent.")
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

    logging.info(f"--- CheXagent VISION Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

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
    visual_model = load_chexagent_model(CHEXAGENT_MODEL_NAME, DEVICE)

    # Prepare dataset (use normalization from the loaded model's processor)
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = CheXagentVisionDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        base_image_dir=dset_cfg['base_image_dir'],
        image_size=CHEXAGENT_IMAGE_SIZE,
        norm_mean=visual_model.mean,
        norm_std=visual_model.std
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} samples. Some may be skipped if images fail to load.")

    extract_vision_layers_and_save(dataset, visual_model, DEVICE, output_dir)
    logging.info("--- CheXagent Vision Extraction Finished ---")
