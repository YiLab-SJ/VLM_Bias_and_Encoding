# medversa_extract_image_layers.py
# Extracts layer-wise vision embeddings from MedVersa's Swin-Base encoder.
# Architecture: Swin-B (4 stages, [2,2,18,2] blocks, dims=[128,256,512,1024])
#               + ln_vision_2d (LN, 1024) + llama_proj_2d (Linear 1024->4096)
#
# For 2D chest X-rays, images are resized to 224x224 and normalized with
# CLIP mean/std. The Swin-B produces 49 spatial tokens (7x7) of dim 1024
# at the final stage.
#
# At each Swin block, we mean-pool across spatial tokens to get a single
# feature vector per image. We also capture:
#   - Vis_LNVision:  ln_vision_2d applied to pooled Swin output -> (B, 1024)
#   - Vis_Projected: llama_proj_2d output -> (B, 4096)
#
# Output format:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python medversa_extract_image_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import sys
import random
import argparse
import logging
import math
import gc
import glob
import shutil
from collections import OrderedDict, defaultdict
from functools import partial
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_medversa import (
    MEDVERSA_REPO_DIR,
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    VISION_LAYER_DIMS, MEDVERSA_IMAGE_SIZE,
    MEDVERSA_IMAGE_MEAN, MEDVERSA_IMAGE_STD,
    SWIN_STAGE_DEPTHS, SWIN_STAGE_DIMS
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

# MedVersa full model ~15 GB in bf16. Swin-B is tiny (~88M params).
# 224x224 images are small. Batch size can be generous.
OPTIMIZED_BATCH_SIZE = 64
OPTIMIZED_NUM_WORKERS = 8


# =============================================================================
# --- Helper: Chunk Management ---
# =============================================================================
def save_chunk(chunk_dir, chunk_idx, features_dict, metadata_list):
    """Saves a partial chunk of data to disk to free up RAM."""
    os.makedirs(chunk_dir, exist_ok=True)

    df = pd.DataFrame(metadata_list)
    df.to_csv(os.path.join(chunk_dir, f"metadata_{chunk_idx:04d}.csv"), index=False)

    for layer_name, feat_list in features_dict.items():
        concatenated = np.concatenate(feat_list, axis=0)
        np.save(os.path.join(chunk_dir, f"{layer_name}_{chunk_idx:04d}.npy"), concatenated)


def combine_chunks(chunk_dir, output_dir, layer_names):
    """Stitches all saved chunks back together into the final output format."""
    logging.info("Combining all chunks into final files. This will take a moment...")

    meta_files = sorted(glob.glob(os.path.join(chunk_dir, "metadata_*.csv")))
    if meta_files:
        dfs = [pd.read_csv(f) for f in meta_files]
        final_df = pd.concat(dfs, ignore_index=True)
        final_df.to_csv(os.path.join(output_dir, "labels_and_metadata.csv"), index=False)
        logging.info(f"Saved combined metadata for {len(final_df)} samples.")

    for layer_name in layer_names:
        feat_files = sorted(glob.glob(os.path.join(chunk_dir, f"{layer_name}_*.npy")))
        if not feat_files:
            logging.warning(f"No chunk files found for layer: {layer_name}")
            continue
        arrays = [np.load(f) for f in feat_files]
        combined = np.concatenate(arrays, axis=0)
        np.save(os.path.join(output_dir, f"{layer_name}_embeddings.npy"), combined)
        logging.info(f"  Saved final {layer_name}: shape {combined.shape}")

    logging.info("Cleaning up temporary chunks directory...")
    shutil.rmtree(chunk_dir)


# =============================================================================
# --- Model Loading ---
# =============================================================================
def load_medversa_model(repo_dir, device_str):
    """Load the full MedVersa (MedOmni) model from the local repository.

    Uses PyTorchModelHubMixin.from_pretrained with the local directory containing
    config.json and model.safetensors.  The __init__ downloads base weights for
    Swin-B and LLaMA-2-7B (cached by HuggingFace) before the safetensors
    checkpoint overwrites them with MedVersa's fine-tuned weights.
    """
    # Ensure the medomni package is importable
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from medomni.common.registry import registry

    model_cls = registry.get_model_class('medomni')
    logging.info(f"Loading MedVersa from: {repo_dir}")
    logging.info("This downloads base Swin-B + LLaMA-2-7B (if not cached), "
                 "then loads MedVersa checkpoint. May take a few minutes...")

    model = model_cls.from_pretrained(repo_dir)
    model = model.to(device_str)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logging.info(f"MedVersa loaded on {device_str}.")
    return model


# =============================================================================
# --- Hook Function ---
# =============================================================================
def capture_vision_hook(module, input_val, output_val, layer_name, storage_dict):
    """
    Forward hook for Swin layers.

    SwinLayer  -> output is tuple (hidden_states, ...) or just hidden_states
    SwinEmbeddings -> output is (embeddings, output_dimensions)
    LayerNorm  -> output is a tensor
    """
    if isinstance(output_val, tuple):
        tensor = output_val[0]
    else:
        tensor = output_val

    if not isinstance(tensor, torch.Tensor):
        return

    tensor = tensor.detach().float()

    if tensor.ndim == 3:
        # (B, num_tokens, dim) -> mean-pool -> (B, dim)
        features = tensor.mean(dim=1)
    elif tensor.ndim == 2:
        features = tensor
    else:
        return

    storage_dict[layer_name] = features.cpu().numpy()


# =============================================================================
# --- Dataset ---
# =============================================================================
class MedVersaVisionDataset(Dataset):
    """Dataset for MedVersa vision feature extraction.

    Images are preprocessed following MedVersa's CXR pipeline:
    - Resize to 224x224
    - ToTensor (0-1 range)
    - Normalize with CLIP mean/std
    """
    def __init__(self, metadata_csv_path, target_split_value, base_image_dir,
                 num_shards=1, shard_idx=0):
        self.base_image_dir = base_image_dir
        self.transform = transforms.Compose([
            transforms.Resize([MEDVERSA_IMAGE_SIZE, MEDVERSA_IMAGE_SIZE]),
            transforms.ToTensor(),
            transforms.Normalize(MEDVERSA_IMAGE_MEAN, MEDVERSA_IMAGE_STD),
        ])

        full_df = pd.read_csv(metadata_csv_path)
        self.metadata_df = full_df[full_df["split"] == target_split_value].reset_index(drop=True)

        # --- Data Sharding Logic ---
        if num_shards > 1:
            total_len = len(self.metadata_df)
            shard_size = math.ceil(total_len / num_shards)
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, total_len)
            self.metadata_df = self.metadata_df.iloc[start_idx:end_idx].reset_index(drop=True)
            logging.info(f"SHARDING ACTIVE: shard {shard_idx + 1}/{num_shards} "
                         f"(Samples {start_idx} to {end_idx - 1})")

        if self.metadata_df.empty:
            logging.warning(f"No data found for split {target_split_value}.")

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
            img = img.convert('RGB')
            img_tensor = self.transform(img)  # (3, 224, 224)

            return {'image': img_tensor, 'labels': data_row.to_dict()}
        except Exception as e:
            logging.warning(f"Failed to load image {relative_img_path}: {e}")
            return None


def collate_fn_robust(batch_list):
    """Filter None entries and collate into a batch."""
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
# --- Projected Embedding Computation ---
# =============================================================================
def compute_projected_embeddings(model, swin_last_hidden, device):
    """
    Replicate MedVersa's encode_img projection pipeline for CXR images.

    Input:  swin_last_hidden  (B, 49, 1024) — Swin-B final output
    Steps:
      1. Reshape to (B, 1024, 7, 7)
      2. AdaptiveAvgPool2d to (B, 1024, 3, 3)
      3. Reshape to (B, 9, 1024)
      4. ln_vision_2d -> (B, 9, 1024)
      5. llama_proj_2d -> (B, 9, 4096)
    Returns:
      ln_vision_pooled  (B, 1024) — mean-pool of step 4
      projected_pooled  (B, 4096) — mean-pool of step 5
    """
    B = swin_last_hidden.shape[0]

    # Reshape: (B, 49, 1024) -> (B, 1024, 7, 7)
    image_embeds_unp = swin_last_hidden.permute(0, 2, 1).view(B, -1, 7, 7)
    # Pool: (B, 1024, 7, 7) -> (B, 1024, 3, 3)
    image_embeds_unp = F.adaptive_avg_pool2d(image_embeds_unp, (3, 3))
    # Reshape: (B, 1024, 3, 3) -> (B, 9, 1024)
    image_embeds = image_embeds_unp.view(B, -1, 9).permute(0, 2, 1)

    # MedVersa's ln_vision_2d: (B, 9, 1024)
    ln_out = model.ln_vision_2d(image_embeds)
    ln_vision_pooled = ln_out.mean(dim=1).float().cpu().numpy()  # (B, 1024)

    # MedVersa's llama_proj_2d: (B, 9, 4096)
    projected = model.llama_proj_2d(ln_out)
    projected_pooled = projected.mean(dim=1).float().cpu().numpy()  # (B, 4096)

    return ln_vision_pooled, projected_pooled


# =============================================================================
# --- Main Extraction ---
# =============================================================================
def extract_vision_layers_and_save(dataset, model, device_str, output_dir):
    """Extract features from all MedVersa Swin-B vision layers and save them."""
    logging.info("Setting up hooks on Swin-B vision layers...")

    swin_model = model.visual_encoder_2d

    # Enumerate all layers to probe
    vision_layers_to_probe = OrderedDict()

    # 1. Swin embeddings output
    vision_layers_to_probe['Vis_Embed'] = swin_model.embeddings

    # 2. Individual SwinLayer blocks within each stage
    for stage_idx, stage in enumerate(swin_model.encoder.layers):
        for block_idx, block in enumerate(stage.blocks):
            layer_name = f'Vis_S{stage_idx}_B{block_idx + 1}'
            vision_layers_to_probe[layer_name] = block

    # 3. Swin final LayerNorm
    vision_layers_to_probe['Vis_FinalNorm'] = swin_model.layernorm

    logging.info(f"Defined {len(vision_layers_to_probe)} Swin layers to hook.")
    logging.info("Will also compute Vis_LNVision and Vis_Projected manually.")

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []

    chunk_size = 20000
    chunk_idx = 0
    chunks_dir = os.path.join(output_dir, "temp_chunks")

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=collate_fn_robust,
        pin_memory=True
    )

    for batch in tqdm(dataloader, desc="Extracting MedVersa Vision Layers"):
        if batch is None:
            continue

        images = batch['image'].to(device_str, non_blocking=True)
        batch_labels = [dict(zip(batch['labels'], t)) for t in zip(*batch['labels'].values())]

        feature_storage_batch = {}
        with torch.no_grad():
            # Register hooks
            hook_handles = []
            for name, module in vision_layers_to_probe.items():
                hook_handles.append(
                    module.register_forward_hook(
                        partial(capture_vision_hook, layer_name=name,
                                storage_dict=feature_storage_batch)
                    )
                )

            # Forward pass through Swin-B
            # MedVersa uses fp16; cast input for consistency
            swin_output = swin_model(images.to(torch.float16 if next(swin_model.parameters()).dtype == torch.float16 else torch.float32))

            # Remove hooks
            for handle in hook_handles:
                handle.remove()

            # swin_output.last_hidden_state: (B, 49, 1024)
            last_hidden = swin_output.last_hidden_state

            # Compute projected embeddings manually
            ln_vision_pooled, projected_pooled = compute_projected_embeddings(
                model, last_hidden, device_str)
            feature_storage_batch['Vis_LNVision'] = ln_vision_pooled
            feature_storage_batch['Vis_Projected'] = projected_pooled

            # Collect all features
            for layer_name, features in feature_storage_batch.items():
                all_layer_features[layer_name].append(features)

            all_labels_and_metadata.extend(batch_labels)

        # --- CHUNKING TRIGGER ---
        if len(all_labels_and_metadata) >= chunk_size:
            logging.info(f"\nReached chunk size limit. Saving Chunk {chunk_idx:04d} to disk...")
            save_chunk(chunks_dir, chunk_idx, all_layer_features, all_labels_and_metadata)
            chunk_idx += 1
            all_layer_features = defaultdict(list)
            all_labels_and_metadata = []
            gc.collect()

    # Save final partial chunk
    if len(all_labels_and_metadata) > 0:
        logging.info(f"\nSaving final partial Chunk {chunk_idx:04d} to disk...")
        save_chunk(chunks_dir, chunk_idx, all_layer_features, all_labels_and_metadata)
        all_layer_features = defaultdict(list)
        all_labels_and_metadata = []
        gc.collect()

    # Build list of all expected layer names
    all_target_layers = list(vision_layers_to_probe.keys()) + ['Vis_LNVision', 'Vis_Projected']
    combine_chunks(chunks_dir, output_dir, all_target_layers)


# =============================================================================
# --- Main ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract VISION layer-wise features from MedVersa's Swin-B encoder.")
    parser.add_argument(
        '--dataset_folder_name', type=str, default='MIMIC-CXR-JPG',
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to process."
    )
    parser.add_argument(
        '--split_value', type=int, required=True, choices=[0, 1, 2],
        help="Split: 0=train, 1=val, 2=test."
    )
    parser.add_argument('--num_shards', type=int, default=1,
                        help="Total shards to split dataset into")
    parser.add_argument('--shard_idx', type=int, default=0,
                        help="Index of this current shard")
    parser.add_argument("--final_only", action="store_true", help="Ignored for compatibility")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting")
    args = parser.parse_args()

    dset_cfg = DATASET_CONFIGS[args.dataset_folder_name]
    split_name_map = {0: "train", 1: "val", 2: "test"}
    split_name = split_name_map[args.split_value]

    logging.info(f"--- MedVersa VISION Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

    features_root = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, args.dataset_folder_name)

    # Adjust output directory if sharding
    if args.num_shards > 1:
        output_dir = os.path.join(features_root,
                                  f"features_vision_only_{split_name}_shard_{args.shard_idx}")
    else:
        output_dir = os.path.join(features_root, f"features_vision_only_{split_name}")

    if os.path.exists(output_dir) and os.listdir(output_dir) and not getattr(args, "final_only", False) and not getattr(args, "overwrite", False):
        logging.error(f"Output directory already exists and is non-empty: {output_dir}")
        logging.error("Refusing to overwrite. Use --overwrite or --final_only.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output: {output_dir}")

    # Load the full MedVersa model
    model = load_medversa_model(MEDVERSA_REPO_DIR, DEVICE)

    # Prepare dataset
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = MedVersaVisionDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        base_image_dir=dset_cfg['base_image_dir'],
        num_shards=args.num_shards,
        shard_idx=args.shard_idx
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} samples.")
    extract_vision_layers_and_save(dataset, model, DEVICE, output_dir)
    logging.info("--- MedVersa Vision Extraction Finished ---")
