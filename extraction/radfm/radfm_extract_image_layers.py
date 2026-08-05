# radfm_extract_image_layers.py
# Extracts layer-wise vision embeddings from RadFM's 3D ViT encoder.
# Architecture: 3D ViT (12 blocks, hidden=768) + Perceiver Resampler (6 layers, 32 latents)
#               + FC Projection (768 -> 5120)
#
# For 2D chest X-rays, images are resized to 512x512 and interpolated to depth=4.
# The ViT produces 256 patch tokens (16x16x1) of dimension 768.
#
# At each ViT block, we mean-pool across all 256 patch tokens to get a single
# 768-dimensional feature vector per image. We also capture the perceiver output
# (mean-pool over 32 latents -> 768) and the projected embedding (mean-pool -> 5120).
#
# Output format:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python radfm_extract_image_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

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
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from torchvision import transforms
from tqdm import tqdm

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_radfm import (
    RADFM_REPO_DIR, RADFM_CHECKPOINT_PATH, RADFM_LANGUAGE_FILES_PATH,
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    VISION_LAYER_DIMS, RADFM_IMAGE_SIZE, RADFM_DEPTH_SIZE
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

# RadFM with LLaMA-13B is ~26GB in fp32. Batch size must be conservative.
# Vision-only forward (just ViT + Perceiver) is lighter, but we load the full model.
OPTIMIZED_BATCH_SIZE = 16
OPTIMIZED_NUM_WORKERS = 4


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
def load_radfm_model(lang_model_path, checkpoint_path, device_str):
    """Load the full RadFM model from checkpoint.
    
    The Language_files directory only contains config.json and tokenizer files
    (no model weights). RadFM's MultiLLaMAForCausalLM.__init__ calls
    LlamaForCausalLM.from_pretrained() which fails without weights in
    transformers 4.28+. We work around this by temporarily monkey-patching
    from_pretrained to do config-only initialization, since all actual weights
    come from the RadFM checkpoint (pytorch_model.bin).
    """
    # Add Quick_demo to path so 'Model.RadFM' package resolves
    quick_demo_dir = os.path.join(RADFM_REPO_DIR, "Quick_demo")
    if quick_demo_dir not in sys.path:
        sys.path.insert(0, quick_demo_dir)

    from transformers import LlamaForCausalLM, LlamaConfig
    from Model.RadFM.multimodality_model import MultiLLaMAForCausalLM

    # Monkey-patch from_pretrained to build from config only (no weight files needed)
    _original_from_pretrained = LlamaForCausalLM.from_pretrained

    @classmethod
    def _config_only_init(cls, pretrained_model_name_or_path, *args, **kwargs):
        config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(config)

    LlamaForCausalLM.from_pretrained = _config_only_init

    logging.info(f"Initializing RadFM model structure from: {lang_model_path}")
    model = MultiLLaMAForCausalLM(lang_model_path=lang_model_path)

    # Restore original from_pretrained
    LlamaForCausalLM.from_pretrained = _original_from_pretrained

    logging.info(f"Loading RadFM checkpoint from: {checkpoint_path}")
    logging.info("This should be fast and memory-efficient with mmap=True...")
    
    # ADDED mmap=True AND weights_only=True to prevent CPU memory crash
    ckpt = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=True)
    
    model.load_state_dict(ckpt)
    del ckpt
    gc.collect()

    # Disable gradient checkpointing (enabled in RadFM's __init__) for clean inference
    if hasattr(model.lang_model, 'gradient_checkpointing_disable'):
        model.lang_model.gradient_checkpointing_disable()

    model = model.to(device_str)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logging.info(f"RadFM loaded on {device_str}.")
    return model


# =============================================================================
# --- Manual ViT Forward with Per-Layer Extraction ---
# =============================================================================
def manual_vit_forward_with_features(vision_encoder, images):
    """
    Manually replicate the ViT forward pass, collecting features at each stage.

    RadFM's ViT Transformer blocks are nn.ModuleList([PreNormAttn, PreNormFFN]),
    which have no forward() method and thus can't receive forward hooks.
    Instead, we step through the layers manually.

    Args:
        vision_encoder: model.embedding_layer.vision_encoder (ViT instance)
        images: (B, C, H, W, D) tensor

    Returns:
        features_dict: {layer_name: (B, dim) numpy array}
        vit_output: (B, num_patches, dim) — final transformer output
    """
    features_dict = {}
    B, C, H, W, D = images.shape

    # 1. Patch embedding: (B, C, H, W, D) -> (B, num_patches, dim)
    x = vision_encoder.to_patch_embedding(images)
    features_dict['Vis_PatchEmbed'] = x.detach().float().mean(dim=1).cpu().numpy()

    # 2. Add positional encoding and dropout
    pos = vision_encoder.pos_embedding(
        B,
        H // vision_encoder.patch_height,
        W // vision_encoder.patch_width,
        D // vision_encoder.frame_patch_size,
        x
    )
    x = x + pos
    x = vision_encoder.dropout(x)

    # 3. Transformer blocks (12 blocks, each is ModuleList[PreNormAttn, PreNormFFN])
    for i, (attn, ff) in enumerate(vision_encoder.transformer.layers):
        x = attn(x) + x
        x = ff(x) + x
        # Mean-pool across patches -> (B, dim)
        features_dict[f'Vis_Block_{i + 1}'] = x.detach().float().mean(dim=1).cpu().numpy()

    return features_dict, x


# =============================================================================
# --- Dataset ---
# =============================================================================
class RadFMVisionDataset(Dataset):
    """Dataset for RadFM vision feature extraction.
    
    Images are preprocessed as in RadFM's inference pipeline:
    - Resize to 512x512
    - Convert to tensor
    - Unsqueeze to (C, H, W, 1) then interpolate to (C, H, W, 4)
    
    We use deterministic Resize (not RandomResizedCrop) for reproducibility.
    """
    def __init__(self, metadata_csv_path, target_split_value, base_image_dir,
                 num_shards=1, shard_idx=0):
        self.base_image_dir = base_image_dir
        self.transform = transforms.Compose([
            transforms.Resize((RADFM_IMAGE_SIZE, RADFM_IMAGE_SIZE),
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
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

            # Transform to tensor: (3, 512, 512)
            img_tensor = self.transform(img)

            # Convert 2D -> 3D: (3, 512, 512, 1) -> (3, 512, 512, 4)
            img_tensor = img_tensor.unsqueeze(-1)  # (C, H, W, 1)
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),  # (1, C, H, W, 1)
                size=(RADFM_IMAGE_SIZE, RADFM_IMAGE_SIZE, RADFM_DEPTH_SIZE),
                mode='trilinear',
                align_corners=False
            ).squeeze(0)  # (C, H, W, D)

            return {'image': img_tensor, 'labels': data_row.to_dict()}
        except Exception as e:
            logging.warning(f"Failed to load image {relative_img_path}: {e}")
            return None


def collate_fn_robust(batch_list):
    """Filter None entries and collate into a batch."""
    batch_list = [item for item in batch_list if item is not None]
    if not batch_list:
        return None

    images = torch.stack([item['image'] for item in batch_list])  # (B, C, H, W, D)

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
# --- Main Extraction ---
# =============================================================================
def extract_vision_layers_and_save(dataset, model, device_str, output_dir):
    """Extract features from all RadFM vision layers and save them.

    Uses manual forward pass through the ViT because transformer blocks are
    nn.ModuleList objects (not nn.Module with forward()) and can't receive hooks.
    """
    logging.info("Setting up manual layer-by-layer extraction for RadFM 3D ViT...")

    # Access the vision encoder and projection layers
    vision_encoder = model.embedding_layer.vision_encoder
    perceiver = model.embedding_layer.perceiver
    fc_proj = model.embedding_layer.fc

    num_blocks = len(vision_encoder.transformer.layers)
    logging.info(f"ViT has {num_blocks} transformer blocks.")
    logging.info("Will also extract Perceiver (32 latents -> 768) and Projected (-> 5120).")

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

    for batch in tqdm(dataloader, desc="Extracting RadFM Vision Layers"):
        if batch is None:
            continue

        # images: (B, C, H, W, D) = (B, 3, 512, 512, 4)
        images = batch['image'].to(device_str, non_blocking=True)
        batch_labels = [dict(zip(batch['labels'], t)) for t in zip(*batch['labels'].values())]
        B = images.shape[0]

        with torch.no_grad():
            # --- Manual ViT forward with per-layer extraction ---
            layer_features, vit_output = manual_vit_forward_with_features(
                vision_encoder, images)

            # Collect all ViT layer features
            for layer_name, features in layer_features.items():
                all_layer_features[layer_name].append(features)

            # --- Perceiver: (B, 256, 768) -> (B, 1, 1, 256, 768) -> (B, 1, 32, 768) ---
            perceiver_input = vit_output.unsqueeze(1).unsqueeze(2)
            perceiver_output = perceiver(perceiver_input)
            perceiver_pooled = perceiver_output.squeeze(1).mean(dim=1).float().cpu().numpy()
            all_layer_features['Vis_Perceiver'].append(perceiver_pooled)

            # --- FC Projection: (B, 32, 768) -> (B, 32, 5120) -> mean-pool -> (B, 5120) ---
            proj_input = perceiver_output.squeeze(1)  # (B, 32, 768)
            proj_input_flat = proj_input.reshape(-1, proj_input.shape[-1])
            projected = fc_proj(proj_input_flat)
            projected = projected.reshape(B, -1, projected.shape[-1])
            projected_pooled = projected.mean(dim=1).float().cpu().numpy()
            all_layer_features['Vis_Projected'].append(projected_pooled)

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
    all_target_layers = ['Vis_PatchEmbed']
    all_target_layers += [f'Vis_Block_{i+1}' for i in range(num_blocks)]
    all_target_layers += ['Vis_Perceiver', 'Vis_Projected']
    combine_chunks(chunks_dir, output_dir, all_target_layers)


# =============================================================================
# --- Main ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract VISION layer-wise features from RadFM's 3D ViT.")
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

    logging.info(f"--- RadFM VISION Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

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

    # Load the full RadFM model
    model = load_radfm_model(RADFM_LANGUAGE_FILES_PATH, RADFM_CHECKPOINT_PATH, DEVICE)

    # Prepare dataset
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = RadFMVisionDataset(
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
    logging.info("--- RadFM Vision Extraction Finished ---")