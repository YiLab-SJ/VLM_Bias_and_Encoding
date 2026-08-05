# medgemma_extract_image_layers.py
# Extracts layer-wise vision embeddings from MedGemma 1.5's SigLIP vision encoder.
# Architecture: SigLIP ViT (27 layers, hidden=1152) + Multi-modal Projector (-> 2560)

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
import shutil
from collections import OrderedDict, defaultdict
from functools import partial
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFile
from tqdm import tqdm

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_medgemma import (
    MEDGEMMA_MODEL_PATH,
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    VISION_LAYER_DIMS, MEDGEMMA_IMAGE_SIZE
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

OPTIMIZED_BATCH_SIZE = 32
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
    import glob
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
def load_medgemma_model(model_path, device_str):
    from transformers import Gemma3ForConditionalGeneration
    logging.info(f"Loading MedGemma 1.5 from: {model_path}")
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

def load_image_processor(model_path):
    from transformers import AutoImageProcessor
    logging.info(f"Loading Gemma3ImageProcessor from: {model_path}")
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    return image_processor

# =============================================================================
# --- Hook Function ---
# =============================================================================
def capture_vision_hook(module, input_val, output_val, layer_name, storage_dict):
    if isinstance(output_val, tuple):
        tensor = output_val[0]
    else:
        tensor = output_val

    if not isinstance(tensor, torch.Tensor):
        return

    tensor = tensor.detach().float()

    if tensor.ndim == 3:
        features = tensor.mean(dim=1)
    elif tensor.ndim == 2:
        features = tensor
    else:
        return

    storage_dict[layer_name] = features.cpu().numpy()

# =============================================================================
# --- Dataset ---
# =============================================================================
class MedGemmaVisionDataset(Dataset):
    def __init__(self, metadata_csv_path, target_split_value, base_image_dir, image_processor, num_shards=1, shard_idx=0):
        self.base_image_dir = base_image_dir
        self.image_processor = image_processor
        full_df = pd.read_csv(metadata_csv_path)
        self.metadata_df = full_df[full_df["split"] == target_split_value].reset_index(drop=True)

        # --- Data Sharding Logic ---
        if num_shards > 1:
            total_len = len(self.metadata_df)
            shard_size = math.ceil(total_len / num_shards)
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, total_len)
            self.metadata_df = self.metadata_df.iloc[start_idx:end_idx].reset_index(drop=True)
            logging.info(f"SHARDING ACTIVE: Processing shard {shard_idx + 1}/{num_shards} (Samples {start_idx} to {end_idx-1})")

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

            if img.mode != 'RGB':
                img = img.convert('RGB')

            processed = self.image_processor(images=img, return_tensors='pt')
            pixel_values = processed['pixel_values'].squeeze(0)

            return {'image': pixel_values, 'labels': data_row.to_dict()}
        except Exception as e:
            logging.warning(f"Failed to load image {relative_img_path}: {e}")
            return None

def collate_fn_robust(batch_list):
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

def compute_projected_embedding(model, vision_last_hidden_state):
    projector = model.model.multi_modal_projector
    hidden = vision_last_hidden_state.detach()

    batch_size, num_tokens, dim = hidden.shape
    h = w = int(math.sqrt(num_tokens)) 

    hidden = hidden.reshape(batch_size, h, w, dim).permute(0, 3, 1, 2)
    hidden = projector.avg_pool(hidden)
    hidden = hidden.flatten(2).transpose(1, 2)
    hidden = projector.mm_soft_emb_norm(hidden)
    hidden = torch.matmul(hidden, projector.mm_input_projection_weight)
    
    projected = hidden.mean(dim=1).float().cpu().numpy()
    return projected

def extract_vision_layers_and_save(dataset, model, device_str, output_dir):
    logging.info("Setting up hooks on SigLIP vision layers...")

    vision_tower = model.model.vision_tower.vision_model
    vision_layers_to_probe = OrderedDict()
    vision_layers_to_probe['Vis_Embed'] = vision_tower.embeddings

    for i, layer in enumerate(vision_tower.encoder.layers):
        vision_layers_to_probe[f'Vis_Block_{i+1}'] = layer

    vision_layers_to_probe['Vis_PostNorm'] = vision_tower.post_layernorm
    logging.info(f"Defined {len(vision_layers_to_probe)} vision layers to hook.")

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []

    chunk_size = 20000 
    chunk_idx = 0
    chunks_dir = os.path.join(output_dir, "temp_chunks")

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=collate_fn_robust,
        pin_memory=False
    )

    for batch in tqdm(dataloader, desc="Extracting MedGemma Vision Layers"):
        if batch is None:
            continue

        vision_device = next(vision_tower.parameters()).device
        images = batch['image'].to(vision_device, dtype=torch.bfloat16, non_blocking=True)
        batch_labels = [dict(zip(batch['labels'], t)) for t in zip(*batch['labels'].values())]

        feature_storage_batch = {}
        with torch.no_grad():
            hook_handles = []
            for name, module in vision_layers_to_probe.items():
                hook_handles.append(
                    module.register_forward_hook(
                        partial(capture_vision_hook, layer_name=name, storage_dict=feature_storage_batch)
                    )
                )

            vision_outputs = vision_tower(images)

            for handle in hook_handles:
                handle.remove()

            for layer_name, features in feature_storage_batch.items():
                all_layer_features[layer_name].append(features)

            projected = compute_projected_embedding(model, vision_outputs.last_hidden_state)
            all_layer_features['Vis_Projected'].append(projected)

            all_labels_and_metadata.extend(batch_labels)

        if len(all_labels_and_metadata) >= chunk_size:
            logging.info(f"\nReached chunk size limit. Saving Chunk {chunk_idx:04d} to disk to free RAM...")
            save_chunk(chunks_dir, chunk_idx, all_layer_features, all_labels_and_metadata)
            chunk_idx += 1
            
            all_layer_features = defaultdict(list)
            all_labels_and_metadata = []
            gc.collect()

    if len(all_labels_and_metadata) > 0:
        logging.info(f"\nSaving final partial Chunk {chunk_idx:04d} to disk...")
        save_chunk(chunks_dir, chunk_idx, all_layer_features, all_labels_and_metadata)
        all_layer_features = defaultdict(list)
        all_labels_and_metadata = []
        gc.collect()

    all_target_layers = list(vision_layers_to_probe.keys()) + ['Vis_Projected']
    combine_chunks(chunks_dir, output_dir, all_target_layers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract VISION layer-wise features from MedGemma 1.5.")
    parser.add_argument(
        '--dataset_folder_name', type=str, default='MIMIC-CXR-JPG',
        choices=list(DATASET_CONFIGS.keys()),
        help="Dataset to process."
    )
    parser.add_argument(
        '--split_value', type=int, required=True, choices=[0, 1, 2],
        help="Split: 0=train, 1=val, 2=test."
    )
    parser.add_argument('--num_shards', type=int, default=1, help="Total shards to split dataset into")
    parser.add_argument('--shard_idx', type=int, default=0, help="Index of this current shard")
    parser.add_argument("--final_only", action="store_true", help="Ignored for compatibility")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting")
    args = parser.parse_args()

    dset_cfg = DATASET_CONFIGS[args.dataset_folder_name]
    split_name_map = {0: "train", 1: "val", 2: "test"}
    split_name = split_name_map[args.split_value]

    logging.info(f"--- MedGemma 1.5 VISION Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

    features_root = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, args.dataset_folder_name)
    
    # Adjust output directory if sharding
    if args.num_shards > 1:
        output_dir = os.path.join(features_root, f"features_vision_only_{split_name}_shard_{args.shard_idx}")
    else:
        output_dir = os.path.join(features_root, f"features_vision_only_{split_name}")

    if os.path.exists(output_dir) and os.listdir(output_dir) and not getattr(args, "final_only", False) and not getattr(args, "overwrite", False):
        logging.error(f"Output directory already exists and is non-empty: {output_dir}")
        logging.error("Refusing to overwrite. Use --overwrite or --final_only.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output: {output_dir}")

    image_model = load_medgemma_model(MEDGEMMA_MODEL_PATH, DEVICE)
    image_processor = load_image_processor(MEDGEMMA_MODEL_PATH)

    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = MedGemmaVisionDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        base_image_dir=dset_cfg['base_image_dir'],
        image_processor=image_processor,
        num_shards=args.num_shards,
        shard_idx=args.shard_idx
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} samples.")
    extract_vision_layers_and_save(dataset, image_model, DEVICE, output_dir)
    logging.info("--- MedGemma 1.5 Vision Extraction Finished ---")