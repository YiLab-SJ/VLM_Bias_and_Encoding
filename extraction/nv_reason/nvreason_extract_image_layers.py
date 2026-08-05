# nvreason_extract_image_layers.py
import torch
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
from transformers import AutoModelForImageTextToText, AutoProcessor

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_nvreason import (
    NVREASON_MODEL_PATH, DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, 
    RANDOM_STATE, OPTIMIZED_BATCH_SIZE
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

def save_chunk(chunk_dir, chunk_idx, features_dict, metadata_list):
    os.makedirs(chunk_dir, exist_ok=True)
    df = pd.DataFrame(metadata_list)
    df.to_csv(os.path.join(chunk_dir, f"metadata_{chunk_idx:04d}.csv"), index=False)
    for layer_name, feat_list in features_dict.items():
        np.save(os.path.join(chunk_dir, f"{layer_name}_{chunk_idx:04d}.npy"), np.concatenate(feat_list, axis=0))

def combine_chunks(chunk_dir, output_dir, layer_names):
    import glob
    meta_files = sorted(glob.glob(os.path.join(chunk_dir, "metadata_*.csv")))
    if meta_files:
        pd.concat([pd.read_csv(f) for f in meta_files], ignore_index=True).to_csv(os.path.join(output_dir, "labels_and_metadata.csv"), index=False)
    for layer_name in layer_names:
        feat_files = sorted(glob.glob(os.path.join(chunk_dir, f"{layer_name}_*.npy")))
        if feat_files:
            np.save(os.path.join(output_dir, f"{layer_name}_embeddings.npy"), np.concatenate([np.load(f) for f in feat_files], axis=0))
    shutil.rmtree(chunk_dir)

def capture_vision_hook(module, input_val, output_val, layer_name, storage_dict):
    tensor = output_val[0] if isinstance(output_val, tuple) else output_val
    if not isinstance(tensor, torch.Tensor): return
    tensor = tensor.detach().float()
    
    # Qwen-VL Vision outputs are usually (num_patches, hidden_size) or (B, seq, hidden)
    # We flatten all but the last dim and mean-pool across patches
    features = tensor.view(-1, tensor.shape[-1]).mean(dim=0).unsqueeze(0)
    storage_dict[layer_name] = features.cpu().numpy()

class NVReasonVisionDataset(Dataset):
    def __init__(self, metadata_csv_path, target_split_value, base_image_dir):
        self.base_image_dir = base_image_dir
        full_df = pd.read_csv(metadata_csv_path)
        self.metadata_df = full_df[full_df["split"] == target_split_value].reset_index(drop=True)

    def __len__(self): return len(self.metadata_df)

    def __getitem__(self, idx):
        data_row = self.metadata_df.iloc[idx]
        try:
            full_img_path = os.path.join(self.base_image_dir, data_row['filename'])
            image = Image.open(full_img_path)
            # Fix for 16-bit PNGs (RexGradient)
            if image.mode in ('I;16', 'I'):
                import numpy as _np
                arr = _np.array(image, dtype=_np.float32)
                mn, mx = arr.min(), arr.max()
                if mx > mn:
                    arr = (arr - mn) / (mx - mn) * 255.0
                else:
                    arr = _np.zeros_like(arr)
                image = Image.fromarray(arr.astype(_np.uint8), mode='L')
            image = image.convert('RGB')
            return {'image': image, 'labels': data_row.to_dict()}
        except Exception:
            return None

def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return {'images': [b['image'] for b in batch], 'labels': [b['labels'] for b in batch]}

def extract_vision_layers(dataset, model, processor, output_dir):
    vision_layers_to_probe = OrderedDict()
    for i, layer in enumerate(model.model.visual.blocks):
        vision_layers_to_probe[f'Vis_Block_{i}'] = layer

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []
    chunk_size, chunk_idx = 20000, 0
    chunks_dir = os.path.join(output_dir, "temp_chunks")

    dataloader = DataLoader(dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate)

    for batch in tqdm(dataloader, desc="Extracting Vision Layers"):
        if not batch: continue
        
        # Prepare inputs as NV-Reason expects to trigger the full vision pipeline properly
        messages = [{"role": "user", "content": [{"type": "image", "image": batch['images'][0]}, {"type": "text", "text": "Analyze image."}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, images=batch['images'], return_tensors="pt").to(DEVICE)

        feature_storage_batch = {}
        with torch.no_grad():
            handles = [module.register_forward_hook(partial(capture_vision_hook, layer_name=name, storage_dict=feature_storage_batch)) 
                       for name, module in vision_layers_to_probe.items()]
            
            _ = model(**inputs)
            
            for h in handles: h.remove()
            for name, feats in feature_storage_batch.items(): all_layer_features[name].append(feats)
            all_labels_and_metadata.extend(batch['labels'])

        if len(all_labels_and_metadata) >= chunk_size:
            save_chunk(chunks_dir, chunk_idx, all_layer_features, all_labels_and_metadata)
            chunk_idx += 1; all_layer_features.clear(); all_labels_and_metadata.clear(); gc.collect()

    if all_labels_and_metadata:
        save_chunk(chunks_dir, chunk_idx, all_layer_features, all_labels_and_metadata)
    combine_chunks(chunks_dir, output_dir, list(vision_layers_to_probe.keys()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder_name', type=str, default='MIMIC-CXR-JPG')
    parser.add_argument('--split_value', type=int, required=True)
    parser.add_argument("--final_only", action="store_true", help="Ignored for compatibility")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting")
    args = parser.parse_args()

    split_name = {0: "train", 1: "val", 2: "test"}[args.split_value]
    output_dir = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, args.dataset_folder_name, f"features_vision_only_{split_name}")
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(NVREASON_MODEL_PATH, torch_dtype=torch.float16).eval().to(DEVICE)
    processor = AutoProcessor.from_pretrained(NVREASON_MODEL_PATH)
    
    dataset = NVReasonVisionDataset(DATASET_CONFIGS[args.dataset_folder_name]['metadata_attr_lr_file'], args.split_value, DATASET_CONFIGS[args.dataset_folder_name]['base_image_dir'])
    extract_vision_layers(dataset, model, processor, output_dir)