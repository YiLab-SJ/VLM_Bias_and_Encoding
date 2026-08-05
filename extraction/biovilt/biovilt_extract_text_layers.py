# biovilt_extract_text_layers.py
# Extracts layer-wise text embeddings from BioViL-T's CXR-BERT text encoder.
# Architecture: 12-layer BERT (hidden_size=768) + BertProjectionHead (768 -> 128)
#
# Output format matches the original CheXzero pipeline:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python biovilt_extract_text_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

import torch
import torch.nn as nn
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
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_biovilt import (
    BIOVILT_MODEL_DIR,
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    TEXT_LAYER_DIMS, CHEXPERT_REPORTS_CSV_PATH, MIMIC_REPORTS_BASE_DIR
)

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

OPTIMIZED_BATCH_SIZE = 128
OPTIMIZED_NUM_WORKERS = min(8, os.cpu_count() if os.cpu_count() else 1)


# =============================================================================
# --- Model & Tokenizer Loading ---
# =============================================================================
def load_biovilt_text_model(model_dir, device):
    """Load the BioViL-T text model (CXR-BERT) with pre-trained weights."""
    logging.info(f"Loading BioViL-T text model (CXR-BERT) from: {model_dir}")
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    model.float().to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    logging.info(f"CXR-BERT loaded to {device} and frozen.")
    return model


def load_biovilt_tokenizer(model_dir):
    """Load the CXR-BERT tokenizer."""
    logging.info(f"Loading CXR-BERT tokenizer from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    logging.info("Tokenizer loaded.")
    return tokenizer


# =============================================================================
# --- Report Text Helpers ---
# =============================================================================
def get_impression_from_mimic_report(report_text):
    """Extract the IMPRESSION section from a MIMIC-CXR report."""
    s_split = report_text.split()
    impression_indices = [i for i, word in enumerate(s_split) if word == "IMPRESSION:"]
    if not impression_indices:
        return report_text.strip()

    begin_idx = impression_indices[-1] + 1
    end_idx = len(s_split)
    end_markers = ["RECOMMENDATION(S):", "RECOMMENDATION:", "RECOMMENDATIONS:",
                    "NOTIFICATION:", "NOTIFICATIONS:"]
    for i in range(begin_idx, len(s_split)):
        if s_split[i] in end_markers:
            end_idx = i
            break
    return " ".join(s_split[begin_idx:end_idx]).strip()


# =============================================================================
# --- Hook Function ---
# =============================================================================
def capture_text_hook(module, input_val, output_val, layer_name, storage_dict):
    """
    Forward hook for CXR-BERT text layers.
    BERT layers output (B, Seq, Dim) tuples; we take the CLS token [0] or mean-pool.
    """
    # BERT layers return a tuple: (hidden_states, ...). Take the first element.
    if isinstance(output_val, tuple):
        tensor = output_val[0]
    else:
        tensor = output_val

    if not isinstance(tensor, torch.Tensor):
        return

    tensor = tensor.detach()

    if layer_name == 'Txt_TokenEmbed':
        # Embedding layer output: (B, Seq, Dim) -> mean pool over Seq
        if tensor.ndim == 3:
            features = tensor.mean(dim=1)
        else:
            features = tensor
    elif layer_name.startswith('Txt_Block_'):
        # BERT encoder layer output: (B, Seq, Dim) -> CLS token (index 0)
        if tensor.ndim == 3:
            features = tensor[:, 0, :]  # CLS token
        else:
            features = tensor
    else:
        return

    storage_dict[layer_name] = features.cpu().numpy()


# =============================================================================
# --- Dataset ---
# =============================================================================
class BioViLTTextDataset(Dataset):
    """Dataset for BioViL-T text feature extraction."""

    def __init__(self, metadata_csv_path, target_split_value, dataset_key):
        self.dataset_key = dataset_key
        full_df = pd.read_csv(metadata_csv_path)
        self.metadata_df = full_df[full_df["split"] == target_split_value].reset_index(drop=True)

        # Load CheXpert reports if needed
        self.chexpert_reports = {}
        if self.dataset_key == 'chexpert':
            logging.info(f"Loading CheXpert reports from: {CHEXPERT_REPORTS_CSV_PATH}")
            reports_df = pd.read_csv(CHEXPERT_REPORTS_CSV_PATH)
            self.chexpert_reports = pd.Series(
                reports_df.section_impression.values,
                index=reports_df.path_to_image
            ).to_dict()
            logging.info(f"Loaded {len(self.chexpert_reports)} CheXpert reports.")

        if self.metadata_df.empty:
            logging.warning(f"No data found for split {target_split_value}.")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        data_row = self.metadata_df.iloc[idx]
        impression_text = ""

        try:
            if self.dataset_key == 'MIMIC-CXR-JPG':
                relative_img_path = data_row['filename']
                path_parts = relative_img_path.split('/')
                study_path_part = '/'.join(path_parts[0:4])
                report_relative_path = f"{study_path_part}.txt"
                report_full_path = os.path.join(MIMIC_REPORTS_BASE_DIR, report_relative_path)
                with open(report_full_path, 'r', encoding='utf-8') as f:
                    raw_report = f.read()
                impression_text = get_impression_from_mimic_report(raw_report)
            elif self.dataset_key == 'chexpert':
                chexpert_path_key = data_row['filename']
                impression_text = self.chexpert_reports.get(chexpert_path_key, "")
                if pd.isna(impression_text):
                    impression_text = ""
            elif self.dataset_key == 'rexgradient':
                impression_text = str(data_row.get('report_text', ''))
                if pd.isna(impression_text) or impression_text == 'nan':
                    impression_text = ""

            if not impression_text:
                return None

            return {'report_text': impression_text, 'labels': data_row.to_dict()}
        except Exception:
            return None


def custom_collate_fn(batch_list):
    """Collate function for text batches."""
    batch_list = [item for item in batch_list if item is not None]
    if not batch_list:
        return None
    report_texts = [item['report_text'] for item in batch_list]
    labels = [item['labels'] for item in batch_list]
    return {'report_text': report_texts, 'labels': labels}


# =============================================================================
# --- Main Extraction Function ---
# =============================================================================
def extract_text_layers_and_save(dataset, model, tokenizer, device, output_dir):
    """Extract features from all CXR-BERT text layers and save them."""

    logging.info("Registering hooks on CXR-BERT text layers...")

    text_layers_to_probe = OrderedDict()

    # Token embedding layer (bert.embeddings)
    text_layers_to_probe['Txt_TokenEmbed'] = model.bert.embeddings

    # BERT encoder layers (12 blocks)
    for i, layer in enumerate(model.bert.encoder.layer):
        text_layers_to_probe[f'Txt_Block_{i+1}'] = layer

    logging.info(f"Defined {len(text_layers_to_probe)} intermediate text layers to hook.")

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=custom_collate_fn,
        pin_memory=False  # Text data, no GPU pinning benefit
    )

    for batch in tqdm(dataloader, desc="Extracting BioViL-T Text Layers"):
        if batch is None:
            continue

        report_texts = batch['report_text']
        batch_labels = batch['labels']

        # Tokenize with CXR-BERT tokenizer
        encoded = tokenizer(
            report_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128  # CXR-BERT uses 128 max length
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        feature_storage_batch = {}
        with torch.no_grad():
            # Register hooks
            hook_handles = []
            for name, module in text_layers_to_probe.items():
                hook_handles.append(
                    module.register_forward_hook(
                        partial(capture_text_hook, layer_name=name, storage_dict=feature_storage_batch)
                    )
                )

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Remove hooks
            for handle in hook_handles:
                handle.remove()

            # Collect hooked features
            for layer_name, features in feature_storage_batch.items():
                all_layer_features[layer_name].append(features)

            # Compute the final projected text embedding (128-dim)
            # CXR-BERT: CLS token -> cls_projection_head -> 128-dim
            cls_token = outputs.last_hidden_state[:, 0, :]
            projected = model.cls_projection_head(cls_token)
            all_layer_features['text_embedding_final'].append(projected.cpu().numpy())

            all_labels_and_metadata.extend(batch_labels)

    if not all_labels_and_metadata:
        logging.error("No features were extracted. No output files will be saved.")
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
    parser = argparse.ArgumentParser(description="Extract TEXT layer-wise features from BioViL-T (CXR-BERT).")
    parser.add_argument('--dataset_folder_name', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to process.")
    parser.add_argument('--split_value', type=int, required=True, choices=[0, 1, 2],
                        help="Split: 0=train, 1=val, 2=test.")
    args = parser.parse_args()

    dset_cfg = DATASET_CONFIGS[args.dataset_folder_name]
    split_name_map = {0: "train", 1: "val", 2: "test"}
    split_name = split_name_map[args.split_value]

    logging.info(f"--- BioViL-T TEXT Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

    # Output directory
    features_root = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, args.dataset_folder_name)
    output_dir = os.path.join(features_root, f"features_text_only_{split_name}")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        logging.error(f"Output directory already exists and is non-empty: {output_dir}")
        logging.error("Refusing to overwrite. Delete manually if you want to re-run.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output: {output_dir}")

    # Load model and tokenizer
    text_model = load_biovilt_text_model(BIOVILT_MODEL_DIR, DEVICE)
    tokenizer = load_biovilt_tokenizer(BIOVILT_MODEL_DIR)

    # Prepare dataset
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = BioViLTTextDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        dataset_key=args.dataset_folder_name
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} metadata entries. Some may be skipped if reports are missing.")

    extract_text_layers_and_save(dataset, text_model, tokenizer, DEVICE, output_dir)
    logging.info("--- BioViL-T Text Extraction Finished ---")
