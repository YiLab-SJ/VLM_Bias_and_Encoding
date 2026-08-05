# llavamed_extract_text_layers.py
# Extracts layer-wise text embeddings from LLaVA-Med 1.5's Mistral-7B language model.
#
# Architecture: Mistral-7B-Instruct-v0.2 (32 decoder layers, hidden_size=4096)
#
# Strategy:
#   - Load the full LLaVA-Med model using the LLaVA-Med builder to ensure proper
#     tokenizer configuration and weight loading.
#   - Run text-only (no images) through the model with output_hidden_states=True.
#   - Mean-pool hidden states over token positions for each layer.
#
# Output format matches the original pipeline:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python llavamed_extract_text_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import random
import argparse
import logging
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_llavamed import (
    LLAVAMED_MODEL_PATH, LLAVAMED_REPO_PATH,
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    TEXT_LAYER_DIMS, CHEXPERT_REPORTS_CSV_PATH, MIMIC_REPORTS_BASE_DIR
)

# Add LLaVA-Med repo to path for model builder
sys.path.insert(0, LLAVAMED_REPO_PATH)

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

# Mistral-7B is large; keep batch size conservative to fit in 16GB GPU memory
# with hidden states for 33 layers × 4096 dims
OPTIMIZED_BATCH_SIZE = 8
OPTIMIZED_NUM_WORKERS = min(8, os.cpu_count() if os.cpu_count() else 1)
MAX_TOKEN_LENGTH = 128  # Truncate reports to this length for efficiency


# =============================================================================
# --- Model & Tokenizer Loading ---
# =============================================================================
def load_llavamed_for_text(model_path, device):
    """
    Load LLaVA-Med 1.5 model for text-only feature extraction.
    Uses the LLaVA-Med builder to ensure correct weight loading.
    """
    logging.info(f"Loading LLaVA-Med 1.5 model from: {model_path}")

    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava-med-v1.5-mistral-7b",
        device=device
    )

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logging.info(f"LLaVA-Med 1.5 model loaded to {device} (fp16) and frozen.")
    logging.info(f"Model has {sum(p.numel() for p in model.parameters())/1e9:.1f}B parameters.")
    return model, tokenizer


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
# --- Dataset ---
# =============================================================================
class LLaVAMedTextDataset(Dataset):
    """Dataset for LLaVA-Med text feature extraction."""

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
    """Extract features from all Mistral decoder layers and save."""

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=custom_collate_fn,
        pin_memory=False
    )

    # Get the underlying Mistral model
    # LLaVA-Med wraps Mistral: model.model is LlavaMistralModel which inherits MistralModel
    mistral_model = model.model

    for batch in tqdm(dataloader, desc="Extracting LLaVA-Med Text Layers"):
        if batch is None:
            continue

        report_texts = batch['report_text']
        batch_labels = batch['labels']

        # Tokenize
        encoded = tokenizer(
            report_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH
        )
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            # Forward pass through the Mistral backbone (not the full CausalLM)
            # MistralModel.forward returns BaseModelOutputWithPast with hidden_states
            outputs = mistral_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            hidden_states = outputs.hidden_states
            # hidden_states is a tuple of (B, Seq, 4096) for 33 states:
            #   [0] = embedding output, [1]-[32] = decoder layer outputs

            # Mean-pool over non-padding token positions
            # attention_mask: (B, Seq) -> expand for broadcasting
            mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, Seq, 1)

            for layer_idx, hs in enumerate(hidden_states):
                # hs: (B, Seq, 4096)
                # Mask and mean-pool over tokens
                masked_hs = hs.float() * mask_expanded  # zero out padding
                summed = masked_hs.sum(dim=1)  # (B, 4096)
                lengths = mask_expanded.sum(dim=1).clamp(min=1)  # (B, 1)
                pooled = (summed / lengths).cpu().numpy()  # (B, 4096)

                if layer_idx == 0:
                    layer_name = 'Txt_Mistral_Embed'
                else:
                    layer_name = f'Txt_Mistral_Layer_{layer_idx}'

                all_layer_features[layer_name].append(pooled)

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
    parser = argparse.ArgumentParser(description="Extract TEXT layer-wise features from LLaVA-Med 1.5 (Mistral-7B).")
    parser.add_argument('--dataset_folder_name', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to process.")
    parser.add_argument('--split_value', type=int, required=True, choices=[0, 1, 2],
                        help="Split: 0=train, 1=val, 2=test.")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU device ID to use (default: 0).")
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu_id >= 0:
        DEVICE = f"cuda:{args.gpu_id}"
    else:
        DEVICE = "cpu"

    dset_cfg = DATASET_CONFIGS[args.dataset_folder_name]
    split_name_map = {0: "train", 1: "val", 2: "test"}
    split_name = split_name_map[args.split_value]

    logging.info(f"--- LLaVA-Med 1.5 TEXT Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

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
    model, tokenizer = load_llavamed_for_text(LLAVAMED_MODEL_PATH, DEVICE)

    # Prepare dataset
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = LLaVAMedTextDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        dataset_key=args.dataset_folder_name
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} metadata entries. Some may be skipped if reports are missing.")

    extract_text_layers_and_save(dataset, model, tokenizer, DEVICE, output_dir)
    logging.info("--- LLaVA-Med 1.5 Text Extraction Finished ---")
