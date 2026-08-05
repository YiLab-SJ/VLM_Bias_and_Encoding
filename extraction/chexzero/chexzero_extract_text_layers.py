# chexzero_extract_text_layers.py
# Extracts layer-wise text embeddings from CheXzero's CLIP text transformer.
# Architecture: 12 ResidualAttentionBlock layers (hidden_size=512)
# CheXzero uses CLIP's contrastive text encoder with SimpleTokenizer (context_length=77).
# The representation for each layer uses the EOT token position (like the original CLIP).
#
# Output format matches the pipeline convention:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python chexzero_extract_text_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

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

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_chexzero import (
    CHEXZERO_MODEL_PATH, CHEXZERO_SRC_DIR,
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    TEXT_LAYER_DIMS, CHEXPERT_REPORTS_CSV_PATH, MIMIC_REPORTS_BASE_DIR
)

# --- CheXzero model & tokenizer imports ---
sys.path.insert(0, CHEXZERO_SRC_DIR)
from model import build_model
from simple_tokenizer import SimpleTokenizer as _Tokenizer

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)

OPTIMIZED_BATCH_SIZE = 64
OPTIMIZED_NUM_WORKERS = min(8, os.cpu_count() if os.cpu_count() else 1)
CONTEXT_LENGTH = 77

# Initialize CLIP's tokenizer
_tokenizer = _Tokenizer()


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
    logging.info(f"  Text: {len(model.transformer.resblocks)} layers, width=512")
    logging.info(f"  Context length: {model.context_length}")

    return model


# =============================================================================
# --- Tokenizer ---
# =============================================================================
def tokenize_truncated(texts, context_length=77):
    """Tokenize texts with truncation (instead of CLIP's default which raises RuntimeError)."""
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]

    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            # Truncate: keep SOT + first (context_length-2) tokens + EOT
            tokens = tokens[:context_length - 1] + [eot_token]
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


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
def capture_text_hook(module, input_val, output_val, layer_name, storage_dict,
                      token_ids_ref):
    """
    Forward hook for CheXzero text transformer layers.
    ResidualAttentionBlock outputs are in LND format: (SeqLen, B, 512).
    For token_embedding, output is in NLD format: (B, SeqLen, 512).

    For Txt_TokenEmbed: mean-pool over non-padding tokens.
    For Txt_Block_*: extract EOT token (highest token ID position).
    """
    tensor = output_val
    if isinstance(tensor, tuple):
        tensor = tensor[0]

    if not isinstance(tensor, torch.Tensor):
        return

    tensor = tensor.detach()

    if layer_name == 'Txt_TokenEmbed':
        # token_embedding output: (B, SeqLen, 512) NLD format
        if tensor.ndim != 3:
            return
        token_ids = token_ids_ref[0]  # (B, SeqLen)
        # Non-padding mask: token_ids != 0
        mask = (token_ids != 0).float().unsqueeze(-1)  # (B, SeqLen, 1)
        sum_hidden = (tensor * mask).sum(dim=1)         # (B, 512)
        count = mask.sum(dim=1).clamp(min=1)            # (B, 1)
        features = sum_hidden / count                   # (B, 512)
    else:
        # Transformer resblock output: (SeqLen, B, 512) LND format
        if tensor.ndim != 3:
            return
        tensor = tensor.permute(1, 0, 2)  # (B, SeqLen, 512) NLD
        token_ids = token_ids_ref[0]  # (B, SeqLen)
        # EOT token is the highest token ID in each sequence
        eot_positions = token_ids.argmax(dim=-1)  # (B,)
        batch_indices = torch.arange(tensor.size(0), device=tensor.device)
        features = tensor[batch_indices, eot_positions, :]  # (B, 512)

    storage_dict[layer_name] = features.cpu().numpy()


# =============================================================================
# --- Dataset ---
# =============================================================================
class CheXzeroTextDataset(Dataset):
    """Dataset for CheXzero text feature extraction."""

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
def extract_text_layers_and_save(dataset, model, device, output_dir):
    """Extract features from all CheXzero text transformer layers and save them."""

    logging.info("Registering hooks on CheXzero text layers...")

    text_layers_to_probe = OrderedDict()

    # Token embedding layer
    text_layers_to_probe['Txt_TokenEmbed'] = model.token_embedding

    # Transformer resblocks (12 blocks)
    for i, block in enumerate(model.transformer.resblocks):
        text_layers_to_probe[f'Txt_Block_{i+1}'] = block

    logging.info(f"Defined {len(text_layers_to_probe)} text layers to hook.")

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []

    # Mutable container to pass token IDs to hooks
    token_ids_ref = [None]

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=custom_collate_fn,
        pin_memory=False
    )

    for batch in tqdm(dataloader, desc="Extracting CheXzero Text Layers"):
        if batch is None:
            continue

        report_texts = batch['report_text']
        batch_labels = batch['labels']

        # Tokenize with truncation
        token_ids = tokenize_truncated(report_texts, context_length=CONTEXT_LENGTH).to(device)

        # Store token IDs for hooks (needed for EOT position and padding mask)
        token_ids_ref[0] = token_ids

        feature_storage_batch = {}
        with torch.no_grad():
            # Register hooks
            hook_handles = []
            for name, module in text_layers_to_probe.items():
                hook_handles.append(
                    module.register_forward_hook(
                        partial(capture_text_hook, layer_name=name,
                                storage_dict=feature_storage_batch,
                                token_ids_ref=token_ids_ref)
                    )
                )

            # Forward pass through encode_text
            # encode_text: token_embedding -> + positional_embedding -> permute(LND)
            #   -> transformer(resblocks) -> permute(NLD) -> ln_final
            #   -> select EOT token -> @ text_projection -> (B, 512)
            final_embedding = model.encode_text(token_ids)  # (B, 512)

            # Remove hooks
            for handle in hook_handles:
                handle.remove()

            # Collect hooked features
            for layer_name, features in feature_storage_batch.items():
                all_layer_features[layer_name].append(features)

            # Collect the final text embedding (512-dim)
            all_layer_features['text_embedding_final'].append(final_embedding.cpu().numpy())

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
    parser = argparse.ArgumentParser(description="Extract TEXT layer-wise features from CheXzero.")
    parser.add_argument('--dataset_folder_name', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to process.")
    parser.add_argument('--split_value', type=int, required=True, choices=[0, 1, 2],
                        help="Split: 0=train, 1=val, 2=test.")
    args = parser.parse_args()

    dset_cfg = DATASET_CONFIGS[args.dataset_folder_name]
    split_name_map = {0: "train", 1: "val", 2: "test"}
    split_name = split_name_map[args.split_value]

    logging.info(f"--- CheXzero TEXT Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

    # Output directory
    features_root = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, args.dataset_folder_name)
    output_dir = os.path.join(features_root, f"features_text_only_{split_name}")

    if os.path.exists(output_dir) and os.listdir(output_dir):
        logging.error(f"Output directory already exists and is non-empty: {output_dir}")
        logging.error("Refusing to overwrite. Delete manually if you want to re-run.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output: {output_dir}")

    # Load model
    model = load_chexzero_model(CHEXZERO_MODEL_PATH, DEVICE)

    # Prepare dataset
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = CheXzeroTextDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        dataset_key=args.dataset_folder_name
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} metadata entries. Some may be skipped if reports are missing.")

    extract_text_layers_and_save(dataset, model, DEVICE, output_dir)
    logging.info("--- CheXzero Text Extraction Finished ---")
