# chexagent_extract_text_layers.py
# Extracts layer-wise text embeddings from CheXagent's Phi-2 decoder.
# Architecture: Phi-2 decoder (32 layers, hidden_size=2560)
# Since CheXagent is a generative VLM (not contrastive), there is no separate
# text encoder. We feed report text through the Phi-2 LLM and extract hidden
# states at each layer. The representation is the last non-padding token's
# hidden state (similar to GPT-style models).
#
# Output format matches the BioViL-T pipeline:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python chexagent_extract_text_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

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
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_chexagent import (
    CHEXAGENT_MODEL_NAME,
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

# Phi-2 3B is large; use moderate batch size
OPTIMIZED_BATCH_SIZE = 16
OPTIMIZED_NUM_WORKERS = min(8, os.cpu_count() if os.cpu_count() else 1)
MAX_TOKEN_LENGTH = 256  # Phi-2 context length; reports are typically short


# =============================================================================
# --- Model & Tokenizer Loading ---
# =============================================================================
def load_chexagent_text_model(model_name, device):
    """Load the full CheXagent model, return the CheXagentModel (LLM backbone)."""
    logging.info(f"Loading CheXagent model from: {model_name}")

    full_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # The LLM backbone is at full_model.model (CheXagentModel, which wraps Phi-2)
    # We need: model.embed_tokens, model.layers[0..31], model.final_layernorm
    llm_model = full_model.model
    llm_model.to(device).eval()
    for param in llm_model.parameters():
        param.requires_grad = False

    logging.info(f"CheXagent Phi-2 LLM loaded to {device} and frozen.")
    logging.info(f"  Decoder layers: {len(llm_model.layers)}")
    return llm_model


def load_chexagent_tokenizer(model_name):
    """Load the CheXagent tokenizer (Phi-2 tokenizer)."""
    logging.info(f"Loading tokenizer from: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
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
def capture_text_hook(module, input_val, output_val, layer_name, storage_dict,
                      attention_mask_ref):
    """
    Forward hook for CheXagent Phi-2 decoder layers.
    Decoder layers output tuples: (hidden_states, ...).
    We extract the last non-padding token representation using attention_mask.
    For the embedding layer, we mean-pool over non-padding tokens.
    """
    if isinstance(output_val, tuple):
        tensor = output_val[0]
    else:
        tensor = output_val

    if not isinstance(tensor, torch.Tensor):
        return

    tensor = tensor.detach()

    if tensor.ndim != 3:
        return

    attn_mask = attention_mask_ref[0]  # (B, SeqLen)

    if layer_name == 'Txt_TokenEmbed':
        # Embedding layer: mean-pool over non-padding tokens
        mask_expanded = attn_mask.unsqueeze(-1).float()  # (B, SeqLen, 1)
        sum_hidden = (tensor * mask_expanded).sum(dim=1)  # (B, Dim)
        count = mask_expanded.sum(dim=1).clamp(min=1)    # (B, 1)
        features = sum_hidden / count
    else:
        # Decoder layers: take last non-padding token
        # attention_mask: 1 for real tokens, 0 for padding
        # last_token_idx = sum(mask) - 1 for each sample
        seq_lengths = attn_mask.sum(dim=1).long() - 1  # (B,)
        batch_indices = torch.arange(tensor.size(0), device=tensor.device)
        features = tensor[batch_indices, seq_lengths, :]  # (B, Dim)

    storage_dict[layer_name] = features.cpu().numpy()


# =============================================================================
# --- Dataset ---
# =============================================================================
class CheXagentTextDataset(Dataset):
    """Dataset for CheXagent text feature extraction."""

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
def extract_text_layers_and_save(dataset, llm_model, tokenizer, device, output_dir):
    """Extract features from all Phi-2 decoder layers and save them."""

    logging.info("Registering hooks on CheXagent Phi-2 layers...")

    text_layers_to_probe = OrderedDict()

    # Token embedding layer
    text_layers_to_probe['Txt_TokenEmbed'] = llm_model.embed_tokens

    # Phi-2 decoder layers (32 blocks)
    for i, layer in enumerate(llm_model.layers):
        text_layers_to_probe[f'Txt_PhiBlock_{i+1}'] = layer

    logging.info(f"Defined {len(text_layers_to_probe)} text layers to hook.")

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []

    # Mutable container to pass attention mask to hooks
    attention_mask_ref = [None]

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=custom_collate_fn,
        pin_memory=False
    )

    for batch in tqdm(dataloader, desc="Extracting CheXagent Text Layers"):
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

        # Store attention mask for hooks
        attention_mask_ref[0] = attention_mask

        feature_storage_batch = {}
        with torch.no_grad():
            # Register hooks
            hook_handles = []
            for name, module in text_layers_to_probe.items():
                hook_handles.append(
                    module.register_forward_hook(
                        partial(capture_text_hook, layer_name=name,
                                storage_dict=feature_storage_batch,
                                attention_mask_ref=attention_mask_ref)
                    )
                )

            # Forward pass through the LLM (text-only, no images)
            # CheXagentModel.forward() auto-detects image tokens in input_ids.
            # Since we pass plain text (no img_start tokens), it skips the
            # visual encoder entirely (images=None, fake_images=None path).
            outputs = llm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Remove hooks
            for handle in hook_handles:
                handle.remove()

            # Collect hooked features
            for layer_name, features in feature_storage_batch.items():
                all_layer_features[layer_name].append(features)

            # Compute text_embedding_final:
            # The last_hidden_state from outputs is after final_layernorm.
            # Take the last non-padding token.
            last_hidden = outputs.last_hidden_state  # (B, SeqLen, 2560)
            seq_lengths = attention_mask.sum(dim=1).long() - 1
            batch_indices = torch.arange(last_hidden.size(0), device=device)
            final_features = last_hidden[batch_indices, seq_lengths, :]
            all_layer_features['text_embedding_final'].append(final_features.cpu().numpy())

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
    parser = argparse.ArgumentParser(description="Extract TEXT layer-wise features from CheXagent (Phi-2).")
    parser.add_argument('--dataset_folder_name', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to process.")
    parser.add_argument('--split_value', type=int, required=True, choices=[0, 1, 2],
                        help="Split: 0=train, 1=val, 2=test.")
    args = parser.parse_args()

    dset_cfg = DATASET_CONFIGS[args.dataset_folder_name]
    split_name_map = {0: "train", 1: "val", 2: "test"}
    split_name = split_name_map[args.split_value]

    logging.info(f"--- CheXagent TEXT Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

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
    llm_model = load_chexagent_text_model(CHEXAGENT_MODEL_NAME, DEVICE)
    tokenizer = load_chexagent_tokenizer(CHEXAGENT_MODEL_NAME)

    # Prepare dataset
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = CheXagentTextDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        dataset_key=args.dataset_folder_name
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} metadata entries. Some may be skipped if reports are missing.")

    extract_text_layers_and_save(dataset, llm_model, tokenizer, DEVICE, output_dir)
    logging.info("--- CheXagent Text Extraction Finished ---")
