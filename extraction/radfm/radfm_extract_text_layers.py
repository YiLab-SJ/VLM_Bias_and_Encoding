# radfm_extract_text_layers.py
# Extracts layer-wise text embeddings from RadFM's LLaMA-13B decoder.
# Architecture: 40-layer LLaMA decoder (hidden_size=5120, decoder-only).
#
# For text-only probing, we feed report text through the language model
# WITHOUT any image input. At each decoder layer, we mean-pool hidden states
# over non-padding tokens to get a single feature vector.
#
# Because RadFM's embedding layer (MyEmbedding) has a complex forward pass
# involving one-hot encoding and image token insertion, for pure text we
# bypass it and feed input_ids directly through the LLaMA model's own
# embed_tokens, then through the decoder layers.
#
# Output format matches the BioViL-T pipeline:
#   - One .npy file per layer: {layer_name}_embeddings.npy
#   - One labels_and_metadata.csv with all metadata columns
#
# Usage:
#   python radfm_extract_text_layers.py --dataset_folder_name MIMIC-CXR-JPG --split_value 0

import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

import torch.nn as nn
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
from tqdm import tqdm

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_radfm import (
    RADFM_REPO_DIR, RADFM_CHECKPOINT_PATH, RADFM_LANGUAGE_FILES_PATH,
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

# LLaMA-13B is ~26GB in fp32. With cached activations, batch size must be small.
OPTIMIZED_BATCH_SIZE = 8
OPTIMIZED_NUM_WORKERS = 4
MAX_SEQ_LENGTH = 512  # LLaMA supports 2048, but radiology reports are short


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
# --- Model & Tokenizer Loading ---
# =============================================================================
def load_radfm_model(lang_model_path, checkpoint_path, device_str):
    """Load the full RadFM model from checkpoint.
    
    The Language_files directory only contains config.json and tokenizer files
    (no model weights). We monkey-patch from_pretrained to do config-only init,
    since all actual weights come from the RadFM checkpoint (pytorch_model.bin).
    """
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
    # This is critical for text extraction where we use forward hooks on decoder layers
    if hasattr(model.lang_model, 'gradient_checkpointing_disable'):
        model.lang_model.gradient_checkpointing_disable()

    model = model.to(device_str)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logging.info(f"RadFM loaded on {device_str}.")
    return model


def load_tokenizer(tokenizer_path):
    """Load the LLaMA tokenizer (base, without image special tokens).
    
    For text-only extraction we use the base tokenizer without image placeholder
    tokens, since we feed plain text through the LLaMA decoder directly.
    """
    from transformers import LlamaTokenizer

    logging.info(f"Loading LLaMA tokenizer from: {tokenizer_path}")
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    # RadFM uses: pad=0, bos=1, eos=2
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
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
# We store the attention mask from each batch to use for masked mean pooling.
_current_attention_mask = None


def capture_text_hook(module, input_val, output_val, layer_name, storage_dict):
    """
    Forward hook for LLaMA decoder layers.
    Decoder layers output tuples: (hidden_states, ...) or BaseModelOutputWithPast.
    Mean-pool over non-padding tokens using the stored attention mask.
    """
    global _current_attention_mask

    if isinstance(output_val, tuple):
        tensor = output_val[0]
    else:
        tensor = output_val

    if not isinstance(tensor, torch.Tensor):
        return

    tensor = tensor.detach().float()  # -> float32 for numpy

    if tensor.ndim == 3:
        # (B, Seq, Dim) -> masked mean pool -> (B, Dim)
        if _current_attention_mask is not None:
            mask = _current_attention_mask.unsqueeze(-1).float().to(tensor.device)
            features = (tensor * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            features = tensor.mean(dim=1)
    elif tensor.ndim == 2:
        features = tensor
    else:
        return

    storage_dict[layer_name] = features.cpu().numpy()


# =============================================================================
# --- Dataset ---
# =============================================================================
class RadFMTextDataset(Dataset):
    """Dataset for RadFM text feature extraction."""

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
def extract_text_layers_and_save(dataset, model, tokenizer, output_dir):
    """Extract features from all LLaMA decoder layers and save them.
    
    For text-only extraction, we bypass RadFM's MyEmbedding layer and feed
    input_ids directly through the LLaMA model. This gives us clean text-only
    representations without any image token interference.
    """
    global _current_attention_mask

    logging.info("Setting up hooks on LLaMA-13B decoder layers...")

    # Access the LLaMA model: model.lang_model.model is the LlamaModel
    llama_model = model.lang_model.model  # LlamaModel (embed_tokens + layers + norm)

    text_layers_to_probe = OrderedDict()

    # Token embedding layer
    text_layers_to_probe['Txt_Embed'] = llama_model.embed_tokens

    # Decoder layers (40 blocks)
    for i, layer in enumerate(llama_model.layers):
        text_layers_to_probe[f'Txt_Block_{i + 1}'] = layer

    # Final RMSNorm
    text_layers_to_probe['Txt_FinalNorm'] = llama_model.norm

    logging.info(f"Defined {len(text_layers_to_probe)} text layers to hook.")

    all_layer_features = defaultdict(list)
    all_labels_and_metadata = []
    layer_names_tracked = set()

    chunk_size = 20000
    chunk_idx = 0
    chunks_dir = os.path.join(output_dir, "temp_chunks")

    dataloader = DataLoader(
        dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False,
        num_workers=OPTIMIZED_NUM_WORKERS, collate_fn=custom_collate_fn,
        pin_memory=False
    )

    for batch in tqdm(dataloader, desc="Extracting RadFM Text Layers"):
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
            max_length=MAX_SEQ_LENGTH
        )

        # Move to model device
        lm_device = next(llama_model.parameters()).device
        input_ids = encoded['input_ids'].to(lm_device)
        attention_mask = encoded['attention_mask'].to(lm_device)

        # Store attention mask for the hook to use
        _current_attention_mask = attention_mask

        feature_storage_batch = {}
        with torch.no_grad():
            # Register hooks
            hook_handles = []
            for name, module in text_layers_to_probe.items():
                hook_handles.append(
                    module.register_forward_hook(
                        partial(capture_text_hook, layer_name=name,
                                storage_dict=feature_storage_batch)
                    )
                )

            # Forward pass through LLaMA model directly (bypass MyEmbedding)
            # LlamaModel.forward() accepts input_ids + attention_mask
            _ = llama_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Remove hooks
            for handle in hook_handles:
                handle.remove()

            # Reset global mask
            _current_attention_mask = None

            # Collect hooked features
            for layer_name, features in feature_storage_batch.items():
                all_layer_features[layer_name].append(features)
                layer_names_tracked.add(layer_name)

            all_labels_and_metadata.extend(batch_labels)

        # --- CHUNKING TRIGGER ---
        if len(all_labels_and_metadata) >= chunk_size:
            logging.info(f"\nReached chunk size limit. Saving Chunk {chunk_idx:04d} to disk...")
            save_chunk(chunks_dir, chunk_idx, all_layer_features, all_labels_and_metadata)
            chunk_idx += 1
            all_layer_features = defaultdict(list)
            all_labels_and_metadata = []
            gc.collect()

    if not all_labels_and_metadata and chunk_idx == 0:
        logging.error("No features were extracted. No output files will be saved.")
        return

    # Save final partial chunk
    if len(all_labels_and_metadata) > 0:
        logging.info(f"\nSaving final partial Chunk {chunk_idx:04d} to disk...")
        save_chunk(chunks_dir, chunk_idx, all_layer_features, all_labels_and_metadata)
        all_layer_features = defaultdict(list)
        all_labels_and_metadata = []
        gc.collect()

    # Stitch it all back together
    combine_chunks(chunks_dir, output_dir, sorted(list(layer_names_tracked)))


# =============================================================================
# --- Main ---
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract TEXT layer-wise features from RadFM (LLaMA-13B decoder).")
    parser.add_argument('--dataset_folder_name', type=str, required=True,
                        choices=list(DATASET_CONFIGS.keys()),
                        help="Dataset to process.")
    parser.add_argument('--split_value', type=int, required=True, choices=[0, 1, 2],
                        help="Split: 0=train, 1=val, 2=test.")
    args = parser.parse_args()

    dset_cfg = DATASET_CONFIGS[args.dataset_folder_name]
    split_name_map = {0: "train", 1: "val", 2: "test"}
    split_name = split_name_map[args.split_value]

    logging.info(f"--- RadFM TEXT Layer Extraction: {args.dataset_folder_name} - {split_name} ---")

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
    full_model = load_radfm_model(RADFM_LANGUAGE_FILES_PATH, RADFM_CHECKPOINT_PATH, DEVICE)
    tokenizer = load_tokenizer(RADFM_LANGUAGE_FILES_PATH)

    # Prepare dataset
    logging.info(f"Loading metadata from: {dset_cfg['metadata_attr_lr_file']}")
    dataset = RadFMTextDataset(
        metadata_csv_path=dset_cfg['metadata_attr_lr_file'],
        target_split_value=args.split_value,
        dataset_key=args.dataset_folder_name
    )

    if len(dataset) == 0:
        logging.error(f"Dataset is empty for {args.dataset_folder_name} - {split_name}. Aborting.")
        sys.exit(1)

    logging.info(f"Found {len(dataset)} metadata entries. Some may be skipped if reports are missing.")

    extract_text_layers_and_save(dataset, full_model, tokenizer, output_dir)
    logging.info("--- RadFM Text Extraction Finished ---")