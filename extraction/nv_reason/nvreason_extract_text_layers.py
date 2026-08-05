# nvreason_extract_text_layers.py
import torch
import pandas as pd
import numpy as np
import os
import sys
import random
import argparse
import gc
import shutil
from collections import OrderedDict, defaultdict
from functools import partial
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_nvreason import (
    NVREASON_MODEL_PATH, DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, 
    MIMIC_REPORTS_BASE_DIR, OPTIMIZED_BATCH_SIZE, CHEXPERT_REPORTS_CSV_PATH
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_current_attention_mask = None

def get_impression(report_text):
    s_split = report_text.split()
    idx = [i for i, w in enumerate(s_split) if w == "IMPRESSION:"]
    if not idx: return report_text.strip()
    begin = idx[-1] + 1
    for i in range(begin, len(s_split)):
        if s_split[i] in ["RECOMMENDATION(S):", "RECOMMENDATION:", "RECOMMENDATIONS:", "NOTIFICATION:", "NOTIFICATIONS:"]:
            return " ".join(s_split[begin:i]).strip()
    return " ".join(s_split[begin:]).strip()

def capture_text_hook(module, input_val, output_val, layer_name, storage_dict):
    global _current_attention_mask
    tensor = output_val[0] if isinstance(output_val, tuple) else output_val
    if not isinstance(tensor, torch.Tensor): return
    tensor = tensor.detach().float()
    
    if tensor.ndim == 3:
        if _current_attention_mask is not None:
            mask = _current_attention_mask.unsqueeze(-1).float().to(tensor.device)
            feats = (tensor * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            feats = tensor.mean(dim=1)
    else: feats = tensor.view(-1, tensor.shape[-1]).mean(dim=0).unsqueeze(0)
    
    storage_dict[layer_name] = feats.cpu().numpy()

class NVReasonTextDataset(Dataset):
    def __init__(self, metadata_csv_path, target_split_value, dataset_key='MIMIC-CXR-JPG'):
        self.dataset_key = dataset_key
        self.df = pd.read_csv(metadata_csv_path)
        self.df = self.df[self.df["split"] == target_split_value].reset_index(drop=True)
        self.chexpert_reports = {}
        if self.dataset_key == 'chexpert':
            import logging
            logging.info(f"Loading CheXpert reports from: {CHEXPERT_REPORTS_CSV_PATH}")
            reports_df = pd.read_csv(CHEXPERT_REPORTS_CSV_PATH)
            self.chexpert_reports = pd.Series(
                reports_df.section_impression.values,
                index=reports_df.path_to_image
            ).to_dict()
            logging.info(f"Loaded {len(self.chexpert_reports)} CheXpert reports.")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            if self.dataset_key == 'MIMIC-CXR-JPG':
                rel = row['filename']
                report_path = os.path.join(MIMIC_REPORTS_BASE_DIR, f"{'/'.join(rel.split('/')[0:4])}.txt")
                with open(report_path, 'r') as f: imp = get_impression(f.read())
            elif self.dataset_key == 'chexpert':
                chexpert_path_key = row['filename']
                imp = self.chexpert_reports.get(chexpert_path_key, "")
                if pd.isna(imp): imp = ""
            elif self.dataset_key == 'rexgradient':
                imp = str(row.get('report_text', ''))
                if pd.isna(imp) or imp == 'nan': imp = ""
            else:
                imp = ""
            if not imp: return None
            return {'text': imp, 'labels': row.to_dict()}
        except Exception: return None

def custom_collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    return {'texts': [b['text'] for b in batch], 'labels': [b['labels'] for b in batch]}

def extract_text_layers(dataset, model, processor, output_dir):
    global _current_attention_mask
    lang_model = model.model.language_model
    
    layers_to_probe = OrderedDict({'Txt_Embed': lang_model.embed_tokens})
    for i, l in enumerate(lang_model.layers): layers_to_probe[f'Txt_Block_{i}'] = l
    layers_to_probe['Txt_FinalNorm'] = lang_model.norm

    all_layer_features = defaultdict(list)
    all_labels = []
    chunk_size, chunk_idx = 20000, 0
    chunks_dir = os.path.join(output_dir, "temp_chunks")

    dataloader = DataLoader(dataset, batch_size=OPTIMIZED_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=custom_collate)

    for batch in tqdm(dataloader, desc="Extracting Text Layers"):
        if not batch: continue
        
        messages = [{"role": "user", "content": [{"type": "text", "text": batch['texts'][0]}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=text, return_tensors="pt").to(DEVICE)
        
        _current_attention_mask = inputs.get('attention_mask', None)
        storage = {}

        with torch.no_grad():
            handles = [m.register_forward_hook(partial(capture_text_hook, layer_name=n, storage_dict=storage)) for n, m in layers_to_probe.items()]
            _ = lang_model(input_ids=inputs['input_ids'], attention_mask=_current_attention_mask)
            for h in handles: h.remove()
            _current_attention_mask = None
            
            for n, f in storage.items(): all_layer_features[n].append(f)
            all_labels.extend(batch['labels'])

        if len(all_labels) >= chunk_size:
            from nvreason_extract_image_layers import save_chunk
            save_chunk(chunks_dir, chunk_idx, all_layer_features, all_labels)
            chunk_idx += 1; all_layer_features.clear(); all_labels.clear(); gc.collect()

    if all_labels:
        from nvreason_extract_image_layers import save_chunk, combine_chunks
        save_chunk(chunks_dir, chunk_idx, all_layer_features, all_labels)
        combine_chunks(chunks_dir, output_dir, list(layers_to_probe.keys()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_folder_name', type=str, default='MIMIC-CXR-JPG')
    parser.add_argument('--split_value', type=int, required=True)
    args = parser.parse_args()

    split_name = {0: "train", 1: "val", 2: "test"}[args.split_value]
    output_dir = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, args.dataset_folder_name, f"features_text_only_{split_name}")
    os.makedirs(output_dir, exist_ok=True)

    model = AutoModelForImageTextToText.from_pretrained(NVREASON_MODEL_PATH, torch_dtype=torch.float16).eval().to(DEVICE)
    processor = AutoProcessor.from_pretrained(NVREASON_MODEL_PATH)
    
    dataset = NVReasonTextDataset(DATASET_CONFIGS[args.dataset_folder_name]['metadata_attr_lr_file'], args.split_value, dataset_key=args.dataset_folder_name)
    extract_text_layers(dataset, model, processor, output_dir)