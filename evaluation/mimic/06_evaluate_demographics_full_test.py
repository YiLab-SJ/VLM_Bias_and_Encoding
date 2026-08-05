"""
06_evaluate_demographics_full_test.py

Evaluate demographic probes (sex, age, ethnicity) on the FULL test split
(no balanced subset filtering). For each model/modality/layer, compute:
  - Overall AUC with bootstrap CI
  - Per-class binary metrics (TPR, FPR, TNR, FNR, AUC)
  - Conditioned metrics (per-class metrics sliced by other demographics)

Models: medgemma_1p5, biovilt, radfm, llavamed1p5, nv_reason, chexagent, medversa, chexzero
"""

import pandas as pd
import numpy as np
import os
import joblib
import json
import logging
import warnings
import argparse
from collections import defaultdict
from sklearn.metrics import roc_auc_score, confusion_matrix
from joblib import Parallel, delayed

# --- Global Configurations ---
BASE_DIR = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models"
RESULTS_BASE = os.path.join(BASE_DIR, "evaluation_results")

TARGET_MODELS = ["medgemma_1p5", "biovilt", "radfm", "llavamed1p5", "nv_reason", "chexagent", "medversa", "chexzero"]
DATASET = "MIMIC-CXR-JPG"
N_BOOTSTRAP_SAMPLES = 1000

# --- Precise Layer Orderings from Configs ---
MODEL_LAYERS = {
    "medgemma_1p5": {
        "vision": ['Vis_Embed'] + [f'Vis_Block_{i}' for i in range(1, 28)] + ['Vis_PostNorm', 'Vis_Projected'],
        "text": ['Txt_Embed'] + [f'Txt_Block_{i}' for i in range(1, 35)] + ['Txt_FinalNorm']
    },
    "biovilt": {
        "vision": ['Vis_ResNet_Layer1', 'Vis_ResNet_Layer2', 'Vis_ResNet_Layer3', 'Vis_ResNet_Layer4', 'Vis_BackboneToViT', 'img_embedding', 'image_embedding_final'],
        "text": ['Txt_TokenEmbed'] + [f'Txt_Block_{i}' for i in range(1, 13)] + ['text_embedding_final']
    },
    "radfm": {
        "vision": ['Vis_PatchEmbed'] + [f'Vis_Block_{i}' for i in range(1, 13)] + ['Vis_Perceiver', 'Vis_Projected'],
        "text": ['Txt_Embed'] + [f'Txt_Block_{i}' for i in range(1, 41)] + ['Txt_FinalNorm']
    },
    "llavamed1p5": {
        "vision": ['Vis_CLIP_Embed'] + [f'Vis_CLIP_Layer_{i}' for i in range(1, 25)] + ['Vis_MM_Projector'],
        "text": ['Txt_Mistral_Embed'] + [f'Txt_Mistral_Layer_{i}' for i in range(1, 33)]
    },
    "nv_reason": {
        "vision": [f'Vis_Block_{i}' for i in range(32)],
        "text": ['Txt_Embed'] + [f'Txt_Block_{i}' for i in range(36)] + ['Txt_FinalNorm']
    },
    "chexagent": {
        "vision": [f'Vis_SigLIP_Block_{i}' for i in range(1, 25)] + ['image_embedding_final'],
        "text": ['Txt_TokenEmbed'] + [f'Txt_PhiBlock_{i}' for i in range(1, 33)] + ['text_embedding_final']
    },
    "medversa": {
        "vision": ['Vis_Embed'] +
                  [f'Vis_S0_B{i}' for i in range(1, 3)] +
                  [f'Vis_S1_B{i}' for i in range(1, 3)] +
                  [f'Vis_S2_B{i}' for i in range(1, 19)] +
                  [f'Vis_S3_B{i}' for i in range(1, 3)] +
                  ['Vis_FinalNorm', 'Vis_LNVision', 'Vis_Projected'],
        "text": ['Txt_Embed'] + [f'Txt_Block_{i}' for i in range(1, 33)] + ['Txt_FinalNorm']
    },
    "chexzero": {
        "vision": [f'Vis_Block_{i}' for i in range(1, 13)] + ['image_embedding_final'],
        "text": ['Txt_TokenEmbed'] + [f'Txt_Block_{i}' for i in range(1, 13)] + ['text_embedding_final']
    }
}

# --- Helper Functions ---
def bootstrap_auroc_ci(y_true, y_pred_proba, n_bootstraps=1000, random_state=42):
    """Bootstrap AUC with 95% CI for binary or multiclass."""
    np.random.seed(random_state)
    bootstrapped_scores = []
    n_samples = len(y_true)
    is_binary = y_pred_proba.shape[1] == 2

    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_proba_boot = y_pred_proba[indices]
        if len(np.unique(y_true_boot)) < 2:
            continue
        try:
            if is_binary:
                score = roc_auc_score(y_true_boot, y_pred_proba_boot[:, 1])
            else:
                score = roc_auc_score(y_true_boot, y_pred_proba_boot, multi_class='ovr', average='macro')
            bootstrapped_scores.append(score)
        except ValueError:
            continue

    if not bootstrapped_scores:
        return {'mean': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}
    return {
        'mean': float(np.mean(bootstrapped_scores)),
        'lower_ci': float(np.percentile(bootstrapped_scores, 2.5)),
        'upper_ci': float(np.percentile(bootstrapped_scores, 97.5))
    }


def calculate_binary_metrics(y_true, y_pred_labels, y_pred_proba):
    """Calculate AUC, TPR, FPR, TNR, FNR for a binary one-vs-rest slice."""
    metrics = {'auc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tnr': np.nan, 'fnr': np.nan, 'n_samples': len(y_true)}
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return metrics
    try:
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
    except ValueError:
        pass
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()
        metrics['tpr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['tnr'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    except ValueError:
        pass
    return metrics


def convert_numpy_types_for_json(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types_for_json(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    return obj


def _bool_mask(series_or_comparison):
    """Safely convert a pandas Series/comparison (possibly nullable BooleanDtype) to a numpy bool array.
    This is required because Int64 columns produce BooleanDtype comparisons whose .values
    returns a pandas BooleanArray (not a numpy array), which numpy refuses to use as an index."""
    return np.asarray(series_or_comparison, dtype=bool)


def _process_single_layer_demo(layer_name, model, modality, feat_dir, model_dir, y_test_df, demographics, log_prefix):
    import warnings
    warnings.filterwarnings("ignore")
    logging.info(f"{log_prefix} ===== Processing Layer: {layer_name} =====")
    
    layer_results = defaultdict(dict)
    
    try:
        x_test_raw = np.load(os.path.join(feat_dir, f"{layer_name}_embeddings.npy"), mmap_mode='r')
    except FileNotFoundError:
        logging.warning(f"{log_prefix}   > Feature file for '{layer_name}' not found. Skipping.")
        return layer_name, None

    for attribute_probe in demographics.keys():
        logging.info(f"{log_prefix}   > Probe: {attribute_probe}")
        try:
            probe_model = joblib.load(os.path.join(model_dir, f"probe_{layer_name}_{attribute_probe}.joblib"))
            scaler = joblib.load(os.path.join(model_dir, f"scaler_{layer_name}_{attribute_probe}.joblib"))
        except FileNotFoundError:
            logging.warning(f"{log_prefix}   > Probe model for {layer_name}/{attribute_probe} not found. Skipping.")
            continue

        x_scaled = scaler.transform(x_test_raw)
        y_pred_proba_full = probe_model.predict_proba(x_scaled)
        y_pred_labels_full = probe_model.predict(x_scaled)

        # --- Tier 1: Overall AUC with bootstrap CI ---
        y_true_overall = y_test_df[attribute_probe].to_numpy(dtype=float, na_value=np.nan)
        valid_mask = ~np.isnan(y_true_overall)
        layer_results[attribute_probe]['overall_auc_ci'] = bootstrap_auroc_ci(
            y_true_overall[valid_mask].astype(int),
            y_pred_proba_full[valid_mask],
            n_bootstraps=N_BOOTSTRAP_SAMPLES
        )

        # --- Tier 2: Per-class binary metrics ---
        layer_results[attribute_probe]['per_class'] = {}
        for class_val in demographics[attribute_probe]:
            y_true_binary = _bool_mask(y_test_df[attribute_probe] == class_val)
            class_idx = np.where(probe_model.classes_ == class_val)[0][0]
            metrics = calculate_binary_metrics(
                y_true_binary,
                (y_pred_labels_full == class_val),
                y_pred_proba_full[:, class_idx]
            )
            layer_results[attribute_probe]['per_class'][f'class_{class_val}'] = metrics

        # --- Tier 3 & 4: Conditioned metrics ---
        layer_results[attribute_probe]['conditioned'] = defaultdict(dict)
        other_demographics = {k: v for k, v in demographics.items() if k != attribute_probe}

        for class_val in demographics[attribute_probe]:
            class_idx = np.where(probe_model.classes_ == class_val)[0][0]

            # Single-attribute conditioning
            for cond_attr, cond_classes in other_demographics.items():
                for cond_val in cond_classes:
                    subset_mask = _bool_mask(y_test_df[cond_attr] == cond_val)
                    y_true_subset = _bool_mask(y_test_df.loc[subset_mask, attribute_probe] == class_val)
                    y_pred_labels_subset = (y_pred_labels_full[subset_mask] == class_val)
                    y_pred_proba_subset = y_pred_proba_full[subset_mask, class_idx]
                    metrics = calculate_binary_metrics(y_true_subset, y_pred_labels_subset, y_pred_proba_subset)
                    layer_results[attribute_probe]['conditioned'][f'class_{class_val}'][f'given_{cond_attr}_{cond_val}'] = metrics

            # Dual-attribute conditioning
            cond_attrs_list = list(other_demographics.keys())
            if len(cond_attrs_list) == 2:
                ca1, ca2 = cond_attrs_list[0], cond_attrs_list[1]
                for v1 in demographics[ca1]:
                    for v2 in demographics[ca2]:
                        subset_mask = _bool_mask((y_test_df[ca1] == v1) & (y_test_df[ca2] == v2))
                        y_true_subset = _bool_mask(y_test_df.loc[subset_mask, attribute_probe] == class_val)
                        y_pred_labels_subset = (y_pred_labels_full[subset_mask] == class_val)
                        y_pred_proba_subset = y_pred_proba_full[subset_mask, class_idx]
                        metrics = calculate_binary_metrics(y_true_subset, y_pred_labels_subset, y_pred_proba_subset)
                        layer_results[attribute_probe]['conditioned'][f'class_{class_val}'][f'given_{ca1}_{v1}_and_{ca2}_{v2}'] = metrics

    return layer_name, dict(layer_results)


# --- Core Evaluator ---
def evaluate_demographics(model, modality, n_jobs):
    log_prefix = f"[{model} | {modality.upper()}]"
    logging.info(f"{log_prefix} --- Starting FULL TEST demographic evaluation for: {DATASET} ---")

    exp_dir = os.path.join(BASE_DIR, model, "probe_experiment_outputs", DATASET)
    feat_dir = os.path.join(exp_dir, f"features_{modality}_only_test")
    model_dir = os.path.join(exp_dir, f"trained_probes_{modality}_only")
    out_dir = os.path.join(RESULTS_BASE, model, f"evaluation_results_{modality}_full_test")

    if not os.path.exists(model_dir):
        logging.info(f"{log_prefix} Trained probes folder missing at {model_dir}. Skipped.")
        return

    # --- Load the FULL test metadata (no subset filtering) ---
    try:
        y_test_df = pd.read_csv(os.path.join(feat_dir, "labels_and_metadata.csv"))
    except FileNotFoundError:
        logging.error(f"{log_prefix} Labels file missing in {feat_dir}. Skipped.")
        return

    y_test_df['dicom_id'] = y_test_df['dicom_id'].astype(str)

    # Clean demographic columns
    for col in ['sex', 'age', 'ethnicity']:
        y_test_df[col] = pd.to_numeric(
            y_test_df[col].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0],
            errors='coerce'
        ).astype('Int64')

    logging.info(f"{log_prefix} Full test set: {len(y_test_df)} samples.")

    demographics = {
        'sex': sorted(y_test_df['sex'].dropna().unique()),
        'age': sorted(y_test_df['age'].dropna().unique()),
        'ethnicity': sorted(y_test_df['ethnicity'].dropna().unique())
    }

    all_results = defaultdict(lambda: defaultdict(dict))
    layers_to_process = MODEL_LAYERS[model][modality]
    os.makedirs(out_dir, exist_ok=True)

    parallel_results = Parallel(n_jobs=n_jobs)(
        delayed(_process_single_layer_demo)(
            layer_name, model, modality, feat_dir, model_dir, y_test_df, demographics, log_prefix
        ) for layer_name in layers_to_process
    )

    # Reconstruct dictionary perfectly in order
    for layer_name, layer_res in parallel_results:
        if layer_res is not None:
            all_results[layer_name] = layer_res

    if all_results:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(out_dir, f"demographics_full_test_{modality}_{DATASET}_{timestamp}.json")
        with open(out_file, 'w') as f:
            json.dump(convert_numpy_types_for_json(all_results), f, indent=4)
        logging.info(f"{log_prefix} Saved FULL TEST demographic results to: {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Model to run (if not set, runs all)')
    parser.add_argument('--n_jobs', type=int, default=14, help='Number of parallel jobs')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s', force=True)

    if args.model:
        models = [args.model]
    else:
        models = TARGET_MODELS

    logging.info(f"Initiating FULL TEST demographic evaluation for: {models}")
    
    for model in models:
        for modality in ['vision', 'text']:
            evaluate_demographics(model, modality, args.n_jobs)
            
    logging.info("Full-test demographic evaluation pipeline completed.")

if __name__ == "__main__":
    main()