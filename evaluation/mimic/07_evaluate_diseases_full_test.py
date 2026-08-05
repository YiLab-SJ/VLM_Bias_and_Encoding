"""
07_evaluate_diseases_full_test.py

Evaluate the multi-label disease probe on the FULL test split
(no balanced subset filtering). For each model/modality/layer, compute per disease:
  - Overall AUC with bootstrap CI
  - Overall binary metrics (TPR, FPR, TNR, FNR) using optimal F1 threshold derived from Validation set
  - Conditioned metrics: disease performance sliced by each
    demographic attribute (sex, age, ethnicity) subgroup
Also compute macro AUC across all diseases.

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
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from joblib import Parallel, delayed

# --- Global Configurations ---
BASE_DIR = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models"
RESULTS_BASE = os.path.join(BASE_DIR, "evaluation_results")

TARGET_MODELS = ["medgemma_1p5", "biovilt", "radfm", "llavamed1p5", "nv_reason", "chexagent", "medversa", "chexzero"]
DATASET = "MIMIC-CXR-JPG"
N_BOOTSTRAP_SAMPLES = 1000

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]
DEMO_LABELS = ['sex', 'age', 'ethnicity']

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


def _find_disease_probe_dir(exp_dir, modality):
    """Auto-detect disease probe directory (naming varies across models: image vs vision)."""
    candidates = [
        os.path.join(exp_dir, f"trained_probes_{'image' if modality=='vision' else 'text'}_diseases"),
        os.path.join(exp_dir, f"trained_probes_{modality}_diseases"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# --- Helper Functions ---
def bootstrap_auroc_ci_1d(y_true, y_pred_proba, n_bootstraps=1000, random_state=42):
    """1D Bootstrap AUC for a single binary label."""
    np.random.seed(random_state)
    scores = []
    n_samples = len(y_true)
    if n_samples == 0 or len(np.unique(y_true)) < 2:
        return {'mean': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}

    for _ in range(n_bootstraps):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        yt, yp = y_true[idx], y_pred_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(roc_auc_score(yt, yp))
        except ValueError:
            continue

    if not scores:
        return {'mean': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}
    return {
        'mean': float(np.mean(scores)),
        'lower_ci': float(np.percentile(scores, 2.5)),
        'upper_ci': float(np.percentile(scores, 97.5))
    }


def calculate_binary_metrics(y_true, y_pred_labels):
    """Calculate TPR, FPR, TNR, FNR and sample counts."""
    metrics = {'tpr': np.nan, 'fpr': np.nan, 'tnr': np.nan, 'fnr': np.nan, 'n_samples': len(y_true)}
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return metrics
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
    """Safely convert a pandas Series/comparison (possibly nullable BooleanDtype) to a numpy bool array."""
    return np.asarray(series_or_comparison, dtype=bool)


def get_optimal_f1_threshold(y_true, y_pred_proba):
    """Find the threshold that maximizes F1 score on the validation set."""
    if len(np.unique(y_true)) < 2:
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Calculate F1 scores safely avoiding division by zero
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0
    )
    
    # Identify the threshold that yields the highest F1 score
    best_idx = np.argmax(f1_scores)
    
    # thresholds array is len(precisions) - 1. Handle bounds.
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    else:
        return float(thresholds[-1]) if len(thresholds) > 0 else 0.5


def _process_single_layer_disease(layer_name, model, modality, feat_dir_test, feat_dir_val, model_dir, y_disease_true_test, y_disease_true_val, disease_cols, demographics, y_test_df, log_prefix):
    import warnings
    warnings.filterwarnings("ignore")
    
    # Extreme verbosity prefix for loky workers
    v_prefix = f"[{model.upper()} | {modality.upper()} | {layer_name}]"
    print(f"{v_prefix} STEP 1/6: Worker initialized. Beginning data load...", flush=True)

    # Load test and val embeddings
    try:
        x_test_raw = np.load(os.path.join(feat_dir_test, f"{layer_name}_embeddings.npy"), mmap_mode='r')
        x_val_raw = np.load(os.path.join(feat_dir_val, f"{layer_name}_embeddings.npy"), mmap_mode='r')
        print(f"{v_prefix} STEP 2/6: .npy embeddings loaded successfully.", flush=True)
    except FileNotFoundError:
        print(f"{v_prefix} ERROR: Feature file for '{layer_name}' not found. Skipping.", flush=True)
        return model, modality, layer_name, None, None

    # Load disease probe + scaler
    try:
        probe_model = joblib.load(os.path.join(model_dir, f"disease_probe_{layer_name}.joblib"))
        scaler = joblib.load(os.path.join(model_dir, f"disease_scaler_{layer_name}.joblib"))
        print(f"{v_prefix} STEP 3/6: Probes and Scaler loaded successfully.", flush=True)
        
        # --- SKLEARN VERSION MISMATCH FIX ---
        # Inject multi_class='ovr' if it is missing from the unpickled model
        if hasattr(probe_model, 'estimators_'):
            for est in probe_model.estimators_:
                if not hasattr(est, 'multi_class'):
                    est.multi_class = 'ovr'
        elif not hasattr(probe_model, 'multi_class'):
            probe_model.multi_class = 'ovr'
        # ------------------------------------
        
    except FileNotFoundError:
        print(f"{v_prefix} ERROR: Disease probe for '{layer_name}' not found. Skipping.", flush=True)
        return model, modality, layer_name, None, None

    print(f"{v_prefix} STEP 4/6: Scaling features and generating probability predictions...", flush=True)
    x_test_scaled = scaler.transform(x_test_raw)
    x_val_scaled = scaler.transform(x_val_raw)

    # Predict probabilities for Test
    y_pred_proba_all_test = probe_model.predict_proba(x_test_scaled)
    if isinstance(y_pred_proba_all_test, list):
        y_pred_proba_all_test = np.column_stack([p[:, 1] if p.shape[1] == 2 else p for p in y_pred_proba_all_test])

    # Predict probabilities for Val
    y_pred_proba_all_val = probe_model.predict_proba(x_val_scaled)
    if isinstance(y_pred_proba_all_val, list):
        y_pred_proba_all_val = np.column_stack([p[:, 1] if p.shape[1] == 2 else p for p in y_pred_proba_all_val])

    layer_results = {}
    layer_macro_aucs = []

    print(f"{v_prefix} STEP 5/6: Beginning disease iteration and bootstrap computations...", flush=True)
    # --- Per-disease evaluation ---
    for j, disease_name in enumerate(disease_cols):
        
        # Validation Truth and Probs for Threshold Tuning
        y_t_d_val = y_disease_true_val[:, j]
        y_prob_d_val = y_pred_proba_all_val[:, j]
        
        # Determine Optimal Threshold on Validation Set
        opt_thresh = get_optimal_f1_threshold(y_t_d_val, y_prob_d_val)
        print(f"    -> {v_prefix} Threshold Found: {disease_name} | F1-Opt: {opt_thresh:.4f}", flush=True)
        
        # Test Truth and Probs
        y_t_d_test = y_disease_true_test[:, j]
        y_prob_d_test = y_pred_proba_all_test[:, j]
        
        # Apply optimal threshold to test probabilities to get binary labels
        y_pred_d_test = (y_prob_d_test >= opt_thresh).astype(int)

        disease_results = {}
        disease_results['optimal_threshold'] = opt_thresh

        # 1. Overall metrics
        overall_auc_ci = bootstrap_auroc_ci_1d(y_t_d_test, y_prob_d_test, n_bootstraps=N_BOOTSTRAP_SAMPLES)
        overall_metrics = calculate_binary_metrics(y_t_d_test, y_pred_d_test)
        disease_results['overall_auc_ci'] = overall_auc_ci
        disease_results['overall_metrics'] = overall_metrics

        if not np.isnan(overall_auc_ci['mean']):
            layer_macro_aucs.append(overall_auc_ci['mean'])

        # 2. Conditioned on demographics
        disease_results['conditioned'] = {}
        for demo_attr, demo_classes in demographics.items():
            for demo_val in demo_classes:
                mask = _bool_mask(y_test_df[demo_attr] == demo_val)

                y_t_sub = y_t_d_test[mask]
                y_prob_sub = y_prob_d_test[mask]
                y_pred_sub = y_pred_d_test[mask]

                sub_auc_ci = bootstrap_auroc_ci_1d(y_t_sub, y_prob_sub, n_bootstraps=N_BOOTSTRAP_SAMPLES)
                sub_metrics = calculate_binary_metrics(y_t_sub, y_pred_sub)

                disease_results['conditioned'][f'given_{demo_attr}_{demo_val}'] = {
                    'auc_ci': sub_auc_ci,
                    'metrics': sub_metrics
                }

        layer_results[disease_name] = disease_results
        print(f"    -> {v_prefix} Bootstrapping completed for {disease_name}.", flush=True)

    # Macro AUC
    layer_results['macro_auc'] = {
        'mean': float(np.mean(layer_macro_aucs)) if layer_macro_aucs else np.nan
    }
    
    print(f"{v_prefix} STEP 6/6: TASK COMPLETED! Macro AUC: {layer_results['macro_auc']['mean']:.4f}", flush=True)
    return model, modality, layer_name, layer_results, layer_results['macro_auc']['mean']


# --- Core Evaluator Task Preparation ---
def prepare_evaluation_tasks(model, modality):
    """Prepares metadata and returns a list of arguments for parallel processing."""
    log_prefix = f"[{model} | {modality.upper()} | DISEASE]"
    logging.info(f"{log_prefix} --- Starting FULL TEST disease evaluation for: {DATASET} ---")

    exp_dir = os.path.join(BASE_DIR, model, "probe_experiment_outputs", DATASET)
    feat_dir_test = os.path.join(exp_dir, f"features_{modality}_only_test")
    feat_dir_val = os.path.join(exp_dir, f"features_{modality}_only_val")
    model_dir = _find_disease_probe_dir(exp_dir, modality)
    out_dir = os.path.join(RESULTS_BASE, model, f"evaluation_results_{modality}_disease_full_test")

    if model_dir is None:
        logging.info(f"{log_prefix} Disease probes folder not found in {exp_dir}. Skipped.")
        return [], None, None

    # --- Load the FULL test metadata (no subset filtering) ---
    try:
        y_test_df = pd.read_csv(os.path.join(feat_dir_test, "labels_and_metadata.csv"))
    except FileNotFoundError:
        logging.error(f"{log_prefix} Labels file missing in {feat_dir_test}. Skipped.")
        return [], None, None
        
    # --- Load the FULL val metadata ---
    try:
        y_val_df = pd.read_csv(os.path.join(feat_dir_val, "labels_and_metadata.csv"))
    except FileNotFoundError:
        logging.error(f"{log_prefix} Validation labels file missing in {feat_dir_val}. Skipped.")
        return [], None, None

    y_test_df['dicom_id'] = y_test_df['dicom_id'].astype(str)
    y_val_df['dicom_id'] = y_val_df['dicom_id'].astype(str)

    # Clean disease columns (Test)
    disease_cols = [d for d in DISEASE_LABELS if d in y_test_df.columns]
    for col in disease_cols:
        y_test_df[col] = pd.to_numeric(
            y_test_df[col].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0],
            errors='coerce'
        )
    y_test_df.dropna(subset=disease_cols, inplace=True)
    for col in disease_cols:
        y_test_df[col] = y_test_df[col].astype(int)
        
    # Clean disease columns (Val)
    for col in disease_cols:
        if col in y_val_df.columns:
            y_val_df[col] = pd.to_numeric(
                y_val_df[col].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0],
                errors='coerce'
            )
    y_val_df.dropna(subset=disease_cols, inplace=True)
    for col in disease_cols:
        if col in y_val_df.columns:
            y_val_df[col] = y_val_df[col].astype(int)

    # Clean demographic columns (Test)
    for col in DEMO_LABELS:
        if col in y_test_df.columns:
            y_test_df[col] = pd.to_numeric(
                y_test_df[col].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0],
                errors='coerce'
            ).astype('Int64')

    y_test_df = y_test_df.reset_index(drop=True)
    y_disease_true_test = y_test_df[disease_cols].values.astype(float)
    
    y_val_df = y_val_df.reset_index(drop=True)
    y_disease_true_val = y_val_df[disease_cols].values.astype(float)

    # Get demographic subgroups
    demographics = {}
    for attr in DEMO_LABELS:
        if attr in y_test_df.columns:
            demographics[attr] = sorted(y_test_df[attr].dropna().unique())

    logging.info(f"{log_prefix} Full test set: {len(y_test_df)} samples. Full val set: {len(y_val_df)} samples. "
                 f"Diseases: {len(disease_cols)}. Demographics: {list(demographics.keys())}")

    tasks = []
    for layer_name in MODEL_LAYERS[model][modality]:
        tasks.append({
            'layer_name': layer_name,
            'model': model,
            'modality': modality,
            'feat_dir_test': feat_dir_test,
            'feat_dir_val': feat_dir_val,
            'model_dir': model_dir,
            'y_disease_true_test': y_disease_true_test,
            'y_disease_true_val': y_disease_true_val,
            'disease_cols': disease_cols,
            'demographics': demographics,
            'y_test_df': y_test_df,
            'log_prefix': log_prefix
        })
        
    return tasks, out_dir, log_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='Model to run (if not set, runs all)')
    parser.add_argument('--n_jobs', type=int, default=132, help='Number of parallel jobs (default optimized for GH200)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s', force=True)

    if args.model:
        models = [args.model]
    else:
        models = TARGET_MODELS

    logging.info(f"Initiating FULL TEST disease evaluation for: {models}")
    
    # Compile the flattened list of every single layer task across all models and modalities
    all_tasks = []
    task_metadata = {}
    
    for model in models:
        for modality in ['vision', 'text']:
            tasks, out_dir, log_prefix = prepare_evaluation_tasks(model, modality)
            if tasks:
                all_tasks.extend(tasks)
                task_metadata[(model, modality)] = (out_dir, log_prefix)
                
    if not all_tasks:
        logging.info("No valid tasks found to process. Exiting.")
        return

    logging.info(f"Executing {len(all_tasks)} total layer tasks in parallel using {args.n_jobs} CPU cores...")

    # Execute all tasks in one massive parallel pool with explicit queue management and extreme verbosity
    parallel_results = Parallel(n_jobs=args.n_jobs, verbose=50, pre_dispatch="1.5*n_jobs")(
        delayed(_process_single_layer_disease)(
            task['layer_name'], task['model'], task['modality'],
            task['feat_dir_test'], task['feat_dir_val'], task['model_dir'],
            task['y_disease_true_test'], task['y_disease_true_val'],
            task['disease_cols'], task['demographics'], task['y_test_df'],
            task['log_prefix']
        ) for task in all_tasks
    )

    # Re-group results by model and modality to match the required JSON structure
    grouped_results = defaultdict(dict)
    for model_name, modality_name, layer_name, layer_res, macro_auc in parallel_results:
        if layer_res is not None:
            grouped_results[(model_name, modality_name)][layer_name] = layer_res

    # Save the JSON files with exact original logging
    for (model_name, modality_name), all_results in grouped_results.items():
        if all_results:
            out_dir, log_prefix = task_metadata[(model_name, modality_name)]
            os.makedirs(out_dir, exist_ok=True)
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            out_file = os.path.join(out_dir, f"diseases_full_test_{modality_name}_{DATASET}_{timestamp}.json")
            with open(out_file, 'w') as f:
                json.dump(convert_numpy_types_for_json(all_results), f, indent=4)
            logging.info(f"{log_prefix} Saved FULL TEST disease results to: {out_file}")
            
    logging.info("Full-test disease evaluation pipeline completed.")

if __name__ == "__main__":
    main()