"""
evaluate_rexgradient_full_test_nonpediatric.py

Evaluate RexGradient-trained probes on the NON-PEDIATRIC subset of the
RexGradient test split (excludes age group 0 = 0-17 years).

This makes the RexGradient evaluation comparable to MIMIC-CXR and CheXpert,
which have 0% pediatric patients.

Key differences from evaluate_rexgradient_full_test.py:
  - Excludes all patients with age == 0 (0-17 years)
  - Age demographic uses only classes [1, 2, 3, 4] (18-39, 40-59, 60-79, 80+)
  - Output dirs have "_nonpediatric" suffix

Output directories:
  - evaluation_results/{model}/evaluation_results_{modality}_full_test_rexgradient_nonpediatric/
  - evaluation_results/{model}/evaluation_results_{modality}_nofinding_full_test_rexgradient_nonpediatric/

Usage:
  python evaluate_rexgradient_full_test_nonpediatric.py --n_jobs 36
  python evaluate_rexgradient_full_test_nonpediatric.py --models nv_reason chexzero --n_jobs 36
"""

import sys
import os

sys.stdout.reconfigure(line_buffering=True)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pandas as pd
import numpy as np
import joblib
import json
import logging
import warnings
import argparse
from collections import defaultdict
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_recall_curve
from joblib import Parallel, delayed

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# --- Global Configurations ---
BASE_DIR = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models"
RESULTS_BASE = os.path.join(BASE_DIR, "evaluation_results")

TARGET_MODELS = ["medgemma_1p5", "biovilt", "radfm", "llavamed1p5", "nv_reason", "chexagent", "medversa", "chexzero"]
DATASET = "rexgradient"
N_BOOTSTRAP_SAMPLES = 1000

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

# RexGradient only has sex and age (no ethnicity)
# Age uses only adult classes: 1=18-39, 2=40-59, 3=60-79, 4=80+
DEMO_ATTRS = ['sex', 'age']


# =============================================================================
# Helper functions (same as original)
# =============================================================================

def bootstrap_auroc_ci(y_true, y_pred_proba, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    scores = []
    n_samples = len(y_true)
    is_binary = y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 2

    for _ in range(n_bootstraps):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred_proba[indices]
        if len(np.unique(y_true_boot)) < 2:
            continue
        try:
            if is_binary:
                proba = y_pred_boot[:, 1] if y_pred_boot.ndim == 2 else y_pred_boot
                score = roc_auc_score(y_true_boot, proba)
            else:
                score = roc_auc_score(y_true_boot, y_pred_boot, multi_class='ovr')
            scores.append(score)
        except ValueError:
            continue

    if not scores:
        return {'mean': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}
    return {
        'mean': float(np.mean(scores)),
        'lower_ci': float(np.percentile(scores, 2.5)),
        'upper_ci': float(np.percentile(scores, 97.5))
    }


def bootstrap_auroc_ci_1d(y_true, y_pred_proba_1d, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    scores = []
    n_samples = len(y_true)
    if n_samples == 0 or len(np.unique(y_true)) < 2:
        return {'mean': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}

    for _ in range(n_bootstraps):
        idx = np.random.choice(n_samples, n_samples, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            scores.append(roc_auc_score(yt, y_pred_proba_1d[idx]))
        except ValueError:
            continue

    if not scores:
        return {'mean': np.nan, 'lower_ci': np.nan, 'upper_ci': np.nan}
    return {
        'mean': float(np.mean(scores)),
        'lower_ci': float(np.percentile(scores, 2.5)),
        'upper_ci': float(np.percentile(scores, 97.5))
    }


def calculate_binary_metrics(y_true, y_pred_labels, y_pred_proba=None):
    metrics = {'auc': np.nan, 'tpr': np.nan, 'fpr': np.nan, 'tnr': np.nan, 'fnr': np.nan, 'n_samples': len(y_true)}
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return metrics
    if y_pred_proba is not None:
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


def get_optimal_f1_threshold(y_true, y_pred_proba):
    if len(np.unique(y_true)) < 2:
        return 0.5
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    f1_scores = np.divide(
        2 * (precisions * recalls), (precisions + recalls),
        out=np.zeros_like(precisions), where=(precisions + recalls) != 0
    )
    best_idx = np.argmax(f1_scores)
    if best_idx < len(thresholds):
        return float(thresholds[best_idx])
    return 0.5


def _bool_mask(series_or_comparison):
    return np.asarray(series_or_comparison, dtype=bool)


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    return obj


# =============================================================================
# DEMOGRAPHIC EVALUATION (per layer) - NON-PEDIATRIC
# =============================================================================

def _process_layer_demographics(layer_name, model, modality, feat_dir, model_dir,
                                y_test_df, adult_indices, demographics, log_prefix):
    """Process one layer for adult-only subset."""
    import warnings
    warnings.filterwarnings("ignore")

    layer_results = defaultdict(dict)

    try:
        x_test_raw = np.load(os.path.join(feat_dir, f"{layer_name}_embeddings.npy"), mmap_mode='r')
    except FileNotFoundError:
        return layer_name, None

    # Subset to adults only
    x_test_adults = x_test_raw[adult_indices]
    y_adults_df = y_test_df.iloc[adult_indices].reset_index(drop=True)

    for attribute_probe in demographics.keys():
        try:
            probe_model = joblib.load(os.path.join(model_dir, f"probe_{layer_name}_{attribute_probe}.joblib"))
            scaler = joblib.load(os.path.join(model_dir, f"scaler_{layer_name}_{attribute_probe}.joblib"))
        except FileNotFoundError:
            continue

        x_scaled = scaler.transform(x_test_adults)
        y_pred_proba_full = probe_model.predict_proba(x_scaled)
        y_pred_labels_full = probe_model.predict(x_scaled)

        # Tier 1: Overall AUC with bootstrap CI
        y_true_overall = y_adults_df[attribute_probe].to_numpy(dtype=float, na_value=np.nan)
        valid_mask = ~np.isnan(y_true_overall)
        layer_results[attribute_probe]['overall_auc_ci'] = bootstrap_auroc_ci(
            y_true_overall[valid_mask].astype(int),
            y_pred_proba_full[valid_mask],
            n_bootstraps=N_BOOTSTRAP_SAMPLES
        )

        # Tier 2: Per-class binary metrics
        layer_results[attribute_probe]['per_class'] = {}
        for class_val in demographics[attribute_probe]:
            if class_val not in probe_model.classes_:
                continue
            y_true_binary = _bool_mask(y_adults_df[attribute_probe] == class_val)
            class_idx = np.where(probe_model.classes_ == class_val)[0][0]
            metrics = calculate_binary_metrics(
                y_true_binary,
                (y_pred_labels_full == class_val),
                y_pred_proba_full[:, class_idx]
            )
            layer_results[attribute_probe]['per_class'][f'class_{class_val}'] = metrics

        # Tier 3: Conditioned metrics
        layer_results[attribute_probe]['conditioned'] = defaultdict(dict)
        other_demographics = {k: v for k, v in demographics.items() if k != attribute_probe}

        for class_val in demographics[attribute_probe]:
            if class_val not in probe_model.classes_:
                continue
            class_idx = np.where(probe_model.classes_ == class_val)[0][0]
            for cond_attr, cond_classes in other_demographics.items():
                for cond_val in cond_classes:
                    subset_mask = _bool_mask(y_adults_df[cond_attr] == cond_val)
                    if subset_mask.sum() < 10:
                        continue
                    y_true_subset = _bool_mask(y_adults_df.loc[subset_mask, attribute_probe] == class_val)
                    y_pred_labels_subset = (y_pred_labels_full[subset_mask] == class_val)
                    y_pred_proba_subset = y_pred_proba_full[subset_mask, class_idx]
                    metrics = calculate_binary_metrics(y_true_subset, y_pred_labels_subset, y_pred_proba_subset)
                    layer_results[attribute_probe]['conditioned'][f'class_{class_val}'][f'given_{cond_attr}_{cond_val}'] = metrics

    return layer_name, dict(layer_results) if layer_results else None


# =============================================================================
# NO FINDING EVALUATION (per layer) - NON-PEDIATRIC
# =============================================================================

def _process_layer_nofinding(layer_name, model, modality, feat_dir_test, feat_dir_val,
                             model_dir, y_test_df, y_val_df, adult_indices_test,
                             adult_indices_val, demographics, log_prefix):
    """Process one layer for adult-only No Finding evaluation."""
    import warnings
    warnings.filterwarnings("ignore")

    try:
        x_test_raw = np.load(os.path.join(feat_dir_test, f"{layer_name}_embeddings.npy"), mmap_mode='r')
    except FileNotFoundError:
        return layer_name, None

    try:
        probe_model = joblib.load(os.path.join(model_dir, f"probe_{layer_name}_No_Finding.joblib"))
        scaler = joblib.load(os.path.join(model_dir, f"scaler_{layer_name}_No_Finding.joblib"))
    except FileNotFoundError:
        return layer_name, None

    # Subset to adults
    x_test_adults = x_test_raw[adult_indices_test]
    y_adults_df = y_test_df.iloc[adult_indices_test].reset_index(drop=True)

    x_test_scaled = scaler.transform(x_test_adults)
    y_pred_proba_test = probe_model.predict_proba(x_test_scaled)
    pos_class_idx = np.where(probe_model.classes_ == 1)[0][0]
    y_pred_proba_pos = y_pred_proba_test[:, pos_class_idx]

    # Get optimal threshold from validation set (adults only)
    try:
        x_val_raw = np.load(os.path.join(feat_dir_val, f"{layer_name}_embeddings.npy"), mmap_mode='r')
        x_val_adults = x_val_raw[adult_indices_val]
        x_val_scaled = scaler.transform(x_val_adults)
        y_pred_proba_val = probe_model.predict_proba(x_val_scaled)[:, pos_class_idx]
        y_val_adults_df = y_val_df.iloc[adult_indices_val].reset_index(drop=True)
        y_val_nf = y_val_adults_df['No Finding'].values.astype(int)
        optimal_threshold = get_optimal_f1_threshold(y_val_nf, y_pred_proba_val)
    except Exception:
        optimal_threshold = 0.5

    y_pred_labels = (y_pred_proba_pos >= optimal_threshold).astype(int)
    y_true_nf = y_adults_df['No Finding'].values.astype(int)

    # Overall AUC with bootstrap CI
    overall_auc = bootstrap_auroc_ci_1d(y_true_nf, y_pred_proba_pos, N_BOOTSTRAP_SAMPLES)
    overall_metrics = calculate_binary_metrics(y_true_nf, y_pred_labels, y_pred_proba_pos)

    # Conditioned metrics by demographics
    conditioned = {}
    for demo_attr in DEMO_ATTRS:
        if demo_attr not in demographics:
            continue
        for demo_val in demographics[demo_attr]:
            mask = _bool_mask(y_adults_df[demo_attr] == demo_val)
            if mask.sum() < 10:
                continue
            sub_auc = bootstrap_auroc_ci_1d(y_true_nf[mask], y_pred_proba_pos[mask], N_BOOTSTRAP_SAMPLES)
            sub_metrics = calculate_binary_metrics(y_true_nf[mask], y_pred_labels[mask], y_pred_proba_pos[mask])
            conditioned[f'given_{demo_attr}_{demo_val}'] = {
                'auc_ci': sub_auc,
                'metrics': sub_metrics
            }

    layer_result = {
        'No_Finding': {
            'overall_auc_ci': overall_auc,
            'overall_metrics': overall_metrics,
            'optimal_threshold': optimal_threshold,
            'conditioned': conditioned
        }
    }

    return layer_name, layer_result


# =============================================================================
# MAIN EVALUATION ORCHESTRATORS
# =============================================================================

def evaluate_demographics(model, modality, n_jobs):
    """Evaluate demographic probes on non-pediatric RexGradient test."""
    log_prefix = f"[{model} | {modality.upper()} | DEMO | ADULT]"
    logging.info(f"{log_prefix} Starting non-pediatric demographic evaluation...")

    exp_dir = os.path.join(BASE_DIR, model, "probe_experiment_outputs", DATASET)
    feat_dir = os.path.join(exp_dir, f"features_{modality}_only_test")
    # Vision: use nonpediatric probes; Text: always use old all-layer probes
    if modality == "text":
        model_dir = os.path.join(exp_dir, f"trained_probes_{modality}_only")
    else:
        model_dir = os.path.join(exp_dir, f"trained_probes_{modality}_only_nonpediatric")
    out_dir = os.path.join(RESULTS_BASE, model, f"evaluation_results_{modality}_full_test_rexgradient_nonpediatric")

    if not os.path.exists(model_dir):
        logging.warning(f"{log_prefix} Probe dir missing: {model_dir}. Skipped.")
        return
    if not os.path.isdir(feat_dir):
        logging.warning(f"{log_prefix} Feature dir missing: {feat_dir}. Skipped.")
        return

    try:
        y_test_df = pd.read_csv(os.path.join(feat_dir, "labels_and_metadata.csv"))
    except FileNotFoundError:
        logging.error(f"{log_prefix} Labels file missing. Skipped.")
        return

    # Clean demographic columns
    for col in DEMO_ATTRS:
        y_test_df[col] = pd.to_numeric(
            y_test_df[col].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0],
            errors='coerce'
        ).astype('Int64')

    # EXCLUDE PEDIATRIC (age == 0, i.e., 0-17 years)
    adult_mask = y_test_df['age'] >= 1
    adult_indices = np.where(adult_mask.values)[0]
    n_excluded = len(y_test_df) - len(adult_indices)
    logging.info(f"{log_prefix} Total: {len(y_test_df)}, Adults: {len(adult_indices)}, Excluded pediatric: {n_excluded}")

    # For age, only use classes 1-4 (no class 0)
    demographics = {
        'sex': sorted(y_test_df.loc[adult_mask, 'sex'].dropna().unique()),
        'age': sorted([v for v in y_test_df.loc[adult_mask, 'age'].dropna().unique() if v >= 1]),
    }
    logging.info(f"{log_prefix} Demographics: {demographics}")

    layers = MODEL_LAYERS[model][modality]
    os.makedirs(out_dir, exist_ok=True)

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_layer_demographics)(
            layer, model, modality, feat_dir, model_dir, y_test_df, adult_indices, demographics, log_prefix
        ) for layer in layers
    )

    all_results = {}
    for layer_name, layer_res in results:
        if layer_res is not None:
            all_results[layer_name] = layer_res

    if all_results:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(out_dir, f"demographics_full_test_{modality}_{DATASET}_nonpediatric_{timestamp}.json")
        with open(out_file, 'w') as f:
            json.dump(convert_numpy_types(all_results), f, indent=4)
        logging.info(f"{log_prefix} Saved: {out_file}")


def evaluate_nofinding(model, modality, n_jobs):
    """Evaluate No Finding probe on non-pediatric RexGradient test."""
    log_prefix = f"[{model} | {modality.upper()} | NF | ADULT]"
    logging.info(f"{log_prefix} Starting non-pediatric No Finding evaluation...")

    exp_dir = os.path.join(BASE_DIR, model, "probe_experiment_outputs", DATASET)
    feat_dir_test = os.path.join(exp_dir, f"features_{modality}_only_test")
    feat_dir_val = os.path.join(exp_dir, f"features_{modality}_only_val")
    # Vision: use nonpediatric probes; Text: always use old all-layer probes
    if modality == "text":
        model_dir = os.path.join(exp_dir, f"trained_probes_{modality}_only")
    else:
        model_dir = os.path.join(exp_dir, f"trained_probes_{modality}_only_nonpediatric")
    out_dir = os.path.join(RESULTS_BASE, model, f"evaluation_results_{modality}_nofinding_full_test_rexgradient_nonpediatric")

    if not os.path.exists(model_dir):
        logging.warning(f"{log_prefix} Probe dir missing: {model_dir}. Skipped.")
        return
    if not os.path.isdir(feat_dir_test):
        logging.warning(f"{log_prefix} Test feature dir missing: {feat_dir_test}. Skipped.")
        return

    try:
        y_test_df = pd.read_csv(os.path.join(feat_dir_test, "labels_and_metadata.csv"))
    except FileNotFoundError:
        logging.error(f"{log_prefix} Test labels missing. Skipped.")
        return

    try:
        y_val_df = pd.read_csv(os.path.join(feat_dir_val, "labels_and_metadata.csv"))
    except FileNotFoundError:
        logging.warning(f"{log_prefix} Val labels missing; using threshold=0.5.")
        y_val_df = pd.DataFrame()

    # Clean columns
    for col in DEMO_ATTRS:
        y_test_df[col] = pd.to_numeric(
            y_test_df[col].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0],
            errors='coerce'
        ).astype('Int64')
        if not y_val_df.empty:
            y_val_df[col] = pd.to_numeric(
                y_val_df[col].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0],
                errors='coerce'
            ).astype('Int64')

    y_test_df['No Finding'] = pd.to_numeric(
        y_test_df['No Finding'].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0],
        errors='coerce'
    ).fillna(0).astype(int)

    if not y_val_df.empty:
        y_val_df['No Finding'] = pd.to_numeric(
            y_val_df['No Finding'].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0],
            errors='coerce'
        ).fillna(0).astype(int)

    # EXCLUDE PEDIATRIC
    adult_mask_test = y_test_df['age'] >= 1
    adult_indices_test = np.where(adult_mask_test.values)[0]

    adult_indices_val = np.array([], dtype=int)
    if not y_val_df.empty:
        adult_mask_val = y_val_df['age'] >= 1
        adult_indices_val = np.where(adult_mask_val.values)[0]

    logging.info(f"{log_prefix} Test adults: {len(adult_indices_test)}, Val adults: {len(adult_indices_val)}")

    # Demographics for conditioning (adults only, age classes 1-4)
    demographics = {
        'sex': sorted(y_test_df.loc[adult_mask_test, 'sex'].dropna().unique()),
        'age': sorted([v for v in y_test_df.loc[adult_mask_test, 'age'].dropna().unique() if v >= 1]),
    }

    layers = MODEL_LAYERS[model][modality]
    os.makedirs(out_dir, exist_ok=True)

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_layer_nofinding)(
            layer, model, modality, feat_dir_test, feat_dir_val,
            model_dir, y_test_df, y_val_df, adult_indices_test,
            adult_indices_val, demographics, log_prefix
        ) for layer in layers
    )

    all_results = {}
    for layer_name, layer_res in results:
        if layer_res is not None:
            all_results[layer_name] = layer_res

    if all_results:
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(out_dir, f"nofinding_full_test_{modality}_{DATASET}_nonpediatric_{timestamp}.json")
        with open(out_file, 'w') as f:
            json.dump(convert_numpy_types(all_results), f, indent=4)
        logging.info(f"{log_prefix} Saved: {out_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RexGradient probes - NON-PEDIATRIC only (age >= 18)")
    parser.add_argument('--models', nargs='+', default=TARGET_MODELS,
                        help='Models to evaluate')
    parser.add_argument('--n_jobs', type=int, default=36,
                        help='Number of parallel layer jobs (default: 36)')
    parser.add_argument('--modalities', nargs='+', default=['vision', 'text'],
                        choices=['vision', 'text'])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(asctime)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)],
                        force=True)

    logging.info("=" * 70)
    logging.info("  REXGRADIENT NON-PEDIATRIC EVALUATION (age >= 18 only)")
    logging.info(f"  Models: {args.models}")
    logging.info(f"  Modalities: {args.modalities}")
    logging.info(f"  n_jobs: {args.n_jobs}")
    logging.info(f"  Probes: sex (adults only), age (classes 1-4 only), No_Finding (adults only)")
    logging.info(f"  Age gap: given_age_1 (18-39) vs given_age_4 (80+)")
    logging.info(f"  Output: evaluation_results/{{model}}/..._rexgradient_nonpediatric/")
    logging.info("=" * 70)

    for model in args.models:
        if model not in MODEL_LAYERS:
            logging.warning(f"Unknown model: {model}. Skipping.")
            continue
        for modality in args.modalities:
            evaluate_demographics(model, modality, args.n_jobs)
            evaluate_nofinding(model, modality, args.n_jobs)

    logging.info("=" * 70)
    logging.info("  NON-PEDIATRIC EVALUATION COMPLETE")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
