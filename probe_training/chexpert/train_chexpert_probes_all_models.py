#!/usr/bin/env python
"""
train_chexpert_probes_all_models.py

Train demographic probes (sex, age, ethnicity) and a No-Finding disease probe
on CheXpert extracted features for ALL 8 models, both vision and text modalities.

Designed to be extremely conservative with memory:
  - n_jobs=1 for all sklearn operations
  - One layer at a time, aggressive garbage collection
  - One model at a time (sequential)

Usage:
    python train_chexpert_probes_all_models.py
    python train_chexpert_probes_all_models.py --models medversa chexzero
    python train_chexpert_probes_all_models.py --models medversa --modalities vision
"""

import os
import sys
import argparse
import logging
import gc
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42

# =============================================================================
# MODEL DEFINITIONS: layer names for each model (vision + text)
# =============================================================================
OTHER_MODELS_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models"

MODEL_LAYERS = {
    "medgemma_1p5": {
        "vision": (["Vis_Embed"] +
                   [f"Vis_Block_{i}" for i in range(1, 28)] +
                   ["Vis_PostNorm", "Vis_Projected"]),
        "text": (["Txt_Embed"] +
                 [f"Txt_Block_{i}" for i in range(1, 35)] +
                 ["Txt_FinalNorm"]),
    },
    "biovilt": {
        "vision": ["Vis_ResNet_Layer1", "Vis_ResNet_Layer2", "Vis_ResNet_Layer3",
                   "Vis_ResNet_Layer4", "Vis_BackboneToViT", "img_embedding",
                   "image_embedding_final"],
        "text": (["Txt_TokenEmbed"] +
                 [f"Txt_Block_{i}" for i in range(1, 13)] +
                 ["text_embedding_final"]),
    },
    "radfm": {
        "vision": (["Vis_PatchEmbed"] +
                   [f"Vis_Block_{i}" for i in range(1, 13)] +
                   ["Vis_Perceiver", "Vis_Projected"]),
        "text": (["Txt_Embed"] +
                 [f"Txt_Block_{i}" for i in range(1, 41)] +
                 ["Txt_FinalNorm"]),
    },
    "llavamed1p5": {
        "vision": (["Vis_CLIP_Embed"] +
                   [f"Vis_CLIP_Layer_{i}" for i in range(1, 25)] +
                   ["Vis_MM_Projector"]),
        "text": (["Txt_Mistral_Embed"] +
                 [f"Txt_Mistral_Layer_{i}" for i in range(1, 33)]),
    },
    "nv_reason": {
        "vision": [f"Vis_Block_{i}" for i in range(32)],
        "text": (["Txt_Embed"] +
                 [f"Txt_Block_{i}" for i in range(36)] +
                 ["Txt_FinalNorm"]),
    },
    "chexagent": {
        "vision": ([f"Vis_SigLIP_Block_{i}" for i in range(1, 25)] +
                   ["image_embedding_final"]),
        "text": (["Txt_TokenEmbed"] +
                 [f"Txt_PhiBlock_{i}" for i in range(1, 33)] +
                 ["text_embedding_final"]),
    },
    "medversa": {
        "vision": (["Vis_Embed"] +
                   [f"Vis_S0_B{i}" for i in range(1, 3)] +
                   [f"Vis_S1_B{i}" for i in range(1, 3)] +
                   [f"Vis_S2_B{i}" for i in range(1, 19)] +
                   [f"Vis_S3_B{i}" for i in range(1, 3)] +
                   ["Vis_FinalNorm", "Vis_LNVision", "Vis_Projected"]),
        "text": (["Txt_Embed"] +
                 [f"Txt_Block_{i}" for i in range(1, 33)] +
                 ["Txt_FinalNorm"]),
    },
    "chexzero": {
        "vision": ([f"Vis_Block_{i}" for i in range(1, 13)] +
                   ["image_embedding_final"]),
        "text": (["Txt_TokenEmbed"] +
                 [f"Txt_Block_{i}" for i in range(1, 13)] +
                 ["text_embedding_final"]),
    },
}

ATTRIBUTES = ["sex", "age", "ethnicity"]
DATASET = "chexpert"


def get_features_root(model_name):
    return os.path.join(OTHER_MODELS_ROOT, model_name, "probe_experiment_outputs", DATASET)


def train_demographic_probe(train_dir, val_dir, models_out_dir, results_out_dir,
                            layer_name, attribute):
    """Train a single demographic probe (binary or multiclass)."""
    log_prefix = f"[{layer_name} | {attribute}]"

    model_path = os.path.join(models_out_dir, f"probe_{layer_name}_{attribute}.joblib")
    scaler_path = os.path.join(models_out_dir, f"scaler_{layer_name}_{attribute}.joblib")
    results_path = os.path.join(results_out_dir, f"gridsearch_{layer_name}_{attribute}.csv")

    if os.path.exists(model_path) and os.path.exists(results_path):
        logging.info(f"{log_prefix} Already exists. Skipping.")
        return True

    try:
        # Load labels
        y_train_df = pd.read_csv(os.path.join(train_dir, "labels_and_metadata.csv"))
        y_val_df = pd.read_csv(os.path.join(val_dir, "labels_and_metadata.csv"))

        y_train_series = y_train_df[attribute].dropna()
        y_train_cleaned = y_train_series.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0].dropna()
        y_train_final = y_train_cleaned.astype(float).astype(int)

        y_val_series = y_val_df[attribute].dropna()
        y_val_cleaned = y_val_series.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0].dropna()
        y_val_final = y_val_cleaned.astype(float).astype(int)

        # Load features
        emb_path = os.path.join(train_dir, f"{layer_name}_embeddings.npy")
        if not os.path.exists(emb_path):
            logging.warning(f"{log_prefix} Train embeddings not found: {emb_path}. Skipping.")
            return False
        x_train_raw = np.load(emb_path)
        x_val_raw = np.load(os.path.join(val_dir, f"{layer_name}_embeddings.npy"))

        x_train_aligned = x_train_raw[y_train_cleaned.index]
        x_val_aligned = x_val_raw[y_val_cleaned.index]
        del x_train_raw, x_val_raw
        gc.collect()

        # Scale
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_aligned).astype(np.float32)
        x_val_scaled = scaler.transform(x_val_aligned).astype(np.float32)
        del x_train_aligned, x_val_aligned
        gc.collect()

        # GridSearchCV
        x_for_gs = np.concatenate([x_train_scaled, x_val_scaled])
        y_for_gs = np.concatenate([y_train_final.values, y_val_final.values])
        del x_train_scaled, x_val_scaled
        gc.collect()

        split_indicator = np.array([-1] * len(y_train_final) + [0] * len(y_val_final))
        pds = PredefinedSplit(test_fold=split_indicator)

        n_classes = len(np.unique(y_for_gs))
        scoring = 'roc_auc' if n_classes == 2 else 'roc_auc_ovr'

        gs = GridSearchCV(
            LogisticRegression(max_iter=5000, solver='lbfgs', random_state=RANDOM_STATE,
                               multi_class='ovr' if n_classes > 2 else 'auto'),
            {'C': np.logspace(-4, -1, 4)},
            cv=pds, scoring=scoring, refit=True, n_jobs=1, verbose=0
        )
        gs.fit(x_for_gs, y_for_gs)

        # Save
        os.makedirs(models_out_dir, exist_ok=True)
        os.makedirs(results_out_dir, exist_ok=True)
        joblib.dump(gs.best_estimator_, model_path)
        joblib.dump(scaler, scaler_path)

        results = {
            'layer': layer_name, 'attribute': attribute,
            'best_C': gs.best_params_['C'], 'best_val_score': gs.best_score_,
            'n_train': len(y_train_final), 'n_val': len(y_val_final), 'n_classes': n_classes
        }
        pd.DataFrame([results]).to_csv(results_path, index=False)
        logging.info(f"{log_prefix} DONE. Best C={gs.best_params_['C']:.6f}, AUC={gs.best_score_:.4f}")

        del x_for_gs, y_for_gs, gs
        gc.collect()
        return True

    except Exception as e:
        logging.error(f"{log_prefix} FAILED: {e}")
        return False


def train_nofinding_probe(train_dir, val_dir, models_out_dir, results_out_dir, layer_name):
    """Train a binary No-Finding probe."""
    log_prefix = f"[{layer_name} | No Finding]"

    model_path = os.path.join(models_out_dir, f"probe_{layer_name}_No_Finding.joblib")
    scaler_path = os.path.join(models_out_dir, f"scaler_{layer_name}_No_Finding.joblib")
    results_path = os.path.join(results_out_dir, f"gridsearch_{layer_name}_No_Finding.csv")

    if os.path.exists(model_path) and os.path.exists(results_path):
        logging.info(f"{log_prefix} Already exists. Skipping.")
        return True

    try:
        y_train_df = pd.read_csv(os.path.join(train_dir, "labels_and_metadata.csv"))
        y_val_df = pd.read_csv(os.path.join(val_dir, "labels_and_metadata.csv"))

        # Extract No Finding column
        col_name = "No Finding"
        if col_name not in y_train_df.columns:
            logging.warning(f"{log_prefix} Column '{col_name}' not in metadata. Skipping.")
            return False

        y_train_series = y_train_df[col_name].dropna()
        y_train_cleaned = y_train_series.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0].dropna()
        y_train_final = y_train_cleaned.astype(float).astype(int)

        y_val_series = y_val_df[col_name].dropna()
        y_val_cleaned = y_val_series.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0].dropna()
        y_val_final = y_val_cleaned.astype(float).astype(int)

        # Load features
        emb_path = os.path.join(train_dir, f"{layer_name}_embeddings.npy")
        if not os.path.exists(emb_path):
            logging.warning(f"{log_prefix} Train embeddings not found: {emb_path}. Skipping.")
            return False
        x_train_raw = np.load(emb_path)
        x_val_raw = np.load(os.path.join(val_dir, f"{layer_name}_embeddings.npy"))

        x_train_aligned = x_train_raw[y_train_cleaned.index]
        x_val_aligned = x_val_raw[y_val_cleaned.index]
        del x_train_raw, x_val_raw
        gc.collect()

        # Scale
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_aligned).astype(np.float32)
        x_val_scaled = scaler.transform(x_val_aligned).astype(np.float32)
        del x_train_aligned, x_val_aligned
        gc.collect()

        # GridSearchCV
        x_for_gs = np.concatenate([x_train_scaled, x_val_scaled])
        y_for_gs = np.concatenate([y_train_final.values, y_val_final.values])
        del x_train_scaled, x_val_scaled
        gc.collect()

        split_indicator = np.array([-1] * len(y_train_final) + [0] * len(y_val_final))
        pds = PredefinedSplit(test_fold=split_indicator)

        gs = GridSearchCV(
            LogisticRegression(max_iter=5000, solver='lbfgs', random_state=RANDOM_STATE),
            {'C': np.logspace(-4, -1, 4)},
            cv=pds, scoring='roc_auc', refit=True, n_jobs=1, verbose=0
        )
        gs.fit(x_for_gs, y_for_gs)

        # Save
        os.makedirs(models_out_dir, exist_ok=True)
        os.makedirs(results_out_dir, exist_ok=True)
        joblib.dump(gs.best_estimator_, model_path)
        joblib.dump(scaler, scaler_path)

        results = {
            'layer': layer_name, 'attribute': 'No_Finding',
            'best_C': gs.best_params_['C'], 'best_val_score': gs.best_score_,
            'n_train': len(y_train_final), 'n_val': len(y_val_final), 'n_classes': 2
        }
        pd.DataFrame([results]).to_csv(results_path, index=False)
        logging.info(f"{log_prefix} DONE. Best C={gs.best_params_['C']:.6f}, AUC={gs.best_score_:.4f}")

        del x_for_gs, y_for_gs, gs
        gc.collect()
        return True

    except Exception as e:
        logging.error(f"{log_prefix} FAILED: {e}")
        return False


def train_model_probes(model_name, modalities=("vision", "text")):
    """Train all probes (demographic + No Finding) for one model."""
    features_root = get_features_root(model_name)
    layers_dict = MODEL_LAYERS[model_name]

    logging.info(f"\n{'='*70}")
    logging.info(f"  MODEL: {model_name} | Dataset: {DATASET}")
    logging.info(f"{'='*70}")

    total_trained = 0
    total_skipped = 0
    total_failed = 0

    for modality in modalities:
        if modality not in layers_dict:
            logging.warning(f"  [{model_name}] No {modality} layers defined. Skipping.")
            continue

        layers = layers_dict[modality]
        train_dir = os.path.join(features_root, f"features_{modality}_only_train")
        val_dir = os.path.join(features_root, f"features_{modality}_only_val")

        if not os.path.isdir(train_dir):
            logging.warning(f"  [{model_name}/{modality}] Train dir not found: {train_dir}. Skipping.")
            continue
        if not os.path.isdir(val_dir):
            logging.warning(f"  [{model_name}/{modality}] Val dir not found: {val_dir}. Skipping.")
            continue

        models_out_dir = os.path.join(features_root, f"trained_probes_{modality}_only")
        results_out_dir = os.path.join(features_root, f"results_{modality}_only_gridsearch")
        os.makedirs(models_out_dir, exist_ok=True)
        os.makedirs(results_out_dir, exist_ok=True)

        logging.info(f"\n  --- {model_name} / {modality} ({len(layers)} layers) ---")

        for layer in layers:
            # Demographics
            for attr in ATTRIBUTES:
                ok = train_demographic_probe(
                    train_dir, val_dir, models_out_dir, results_out_dir, layer, attr
                )
                if ok:
                    total_trained += 1
                else:
                    total_failed += 1
                gc.collect()

            # No Finding
            ok = train_nofinding_probe(
                train_dir, val_dir, models_out_dir, results_out_dir, layer
            )
            if ok:
                total_trained += 1
            else:
                total_failed += 1
            gc.collect()

    logging.info(f"\n  [{model_name}] Summary: trained={total_trained}, failed={total_failed}")
    return total_failed == 0


def train_single_layer(model_name, modality, layer_name):
    """Train all 4 probes (sex, age, ethnicity, No_Finding) for a single layer.
    
    Intended to be called as a standalone process so memory is fully freed on exit.
    """
    features_root = get_features_root(model_name)
    train_dir = os.path.join(features_root, f"features_{modality}_only_train")
    val_dir = os.path.join(features_root, f"features_{modality}_only_val")
    models_out_dir = os.path.join(features_root, f"trained_probes_{modality}_only")
    results_out_dir = os.path.join(features_root, f"results_{modality}_only_gridsearch")

    if not os.path.isdir(train_dir):
        logging.error(f"Train dir not found: {train_dir}")
        return False
    if not os.path.isdir(val_dir):
        logging.error(f"Val dir not found: {val_dir}")
        return False

    os.makedirs(models_out_dir, exist_ok=True)
    os.makedirs(results_out_dir, exist_ok=True)

    success = True
    for attr in ATTRIBUTES:
        ok = train_demographic_probe(train_dir, val_dir, models_out_dir, results_out_dir,
                                     layer_name, attr)
        if not ok:
            success = False
        gc.collect()

    ok = train_nofinding_probe(train_dir, val_dir, models_out_dir, results_out_dir, layer_name)
    if not ok:
        success = False
    gc.collect()

    return success


def main():
    parser = argparse.ArgumentParser(
        description="Train demographic + No Finding probes on CheXpert features for all models."
    )
    parser.add_argument('--models', nargs='+', default=list(MODEL_LAYERS.keys()),
                        choices=list(MODEL_LAYERS.keys()),
                        help="Models to train probes for (default: all 8)")
    parser.add_argument('--modalities', nargs='+', default=['vision', 'text'],
                        choices=['vision', 'text'],
                        help="Modalities to train (default: both)")
    # Single-layer mode: for shell-level parallelism
    parser.add_argument('--single_layer', type=str, default=None,
                        help="Train probes for a single layer only (for parallel dispatch)")
    parser.add_argument('--single_model', type=str, default=None,
                        choices=list(MODEL_LAYERS.keys()),
                        help="Model name (required with --single_layer)")
    parser.add_argument('--single_modality', type=str, default=None,
                        choices=['vision', 'text'],
                        help="Modality (required with --single_layer)")
    args = parser.parse_args()

    # --- Single-layer mode (used by shell script for parallel dispatch) ---
    if args.single_layer:
        if not args.single_model or not args.single_modality:
            parser.error("--single_layer requires --single_model and --single_modality")
        logging.info(f"[Single-layer] {args.single_model}/{args.single_modality}/{args.single_layer}")
        ok = train_single_layer(args.single_model, args.single_modality, args.single_layer)
        sys.exit(0 if ok else 1)

    # --- Full mode (sequential, all layers) ---
    logging.info("=" * 70)
    logging.info("  CHEXPERT PROBE TRAINING - ALL MODELS")
    logging.info(f"  Models: {args.models}")
    logging.info(f"  Modalities: {args.modalities}")
    logging.info(f"  Probes: sex, age, ethnicity, No_Finding")
    logging.info(f"  n_jobs=1 (memory-safe)")
    logging.info("=" * 70)

    results = {}
    for model_name in args.models:
        success = train_model_probes(model_name, modalities=tuple(args.modalities))
        results[model_name] = "OK" if success else "PARTIAL FAILURE"

    logging.info("\n" + "=" * 70)
    logging.info("  FINAL SUMMARY")
    logging.info("=" * 70)
    for m, status in results.items():
        logging.info(f"  {m:20s} -> {status}")


if __name__ == "__main__":
    main()
