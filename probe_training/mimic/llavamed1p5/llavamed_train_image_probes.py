# llavamed_train_image_probes.py
# Trains a single vision-layer demographic probe for LLaVA-Med 1.5.
# This is the LLaVA-Med 1.5 equivalent of script5_train_image_probes.py.
#
# Usage:
#   python llavamed_train_image_probes.py --dataset_folder_name MIMIC-CXR-JPG \
#       --layer_name Vis_CLIP_Layer_22 --attribute sex

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np
import pandas as pd
import os
import sys
import argparse
import logging
import warnings
import joblib

# --- Config import ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from config_llavamed import (
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    VISION_LAYER_DIMS
)

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def train_single_vision_probe():
    parser = argparse.ArgumentParser(description="Train a single LLaVA-Med 1.5 vision-layer probe.")
    parser.add_argument('--dataset_folder_name', type=str, default='MIMIC-CXR-JPG',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--layer_name', type=str, required=True,
                        help="Vision layer to probe (e.g., Vis_CLIP_Layer_22).")
    parser.add_argument('--attribute', type=str, required=True,
                        help="Attribute to predict (e.g., sex, age, ethnicity).")
    args = parser.parse_args()

    layer_name = args.layer_name
    attribute = args.attribute
    dataset_folder_name = args.dataset_folder_name

    log_prefix = f"[{layer_name} | {attribute}]"
    logging.info(f"{log_prefix} Starting LLaVA-Med 1.5 vision probe training.")

    # --- Setup Paths ---
    features_root_dir = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, dataset_folder_name)
    train_features_dir = os.path.join(features_root_dir, "features_vision_only_train")
    val_features_dir = os.path.join(features_root_dir, "features_vision_only_val")

    models_dir = os.path.join(features_root_dir, "trained_probes_vision_only")
    results_dir = os.path.join(features_root_dir, "results_vision_only_gridsearch")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

# --- SKIP LOGIC ---
    model_path = os.path.join(models_dir, f"probe_{layer_name}_{attribute}.joblib")
    scaler_path = os.path.join(models_dir, f"scaler_{layer_name}_{attribute}.joblib")
    results_path = os.path.join(results_dir, f"gridsearch_{layer_name}_{attribute}.csv")
    
    if os.path.exists(model_path) and os.path.exists(results_path):
        logging.info(f"{log_prefix} Model and results already exist. Skipping training to save compute.")
        return

    try:
        # --- Load Data ---
        logging.info(f"{log_prefix} Loading data...")
        y_train_df = pd.read_csv(os.path.join(train_features_dir, "labels_and_metadata.csv"))
        y_val_df = pd.read_csv(os.path.join(val_features_dir, "labels_and_metadata.csv"))

        x_train_raw = np.load(os.path.join(train_features_dir, f"{layer_name}_embeddings.npy"), mmap_mode='r')
        x_val_raw = np.load(os.path.join(val_features_dir, f"{layer_name}_embeddings.npy"), mmap_mode='r')

        # --- Clean Labels ---
        y_train_series = y_train_df[attribute].dropna()
        y_train_cleaned = y_train_series.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0].dropna()
        x_train_aligned = x_train_raw[y_train_cleaned.index]
        y_train_final = y_train_cleaned.astype(float).astype(int)

        y_val_series = y_val_df[attribute].dropna()
        y_val_cleaned = y_val_series.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0].dropna()
        x_val_aligned = x_val_raw[y_val_cleaned.index]
        y_val_final = y_val_cleaned.astype(float).astype(int)

        # --- Scale Data ---
        logging.info(f"{log_prefix} Scaling. Train: {len(y_train_final)}, Val: {len(y_val_final)}")
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_aligned)
        x_val_scaled = scaler.transform(x_val_aligned)

        # --- GridSearchCV with PredefinedSplit ---
        x_for_gs = np.concatenate([x_train_scaled, x_val_scaled])
        y_for_gs = np.concatenate([y_train_final, y_val_final])
        split_indicator = np.array([-1] * len(x_train_scaled) + [0] * len(x_val_scaled))
        pds = PredefinedSplit(test_fold=split_indicator)

        param_grid = {'C': np.logspace(-4, -1, 4)}
        n_classes = len(np.unique(y_for_gs))
        scoring = 'roc_auc' if n_classes == 2 else 'roc_auc_ovr'

        logging.info(f"{log_prefix} GridSearchCV: {n_classes} classes, scoring={scoring}")
        gs = GridSearchCV(
            LogisticRegression(max_iter=5000, solver='lbfgs', random_state=RANDOM_STATE,
                               multi_class='ovr' if n_classes > 2 else 'auto'),
            param_grid, cv=pds, scoring=scoring, refit=True, n_jobs=1, verbose=0
        )
        gs.fit(x_for_gs, y_for_gs)

        logging.info(f"{log_prefix} Best C={gs.best_params_['C']:.6f}, Best score={gs.best_score_:.4f}")

        # --- Save ---
        model_path = os.path.join(models_dir, f"probe_{layer_name}_{attribute}.joblib")
        scaler_path = os.path.join(models_dir, f"scaler_{layer_name}_{attribute}.joblib")
        joblib.dump(gs.best_estimator_, model_path)
        joblib.dump(scaler, scaler_path)

        results = {
            'layer': layer_name, 'attribute': attribute,
            'best_C': gs.best_params_['C'], 'best_val_score': gs.best_score_,
            'n_train': len(y_train_final), 'n_val': len(y_val_final),
            'n_classes': n_classes
        }
        results_path = os.path.join(results_dir, f"gridsearch_{layer_name}_{attribute}.csv")
        pd.DataFrame([results]).to_csv(results_path, index=False)
        logging.info(f"{log_prefix} Saved model, scaler, and results.")

    except Exception as e:
        logging.error(f"{log_prefix} FAILED: {e}", exc_info=True)


if __name__ == "__main__":
    train_single_vision_probe()
