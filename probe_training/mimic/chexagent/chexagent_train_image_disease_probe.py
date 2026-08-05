# chexagent_train_image_disease_probe.py
# Trains multi-label disease probes on CheXagent vision layer embeddings.
# This is the CheXagent equivalent of biovilt_train_image_disease_probe.py.
#
# Usage:
#   python chexagent_train_image_disease_probe.py --dataset_folder_name MIMIC-CXR-JPG \
#       --layer_name Vis_SigLIP_Block_24

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.multiclass import OneVsRestClassifier
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
from config_chexagent import (
    DATASET_CONFIGS, PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, RANDOM_STATE,
    VISION_LAYER_DIMS, DISEASE_LABELS
)

# --- Global Settings ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def train_vision_disease_probes():
    parser = argparse.ArgumentParser(description="Train CheXagent vision disease probes for a single layer.")
    parser.add_argument('--dataset_folder_name', type=str, default='MIMIC-CXR-JPG',
                        choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--layer_name', type=str, required=True,
                        help="Vision layer to probe.")
    args = parser.parse_args()

    layer_name = args.layer_name
    dataset_folder_name = args.dataset_folder_name

    log_prefix = f"[Vision Disease | {layer_name}]"
    logging.info(f"{log_prefix} Starting disease probe training.")

    # --- Setup Paths ---
    features_root_dir = os.path.join(PROBE_EXPERIMENT_OUTPUT_ROOT_DIR, dataset_folder_name)
    train_features_dir = os.path.join(features_root_dir, "features_vision_only_train")
    val_features_dir = os.path.join(features_root_dir, "features_vision_only_val")

    models_dir = os.path.join(features_root_dir, "trained_probes_image_diseases")
    results_dir = os.path.join(features_root_dir, "results_vision_only_disease_gridsearch")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    try:
        # --- Load Data ---
        logging.info(f"{log_prefix} Loading data...")
        y_train_df = pd.read_csv(os.path.join(train_features_dir, "labels_and_metadata.csv"))
        y_val_df = pd.read_csv(os.path.join(val_features_dir, "labels_and_metadata.csv"))

        x_train_raw = np.load(os.path.join(train_features_dir, f"{layer_name}_embeddings.npy"), mmap_mode='r')
        x_val_raw = np.load(os.path.join(val_features_dir, f"{layer_name}_embeddings.npy"), mmap_mode='r')

        # --- Clean Labels ---
        logging.info(f"{log_prefix} Cleaning disease labels...")
        y_train = y_train_df[DISEASE_LABELS].apply(
            lambda col: col.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0]
        ).astype(float).astype(int)

        y_val = y_val_df[DISEASE_LABELS].apply(
            lambda col: col.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0]
        ).astype(float).astype(int)

        # --- Scale Data ---
        logging.info(f"{log_prefix} Scaling data...")
        scaler = StandardScaler()
        scaler.fit(x_train_raw)

        x_for_gs = np.concatenate([scaler.transform(x_train_raw), scaler.transform(x_val_raw)])
        y_for_gs = pd.concat([y_train, y_val]).values

        # --- GridSearchCV with PredefinedSplit ---
        split_indicator = np.array([-1] * len(x_train_raw) + [0] * len(x_val_raw))
        pds = PredefinedSplit(test_fold=split_indicator)

        param_grid = {'estimator__C': np.logspace(-4, -1, 4)}

        logging.info(f"{log_prefix} Starting GridSearchCV for multi-label disease classification...")
        base_lr = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=RANDOM_STATE)
        ovr_model = OneVsRestClassifier(base_lr)

        gs = GridSearchCV(
            ovr_model, param_grid, cv=pds,
            scoring='roc_auc', refit=True, n_jobs=2, verbose=0
        )
        gs.fit(x_for_gs, y_for_gs)

        logging.info(f"{log_prefix} Best C={gs.best_params_['estimator__C']:.6f}, Score={gs.best_score_:.4f}")

        # --- Save ---
        model_path = os.path.join(models_dir, f"disease_probe_{layer_name}.joblib")
        scaler_path = os.path.join(models_dir, f"disease_scaler_{layer_name}.joblib")
        joblib.dump(gs.best_estimator_, model_path)
        joblib.dump(scaler, scaler_path)

        results = {
            'layer': layer_name,
            'best_C': gs.best_params_['estimator__C'],
            'best_val_score': gs.best_score_,
            'n_train': len(y_train), 'n_val': len(y_val)
        }
        results_path = os.path.join(results_dir, f"disease_gridsearch_{layer_name}.csv")
        pd.DataFrame([results]).to_csv(results_path, index=False)
        logging.info(f"{log_prefix} Saved model, scaler, and results.")

    except Exception as e:
        logging.error(f"{log_prefix} FAILED: {e}", exc_info=True)


if __name__ == "__main__":
    train_vision_disease_probes()
