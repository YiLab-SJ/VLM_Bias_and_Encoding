# universal_train_disease_probe.py
# Universal script to train multi-label disease probes for any model.

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
import pandas as pd
import os
import argparse
import logging
import warnings
import joblib
import gc

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

RANDOM_STATE = 42
DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
    "Fracture", "Lung Lesion", "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"
]

def train_universal_disease_probe():
    parser = argparse.ArgumentParser(description="Train a universal disease probe.")
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--models_out_dir', type=str, required=True)
    parser.add_argument('--results_out_dir', type=str, required=True)
    parser.add_argument('--layer_name', type=str, required=True)
    args = parser.parse_args()

    layer_name = args.layer_name
    log_prefix = f"[Disease | {layer_name}]"

    os.makedirs(args.models_out_dir, exist_ok=True)
    os.makedirs(args.results_out_dir, exist_ok=True)

    # Skip logic
    model_path = os.path.join(args.models_out_dir, f"disease_probe_{layer_name}.joblib")
    scaler_path = os.path.join(args.models_out_dir, f"disease_scaler_{layer_name}.joblib")
    results_path = os.path.join(args.results_out_dir, f"disease_gridsearch_{layer_name}.csv")

    if os.path.exists(model_path) and os.path.exists(results_path):
        logging.info(f"{log_prefix} Model/results exist. Skipping.")
        return
        
    try:
        logging.info(f"{log_prefix} Loading labels...")
        y_train_df = pd.read_csv(os.path.join(args.train_dir, "labels_and_metadata.csv"))
        y_val_df = pd.read_csv(os.path.join(args.val_dir, "labels_and_metadata.csv"))

        # Clean Labels
        y_train = y_train_df[DISEASE_LABELS].apply(
            lambda col: col.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0]
        ).astype(float).astype(int)

        y_val = y_val_df[DISEASE_LABELS].apply(
            lambda col: col.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0]
        ).astype(float).astype(int)

        # Scale Data sequentially to save RAM
        logging.info(f"{log_prefix} Loading and scaling Train arrays...")
        scaler = StandardScaler()
        x_train_raw = np.load(os.path.join(args.train_dir, f"{layer_name}_embeddings.npy"))
        x_train_scaled = scaler.fit_transform(x_train_raw).astype(np.float32)
        
        # AGGRESSIVE MEMORY CLEANUP
        del x_train_raw
        gc.collect()

        logging.info(f"{log_prefix} Loading and scaling Val arrays...")
        x_val_raw = np.load(os.path.join(args.val_dir, f"{layer_name}_embeddings.npy"))
        x_val_scaled = scaler.transform(x_val_raw).astype(np.float32)
        
        # AGGRESSIVE MEMORY CLEANUP
        del x_val_raw
        gc.collect()

        logging.info(f"{log_prefix} Concatenating arrays...")
        x_for_gs = np.concatenate([x_train_scaled, x_val_scaled])
        y_for_gs = pd.concat([y_train, y_val]).values

        # AGGRESSIVE MEMORY CLEANUP
        del x_train_scaled
        del x_val_scaled
        gc.collect()

        # GridSearchCV setup
        split_indicator = np.array([-1] * len(y_train) + [0] * len(y_val))
        pds = PredefinedSplit(test_fold=split_indicator)
        param_grid = {'estimator__C': np.logspace(-4, -1, 4)}

        base_lr = LogisticRegression(max_iter=5000, solver='lbfgs', random_state=RANDOM_STATE)
        ovr_model = OneVsRestClassifier(base_lr)

        logging.info(f"{log_prefix} Starting GridSearchCV (n_jobs=2)...")
        # n_jobs=2 utilizes 2 cores per process
        gs = GridSearchCV(
            ovr_model, param_grid, cv=pds,
            scoring='roc_auc', refit=True, n_jobs=1, verbose=0
        )
        gs.fit(x_for_gs, y_for_gs)

        # Save
        logging.info(f"{log_prefix} Saving models...")
        joblib.dump(gs.best_estimator_, model_path)
        joblib.dump(scaler, scaler_path)

        results = {
            'layer': layer_name, 'best_C': gs.best_params_['estimator__C'],
            'best_val_score': gs.best_score_, 'n_train': len(y_train), 'n_val': len(y_val)
        }
        pd.DataFrame([results]).to_csv(results_path, index=False)
        logging.info(f"{log_prefix} DONE. Best C={gs.best_params_['estimator__C']:.6f}")

    except Exception as e:
        logging.error(f"{log_prefix} FAILED: {e}", exc_info=True)

if __name__ == "__main__":
    train_universal_disease_probe()