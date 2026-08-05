# universal_train_demographic_probe.py
# Universal script to train a demographic probe (binary or multiclass) 
# for any model, vision or text.

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit
import numpy as np
import pandas as pd
import os
import argparse
import logging
import warnings
import joblib
import gc

# Set up logging to flush immediately (belt and suspenders with python -u)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
RANDOM_STATE = 42

def train_universal_demographic_probe():
    parser = argparse.ArgumentParser(description="Train a universal demographic probe.")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to training features directory")
    parser.add_argument('--val_dir', type=str, required=True, help="Path to validation features directory")
    parser.add_argument('--models_out_dir', type=str, required=True, help="Directory to save trained models")
    parser.add_argument('--results_out_dir', type=str, required=True, help="Directory to save CSV results")
    parser.add_argument('--layer_name', type=str, required=True, help="Name of the layer (e.g., Vis_Block_1)")
    parser.add_argument('--attribute', type=str, required=True, help="Attribute to predict (sex, age, ethnicity)")
    args = parser.parse_args()

    layer_name = args.layer_name
    attribute = args.attribute
    log_prefix = f"[{layer_name} | {attribute}]"

    os.makedirs(args.models_out_dir, exist_ok=True)
    os.makedirs(args.results_out_dir, exist_ok=True)

    # Skip logic
    model_path = os.path.join(args.models_out_dir, f"probe_{layer_name}_{attribute}.joblib")
    scaler_path = os.path.join(args.models_out_dir, f"scaler_{layer_name}_{attribute}.joblib")
    results_path = os.path.join(args.results_out_dir, f"gridsearch_{layer_name}_{attribute}.csv")
    
    if os.path.exists(model_path) and os.path.exists(results_path):
        logging.info(f"{log_prefix} Model/results exist. Skipping.")
        return

    try:
        logging.info(f"{log_prefix} Loading labels...")
        y_train_df = pd.read_csv(os.path.join(args.train_dir, "labels_and_metadata.csv"))
        y_val_df = pd.read_csv(os.path.join(args.val_dir, "labels_and_metadata.csv"))

        # Clean Labels first to get indices
        y_train_series = y_train_df[attribute].dropna()
        y_train_cleaned = y_train_series.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0].dropna()
        y_train_final = y_train_cleaned.astype(float).astype(int)

        y_val_series = y_val_df[attribute].dropna()
        y_val_cleaned = y_val_series.astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0].dropna()
        y_val_final = y_val_cleaned.astype(float).astype(int)

        logging.info(f"{log_prefix} Loading arrays into RAM...")
        x_train_raw = np.load(os.path.join(args.train_dir, f"{layer_name}_embeddings.npy"))
        x_val_raw = np.load(os.path.join(args.val_dir, f"{layer_name}_embeddings.npy"))

        logging.info(f"{log_prefix} Aligning arrays to valid labels...")
        x_train_aligned = x_train_raw[y_train_cleaned.index]
        x_val_aligned = x_val_raw[y_val_cleaned.index]

        # AGGRESSIVE MEMORY CLEANUP
        logging.info(f"{log_prefix} Freeing raw arrays from RAM...")
        del x_train_raw
        del x_val_raw
        gc.collect()

        # Scale Data (force back to float32 to save 50% RAM compared to float64)
        logging.info(f"{log_prefix} Scaling data...")
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train_aligned).astype(np.float32)
        x_val_scaled = scaler.transform(x_val_aligned).astype(np.float32)

        # AGGRESSIVE MEMORY CLEANUP
        del x_train_aligned
        del x_val_aligned
        gc.collect()

        # GridSearchCV Setup
        x_for_gs = np.concatenate([x_train_scaled, x_val_scaled])
        y_for_gs = np.concatenate([y_train_final, y_val_final])
        
        # AGGRESSIVE MEMORY CLEANUP
        del x_train_scaled
        del x_val_scaled
        gc.collect()

        split_indicator = np.array([-1] * len(y_train_final) + [0] * len(y_val_final))
        pds = PredefinedSplit(test_fold=split_indicator)

        param_grid = {'C': np.logspace(-4, -1, 4)}
        n_classes = len(np.unique(y_for_gs))
        scoring = 'roc_auc' if n_classes == 2 else 'roc_auc_ovr'

        logging.info(f"{log_prefix} Starting GridSearchCV (n_jobs=2)...")
        # n_jobs=2 here uses 2 cores per job.
        gs = GridSearchCV(
            LogisticRegression(max_iter=5000, solver='lbfgs', random_state=RANDOM_STATE,
                               multi_class='ovr' if n_classes > 2 else 'auto'),
            param_grid, cv=pds, scoring=scoring, refit=True, n_jobs=1, verbose=0
        )
        gs.fit(x_for_gs, y_for_gs)

        # Save
        logging.info(f"{log_prefix} Saving models...")
        joblib.dump(gs.best_estimator_, model_path)
        joblib.dump(scaler, scaler_path)

        results = {
            'layer': layer_name, 'attribute': attribute,
            'best_C': gs.best_params_['C'], 'best_val_score': gs.best_score_,
            'n_train': len(y_train_final), 'n_val': len(y_val_final), 'n_classes': n_classes
        }
        pd.DataFrame([results]).to_csv(results_path, index=False)
        logging.info(f"{log_prefix} DONE. Best C={gs.best_params_['C']:.6f}")

    except Exception as e:
        logging.error(f"{log_prefix} FAILED: {e}", exc_info=True)

if __name__ == "__main__":
    train_universal_demographic_probe()