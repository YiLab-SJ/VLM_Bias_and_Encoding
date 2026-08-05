#!/usr/bin/env python
"""
train_rexgradient_nonpediatric_all_layers.py

Train sex, age, and No Finding probes on ALL LAYERS for BOTH vision and text,
using NON-PEDIATRIC (age class >= 1, i.e. 18+) samples only.

This OVERWRITES any existing probes in trained_probes_{modality}_only_nonpediatric/.
Age probes: 4-class (1=18-40, 2=40-60, 3=60-80, 4=80+).
Sex probes: 2-class (0=F, 1=M).
No Finding: 2-class (0=abnormal, 1=normal).

Usage:
    python train_rexgradient_nonpediatric_all_layers.py
    python train_rexgradient_nonpediatric_all_layers.py --models chexzero biovilt
    python train_rexgradient_nonpediatric_all_layers.py --models chexzero --n_jobs 5
"""

import os
import sys
import argparse
import logging
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

OTHER_MODELS_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models"
DATASET = "rexgradient"

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

TARGET_MODELS = list(MODEL_LAYERS.keys())


def train_probe(x_train, y_train, x_val, y_val, n_jobs_cv=5):
    """Train a logistic regression probe with grid search on C."""
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train).astype(np.float32)
    x_val_s = scaler.transform(x_val).astype(np.float32)

    x_all = np.concatenate([x_train_s, x_val_s])
    y_all = np.concatenate([y_train, y_val])

    split_indicator = np.array([-1] * len(y_train) + [0] * len(y_val))
    pds = PredefinedSplit(test_fold=split_indicator)

    n_classes = len(np.unique(y_all))
    scoring = 'roc_auc' if n_classes == 2 else 'roc_auc_ovr'

    gs = GridSearchCV(
        LogisticRegression(max_iter=5000, solver='lbfgs', random_state=RANDOM_STATE,
                           multi_class='ovr' if n_classes > 2 else 'auto'),
        {'C': np.logspace(-4, -1, 5)},
        cv=pds, scoring=scoring, refit=True, n_jobs=n_jobs_cv, verbose=0
    )
    gs.fit(x_all, y_all)
    return gs.best_estimator_, scaler, gs.best_params_['C'], gs.best_score_


def _train_one_layer(layer, train_dir, val_dir, out_dir,
                     adult_train_idx, adult_val_idx,
                     meta_train_adults, meta_val_adults, n_jobs_cv):
    """Train sex + age + No Finding probes for a single layer. Called in parallel."""
    # Skip if all 3 probes already exist
    sex_exists = os.path.exists(os.path.join(out_dir, f"probe_{layer}_sex.joblib"))
    age_exists = os.path.exists(os.path.join(out_dir, f"probe_{layer}_age.joblib"))
    nf_exists = os.path.exists(os.path.join(out_dir, f"probe_{layer}_No_Finding.joblib"))
    if sex_exists and age_exists and nf_exists:
        return f"    [{layer}] all probes exist, skipping"

    # Load embeddings
    try:
        x_train_full = np.load(os.path.join(train_dir, f"{layer}_embeddings.npy"))
        x_val_full = np.load(os.path.join(val_dir, f"{layer}_embeddings.npy"))
    except FileNotFoundError:
        return f"    [{layer}] missing embeddings, skipping"

    x_train = x_train_full[adult_train_idx]
    x_val = x_val_full[adult_val_idx]
    results = []

    # ── SEX ──
    if not sex_exists:
        y_tr_sex = meta_train_adults['sex'].dropna().astype(int)
        y_va_sex = meta_val_adults['sex'].dropna().astype(int)
        if len(y_tr_sex) > 50 and len(y_va_sex) > 50:
            probe, scaler, best_c, best_auc = train_probe(
                x_train[y_tr_sex.index], y_tr_sex.values,
                x_val[y_va_sex.index], y_va_sex.values, n_jobs_cv)
            joblib.dump(probe, os.path.join(out_dir, f"probe_{layer}_sex.joblib"))
            joblib.dump(scaler, os.path.join(out_dir, f"scaler_{layer}_sex.joblib"))
            results.append(f"sex C={best_c:.4f} AUC={best_auc:.4f}")

    # ── AGE (4-class: 1,2,3,4) ──
    if not age_exists:
        y_tr_age = meta_train_adults['age'].dropna().astype(int)
        y_va_age = meta_val_adults['age'].dropna().astype(int)
        y_tr_age_filt = y_tr_age[y_tr_age.isin([1, 2, 3, 4])]
        y_va_age_filt = y_va_age[y_va_age.isin([1, 2, 3, 4])]
        if len(y_tr_age_filt) > 50 and len(y_va_age_filt) > 50:
            probe, scaler, best_c, best_auc = train_probe(
                x_train[y_tr_age_filt.index], y_tr_age_filt.values,
                x_val[y_va_age_filt.index], y_va_age_filt.values, n_jobs_cv)
            joblib.dump(probe, os.path.join(out_dir, f"probe_{layer}_age.joblib"))
            joblib.dump(scaler, os.path.join(out_dir, f"scaler_{layer}_age.joblib"))
            results.append(f"age C={best_c:.4f} AUC={best_auc:.4f}")

    # ── NO FINDING ──
    if not nf_exists and 'No Finding' in meta_train_adults.columns:
        y_tr_nf = meta_train_adults['No Finding'].dropna().astype(int)
        y_va_nf = meta_val_adults['No Finding'].dropna().astype(int)
        if len(y_tr_nf) > 50 and len(y_va_nf) > 50 and len(y_tr_nf.unique()) >= 2:
            probe, scaler, best_c, best_auc = train_probe(
                x_train[y_tr_nf.index], y_tr_nf.values,
                x_val[y_va_nf.index], y_va_nf.values, n_jobs_cv)
            joblib.dump(probe, os.path.join(out_dir, f"probe_{layer}_No_Finding.joblib"))
            joblib.dump(scaler, os.path.join(out_dir, f"scaler_{layer}_No_Finding.joblib"))
            results.append(f"NF C={best_c:.4f} AUC={best_auc:.4f}")

    return f"    [{layer}] {', '.join(results)}" if results else f"    [{layer}] nothing to train"


def train_model(model, n_jobs_cv=5, n_parallel_layers=8, modalities=None):
    """Train all probes for all layers of one model (nonpediatric, adults only).
    Trains n_parallel_layers layers simultaneously, each with n_jobs_cv inside GridSearchCV."""
    if modalities is None:
        modalities = ['vision']
    exp_dir = os.path.join(OTHER_MODELS_ROOT, model, "probe_experiment_outputs", DATASET)

    for modality in modalities:
        layers = MODEL_LAYERS[model][modality]
        train_dir = os.path.join(exp_dir, f"features_{modality}_only_train")
        val_dir = os.path.join(exp_dir, f"features_{modality}_only_val")
        out_dir = os.path.join(exp_dir, f"trained_probes_{modality}_only_nonpediatric")
        os.makedirs(out_dir, exist_ok=True)

        # Load metadata once
        try:
            meta_train = pd.read_csv(os.path.join(train_dir, "labels_and_metadata.csv"))
            meta_val = pd.read_csv(os.path.join(val_dir, "labels_and_metadata.csv"))
        except FileNotFoundError as e:
            logging.warning(f"  [{model}] {modality}: missing metadata: {e}")
            continue

        for df in [meta_train, meta_val]:
            for col in ['sex', 'age']:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0], errors='coerce')
            if 'No Finding' in df.columns:
                df['No Finding'] = pd.to_numeric(
                    df['No Finding'].astype(str).str.extract(r'(-?\d+\.?\d*)').iloc[:, 0],
                    errors='coerce').fillna(0).astype(int)

        # Adult only (age class >= 1)
        adult_train_idx = np.where((meta_train['age'] >= 1).values)[0]
        adult_val_idx = np.where((meta_val['age'] >= 1).values)[0]
        meta_train_adults = meta_train.iloc[adult_train_idx].reset_index(drop=True)
        meta_val_adults = meta_val.iloc[adult_val_idx].reset_index(drop=True)

        logging.info(f"  [{model}] {modality}: {len(layers)} layers × {n_parallel_layers} parallel, "
                     f"{len(adult_train_idx)} train adults, {len(adult_val_idx)} val adults")

        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_parallel_layers, verbose=10)(
            delayed(_train_one_layer)(
                layer, train_dir, val_dir, out_dir,
                adult_train_idx, adult_val_idx,
                meta_train_adults, meta_val_adults, n_jobs_cv
            ) for layer in layers
        )

        for r in results:
            if r:
                logging.info(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=TARGET_MODELS)
    parser.add_argument('--modalities', nargs='+', default=['vision'],
                        choices=['vision', 'text'],
                        help='Modalities to train (default: vision only)')
    parser.add_argument('--n_jobs', type=int, default=5,
                        help='n_jobs for GridSearchCV (per-layer)')
    parser.add_argument('--n_parallel_layers', type=int, default=8,
                        help='Number of layers to train in parallel')
    args = parser.parse_args()

    logging.info("=" * 70)
    logging.info("  REXGRADIENT NONPEDIATRIC - TRAIN ALL-LAYER PROBES")
    logging.info(f"  Models: {args.models}")
    logging.info(f"  Modalities: {args.modalities}")
    logging.info(f"  Attributes: sex (2-class) + age (4-class) + No Finding")
    logging.info(f"  Adults only (age class >= 1)")
    logging.info(f"  Parallel: {args.n_parallel_layers} layers × n_jobs={args.n_jobs} per GridSearchCV")
    logging.info(f"  Output: trained_probes_{{mod}}_only_nonpediatric/")
    logging.info("=" * 70)

    for model in args.models:
        logging.info(f"\n{'='*50}")
        logging.info(f"  MODEL: {model}")
        logging.info(f"{'='*50}")
        train_model(model, n_jobs_cv=args.n_jobs,
                    n_parallel_layers=args.n_parallel_layers,
                    modalities=args.modalities)

    logging.info("\n" + "=" * 70)
    logging.info("  ALL MODELS COMPLETE")
    logging.info("=" * 70)


if __name__ == "__main__":
    main()
