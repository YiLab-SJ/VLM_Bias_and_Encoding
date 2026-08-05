#!/usr/bin/env python3
"""
run_lmm_all_datasets.py

Orchestrates the full LMM analysis pipeline for all 3 datasets:
  1. Builds CSVs from evaluation JSONs (build_lmm_csvs_from_jsons.py)
  2. Runs the LMM analysis (linear_mixed_model_analysis.py) with paths
     patched for each dataset.

This script monkey-patches the MODEL_REGISTRY and path constants in the
LMM analysis module before triggering its execution, so we don't need to
edit that complex R/Python hybrid script.

Usage:
    python run_lmm_all_datasets.py                # All 3 datasets
    python run_lmm_all_datasets.py --dataset mimic
    python run_lmm_all_datasets.py --dataset chexpert
    python run_lmm_all_datasets.py --dataset rexgradient
"""

import os
import sys
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = "/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models/all_models"
CSV_BASE = os.path.join(PROJECT_ROOT, "lmm_input_csvs")
LMM_SCRIPT = os.path.join(PROJECT_ROOT, "linear_mixed_model_analysis.py")


def run_csv_builder(dataset):
    """Run the CSV builder for a specific dataset."""
    builder = os.path.join(PROJECT_ROOT, "build_lmm_csvs_from_jsons.py")
    cmd = [sys.executable, builder, "--dataset", dataset]
    print(f"\n>>> Building CSVs for {dataset}...")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print(f"ERROR: CSV builder failed for {dataset}")
        return False
    return True


def run_lmm_for_dataset(dataset):
    """
    Build the LMM-ready dataset CSV for a given dataset.
    Self-contained: does not import linear_mixed_model_analysis.py (which needs rpy2/R).
    Produces 01_dataset.csv in lmm_outputs/{dataset}/ ready for LMM analysis.
    """
    import re
    import numpy as np
    import pandas as pd
    from pathlib import Path

    csv_dir = os.path.join(CSV_BASE, dataset)
    out_dir = os.path.join(PROJECT_ROOT, "lmm_outputs", dataset)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(csv_dir):
        print(f"ERROR: CSV directory not found: {csv_dir}")
        return False

    # Demographics pairs per dataset
    if dataset in ("rexgradient", "rexgradient_nonpediatric"):
        demo_pairs = {
            "sex": ("given_sex_0", "given_sex_1"),
            "age": ("given_age_1", "given_age_4"),
        }
    else:
        demo_pairs = {
            "sex": ("given_sex_0", "given_sex_1"),
            "age": ("given_age_0", "given_age_3"),
            "ethnicity": ("given_ethnicity_0", "given_ethnicity_1"),
        }

    # --- Layer numbering functions (same as linear_mixed_model_analysis.py) ---
    def _extract_int(pattern, text):
        m = re.search(pattern, str(text))
        return int(m.group(1)) if m else None

    def layer_chexzero(l):
        n = _extract_int(r"Vis_Block_(\d+)", l)
        if n is not None: return n
        if l == "image_embedding_final": return 13
        return None

    def layer_medgemma(l):
        if l == "Vis_Embed": return 0
        n = _extract_int(r"^Vis_Block_(\d+)$", l)
        if n is not None: return n
        if l == "Vis_PostNorm": return 28
        if l == "Vis_Projected": return 29
        return None

    def layer_radfm(l):
        if l == "Vis_PatchEmbed": return 0
        n = _extract_int(r"^Vis_Block_(\d+)$", l)
        if n is not None: return n
        if l == "Vis_Perceiver": return 13
        if l == "Vis_Projected": return 14
        return None

    def layer_nvreason(l):
        return _extract_int(r"^Vis_Block_(\d+)$", l)

    def layer_medversa(l):
        if l == "Vis_Embed": return 0
        if l == "Vis_S0_B1": return 1
        if l == "Vis_S0_B2": return 2
        if l == "Vis_S1_B1": return 3
        if l == "Vis_S1_B2": return 4
        n = _extract_int(r"^Vis_S2_B(\d+)$", l)
        if n is not None: return 4 + n
        if l == "Vis_S3_B1": return 23
        if l == "Vis_S3_B2": return 24
        if l == "Vis_FinalNorm": return 25
        if l == "Vis_LNVision": return 26
        if l == "Vis_Projected": return 27
        return None

    def layer_biovilt(l):
        mapping = {
            "Vis_ResNet_Layer1": 1, "Vis_ResNet_Layer2": 2,
            "Vis_ResNet_Layer3": 3, "Vis_ResNet_Layer4": 4,
            "Vis_BackboneToViT": 5, "img_embedding": 6,
            "image_embedding_final": 7,
        }
        return mapping.get(l)

    def layer_llavamed(l):
        if l == "Vis_CLIP_Embed": return 0
        n = _extract_int(r"^Vis_CLIP_Layer_(\d+)$", l)
        if n is not None: return n
        if l == "Vis_MM_Projector": return 25
        return None

    def layer_chexagent(l):
        n = _extract_int(r"^Vis_SigLIP_Block_(\d+)$", l)
        if n is not None: return n
        if l == "image_embedding_final": return 25
        return None

    # Model configurations: (display_name, csv_name, layer_fn, final_num)
    model_configs = [
        ('CheXZero', 'chexzero', layer_chexzero, 13),
        ('MedGemma', 'medgemma_1p5', layer_medgemma, 29),
        ('RadFM', 'radfm', layer_radfm, 14),
        ('NVReason', 'nv_reason', layer_nvreason, None),
        ('MedVersa', 'medversa', layer_medversa, 27),
        ('BioViLT', 'biovilt', layer_biovilt, 7),
        ('LLaVAMed', 'llavamed1p5', layer_llavamed, 25),
        ('CheXAgent', 'chexagent', layer_chexagent, 25),
    ]

    # --- Build dataset (mirrors build_data from linear_mixed_model_analysis.py) ---
    print(f"\n{'='*70}")
    print(f"  LMM DATASET BUILD: {dataset.upper()}")
    print(f"  Output: {out_dir}")
    print(f"{'='*70}")

    all_rows = []

    for display_name, csv_name, layer_fn, final_num in model_configs:
        dem_path = os.path.join(csv_dir, f"demographic_{csv_name}.csv")
        dis_path = os.path.join(csv_dir, f"disease_conditioned_{csv_name}.csv")

        if not os.path.exists(dem_path) or not os.path.exists(dis_path):
            print(f"  [{display_name}] SKIP - CSVs not found")
            continue

        dem = pd.read_csv(dem_path)
        dem["layer_num"] = dem["layer"].astype(str).map(layer_fn).astype("float")
        dem = dem.loc[dem["layer_num"].notna()].copy()

        dis = pd.read_csv(dis_path)
        dis["layer_num"] = dis["layer"].astype(str).map(layer_fn).astype("float")
        dis = dis.loc[dis["layer_num"].notna() & (dis["disease"] == "No Finding")].copy()

        if dem.empty or dis.empty:
            print(f"  [{display_name}] SKIP - empty after filtering")
            continue

        # Auto-detect final_num if None
        if final_num is None:
            final_num = int(dem["layer_num"].max())

        # Extract encoding AUC
        enc = (
            dem.loc[(dem["level"] == "OVERALL") & (dem["metric"] == "auc"),
                    ["attribute", "layer_num", "value"]]
            .rename(columns={"value": "encoding_auc"})
            .copy()
        )

        # For each demographic, compute FPR gap
        for attr, (c1, c2) in demo_pairs.items():
            dis_sub = dis.loc[dis["condition"].isin([c1, c2]), ["layer_num", "condition", "fpr"]].copy()
            if dis_sub.empty:
                continue
            fpr_wide = dis_sub.pivot_table(
                index="layer_num", columns="condition", values="fpr", aggfunc="first"
            ).reset_index()
            if c1 not in fpr_wide.columns or c2 not in fpr_wide.columns:
                continue
            fpr_gap_df = fpr_wide.loc[
                fpr_wide[c1].notna() & fpr_wide[c2].notna(), ["layer_num", c1, c2]
            ].copy()
            fpr_gap_df["fpr_gap"] = (fpr_gap_df[c1] - fpr_gap_df[c2]).abs()
            fpr_gap_df = fpr_gap_df[["layer_num", "fpr_gap"]]

            out = enc.loc[enc["attribute"] == attr].merge(fpr_gap_df, on="layer_num", how="inner")
            out["model"] = display_name
            out["attribute"] = attr
            out["final_num"] = final_num
            all_rows.append(out)

        print(f"  [{display_name}] OK")

    if not all_rows:
        print("ERROR: No data built. Check CSVs.")
        return False

    dat = pd.concat(all_rows, ignore_index=True)
    dat["layer_rel"] = dat["layer_num"] / dat["final_num"]
    dat = dat.drop(columns=["final_num"])
    dat["encoding_auc_c"] = dat.groupby("attribute")["encoding_auc"].transform(
        lambda x: x - x.mean(skipna=True)
    )
    dat = dat[["attribute", "layer_num", "encoding_auc", "fpr_gap", "model", "layer_rel", "encoding_auc_c"]]

    out_csv = os.path.join(out_dir, "01_dataset.csv")
    dat.to_csv(out_csv, index=False)

    print(f"\n  Dataset: {len(dat)} rows | {dat['model'].nunique()} models | {dat['attribute'].nunique()} demographics")
    print(f"  AUC range: {np.round([dat['encoding_auc'].min(), dat['encoding_auc'].max()], 3)}")
    print(f"  FPR gap range: {np.round([dat['fpr_gap'].min(), dat['fpr_gap'].max()], 3)}")
    print(f"  Saved: {out_csv}")
    print(f"\n  This CSV is ready for linear_mixed_model_analysis.py (requires rpy2 + R).")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mimic', 'chexpert', 'rexgradient', 'rexgradient_nonpediatric', 'all'],
                        default='all')
    parser.add_argument('--skip-csv-build', action='store_true',
                        help='Skip CSV generation (use existing CSVs)')
    args = parser.parse_args()

    datasets = ['mimic', 'chexpert', 'rexgradient'] if args.dataset == 'all' else [args.dataset]

    for ds in datasets:
        if not args.skip_csv_build:
            if not run_csv_builder(ds):
                continue
        run_lmm_for_dataset(ds)

    print("\n" + "=" * 70)
    print("  ALL DATASETS PROCESSED")
    print("  Outputs: other_models/all_models/lmm_outputs/{mimic,chexpert,rexgradient}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
