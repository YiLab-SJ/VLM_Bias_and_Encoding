#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
FAIRNESS-ENCODING LMM ANALYSIS
8 VLMs - layer-level - demographic encoding -> FPR disparity (No Finding)
Model: fpr_gap ~ encoding_auc_c + encoding_auc_c^2 + (1 + encoding_auc_c | model) + AR1
Layer depth excluded - absorbed by random slope structure
===============================================================================

Hybrid Python/R implementation
------------------------------
This script ports the provided R analysis into Python while preserving the exact
R model-estimation machinery where it matters most:

- Python/pandas handles data construction and CSV IO.
- R, called under the hood via rpy2, handles nlme::lme, mgcv::gamm, emmeans,
  intervals(), anova(), AIC(), VarCorr(), Shapiro-Wilk, and AR(1) residuals.
- Python/matplotlib handles figures, with styling matched closely to the R plots.

Important reproducibility note
------------------------------
The model estimates, AICs, emmeans, contrasts, residuals, and fixed-effect
coefficients come from the same R packages used in the original script. The plots
are recreated in matplotlib and are designed to visually match the ggplot2 output,
but they will not be pixel-identical because the graphics engines differ.

Stdout buffering
----------------
This script intentionally does not force flush on print/log statements. When run
with normal Python in a non-interactive environment, stdout remains buffered.
To force unbuffered output, users would need to run python -u, but that is not
used or required by this script.
"""

from __future__ import annotations

import io
import os
import re
import sys
import math
import json
import textwrap
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

# Make stdout explicitly buffered when possible. Do not use write_through=True.
# This preserves buffered stdout behavior requested for long runs/log capture.
try:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding=sys.stdout.encoding or "utf-8",
            line_buffering=False,
            write_through=False,
        )
except Exception:
    # If the environment has a special stdout wrapper, leave it unchanged.
    pass

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import acf as sm_acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

# rpy2 imports. R is required for exact nlme/emmeans behavior.
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
except ImportError as exc:
    raise ImportError(
        "This script requires rpy2. Install it with `pip install rpy2` and make "
        "sure R is installed and discoverable by Python."
    ) from exc

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------------------------------------------------------
# Logging / printing helpers
# ------------------------------------------------------------------------------

def log(*args, **kwargs) -> None:
    """Buffered stdout logging. Intentionally does not flush."""
    kwargs.pop("flush", None)
    print(*args, **kwargs, flush=False)


def log_df(title: str, df: pd.DataFrame, max_rows: Optional[int] = None) -> None:
    """Print a dataframe to buffered stdout with a title."""
    log(title)
    if max_rows is not None:
        log(df.head(max_rows).to_string(index=False))
    else:
        log(df.to_string(index=False))


# ------------------------------------------------------------------------------
# 0. CONSTANTS
# ------------------------------------------------------------------------------

# Dataset selection via --dataset argument (default: mimic)
_DATASET_ARG = "mimic"
for _i, _a in enumerate(sys.argv):
    if _a == "--dataset" and _i + 1 < len(sys.argv):
        _DATASET_ARG = sys.argv[_i + 1]
        sys.argv.pop(_i)
        sys.argv.pop(_i)
        break

_CSV_ROOT = Path("/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models/all_models/lmm_input_csvs")
_OUT_ROOT = Path("/home/apalliko/ondemand/embedding_project/embedding_demographic_info/embeddings_info_protocol/other_models/all_models/lmm_outputs")

BASE = _CSV_ROOT / _DATASET_ARG
OUT = _OUT_ROOT / _DATASET_ARG
OUT.mkdir(parents=True, exist_ok=True)

log(f"\n*** DATASET: {_DATASET_ARG.upper()} ***")
log(f"*** INPUT:   {BASE}")
log(f"*** OUTPUT:  {OUT}\n")

# RexGradient (and its nonpediatric variant) has no ethnicity
if _DATASET_ARG in ("rexgradient", "rexgradient_nonpediatric"):
    DEMO_PAIRS: Dict[str, Tuple[str, str]] = {
        "sex": ("given_sex_0", "given_sex_1"),
        "age": ("given_age_1", "given_age_4"),
    }
else:
    DEMO_PAIRS: Dict[str, Tuple[str, str]] = {
        "sex": ("given_sex_0", "given_sex_1"),
        "age": ("given_age_0", "given_age_3"),
        "ethnicity": ("given_ethnicity_0", "given_ethnicity_1"),
    }

DEMO_LABELS: Dict[str, str] = {
    "sex": "Sex (Female vs Male)",
    "age": "Age (80+ vs 18-39)",
    "ethnicity": "Race (Black vs White)",
}

MODEL_COLORS: Dict[str, str] = {
    "BioViLT": "#1B7837",
    "CheXAgent": "#762A83",
    "CheXZero": "#E08214",
    "LLaVAMed": "#C51B7D",
    "MedGemma": "#2166AC",
    "MedVersa": "#D6604D",
    "NVReason": "#4DAC26",
    "RadFM": "#4393C3",
}

# R ggplot shape IDs from the original script are mapped to close matplotlib markers.
# ggplot 21/22/23/24/25 are filled circle/square/diamond/triangle-up/triangle-down.
MODEL_SHAPES_R: Dict[str, int] = {
    "BioViLT": 21,
    "CheXAgent": 22,
    "CheXZero": 23,
    "LLaVAMed": 24,
    "MedGemma": 25,
    "MedVersa": 21,
    "NVReason": 22,
    "RadFM": 23,
}

MODEL_MARKERS: Dict[str, str] = {
    "BioViLT": "o",
    "CheXAgent": "s",
    "CheXZero": "D",
    "LLaVAMed": "^",
    "MedGemma": "v",
    "MedVersa": "o",
    "NVReason": "s",
    "RadFM": "D",
}

LAYER_REFS: List[float] = [0.1, 0.5, 1.0]
LAYER_LABS: List[str] = ["Early (0.1)", "Middle (0.5)", "Late (1.0)"]

LAYER_LINE_COLORS: Dict[str, str] = {
    "Early (0.1)": "#2166AC",
    "Middle (0.5)": "#4DAC26",
    "Late (1.0)": "#B2182B",
}

LAYER_LINESTYLES: Dict[str, str] = {
    "Early (0.1)": ":",
    "Middle (0.5)": "-",
    "Late (1.0)": "--",
}


def label_demo(x: str) -> str:
    if x == "sex":
        return "Sex (Female vs Male)"
    if x == "age":
        return "Age (80+ vs 18-39)"
    if x == "ethnicity":
        return "Race (Black vs White)"
    return x


# ------------------------------------------------------------------------------
# 1. MODEL REGISTRY
# ------------------------------------------------------------------------------

@dataclass
class ModelSpec:
    name: str
    dem_csv: Path
    dis_csv: Path
    layer_fn: Callable[[str], Optional[int]]
    final_num: Optional[int]


def _extract_int(pattern: str, text: str) -> Optional[int]:
    m = re.search(pattern, str(text))
    return int(m.group(1)) if m else None


def layer_chexzero(l: str) -> Optional[int]:
    n = _extract_int(r"Vis_Block_(\d+)", l)
    if n is not None:
        return n
    if l == "image_embedding_final":
        return 13
    return None


def layer_medgemma(l: str) -> Optional[int]:
    if l == "Vis_Embed":
        return 0
    n = _extract_int(r"^Vis_Block_(\d+)$", l)
    if n is not None:
        return n
    if l == "Vis_PostNorm":
        return 28
    if l == "Vis_Projected":
        return 29
    return None


def layer_radfm(l: str) -> Optional[int]:
    if l == "Vis_PatchEmbed":
        return 0
    n = _extract_int(r"^Vis_Block_(\d+)$", l)
    if n is not None:
        return n
    if l == "Vis_Perceiver":
        return 13
    if l == "Vis_Projected":
        return 14
    return None


def layer_nvreason(l: str) -> Optional[int]:
    n = _extract_int(r"^Vis_Block_(\d+)$", l)
    if n is not None:
        return n
    return None


def layer_medversa(l: str) -> Optional[int]:
    if l == "Vis_Embed":
        return 0
    if l == "Vis_S0_B1":
        return 1
    if l == "Vis_S0_B2":
        return 2
    if l == "Vis_S1_B1":
        return 3
    if l == "Vis_S1_B2":
        return 4
    n = _extract_int(r"^Vis_S2_B(\d+)$", l)
    if n is not None:
        return 4 + n
    if l == "Vis_S3_B1":
        return 23
    if l == "Vis_S3_B2":
        return 24
    if l == "Vis_FinalNorm":
        return 25
    if l == "Vis_LNVision":
        return 26
    if l == "Vis_Projected":
        return 27
    return None


def layer_biovilt(l: str) -> Optional[int]:
    mapping = {
        "Vis_ResNet_Layer1": 1,
        "Vis_ResNet_Layer2": 2,
        "Vis_ResNet_Layer3": 3,
        "Vis_ResNet_Layer4": 4,
        "Vis_BackboneToViT": 5,
        "img_embedding": 6,
        "image_embedding_final": 7,
    }
    return mapping.get(l)


def layer_llavamed(l: str) -> Optional[int]:
    if l == "Vis_CLIP_Embed":
        return 0
    n = _extract_int(r"^Vis_CLIP_Layer_(\d+)$", l)
    if n is not None:
        return n
    if l == "Vis_MM_Projector":
        return 25
    return None


def layer_chexagent(l: str) -> Optional[int]:
    n = _extract_int(r"^Vis_SigLIP_Block_(\d+)$", l)
    if n is not None:
        return n
    if l == "image_embedding_final":
        return 25
    return None


MODEL_REGISTRY: List[ModelSpec] = [
    ModelSpec(
        name="CheXZero",
        dem_csv=BASE / "demographic_chexzero.csv",
        dis_csv=BASE / "disease_conditioned_chexzero.csv",
        layer_fn=layer_chexzero,
        final_num=13,
    ),
    ModelSpec(
        name="MedGemma",
        dem_csv=BASE / "demographic_medgemma_1p5.csv",
        dis_csv=BASE / "disease_conditioned_medgemma_1p5.csv",
        layer_fn=layer_medgemma,
        final_num=29,
    ),
    ModelSpec(
        name="RadFM",
        dem_csv=BASE / "demographic_radfm.csv",
        dis_csv=BASE / "disease_conditioned_radfm.csv",
        layer_fn=layer_radfm,
        final_num=14,
    ),
    ModelSpec(
        name="NVReason",
        dem_csv=BASE / "demographic_nv_reason.csv",
        dis_csv=BASE / "disease_conditioned_nv_reason.csv",
        layer_fn=layer_nvreason,
        final_num=None,
    ),
    ModelSpec(
        name="MedVersa",
        dem_csv=BASE / "demographic_medversa.csv",
        dis_csv=BASE / "disease_conditioned_medversa.csv",
        layer_fn=layer_medversa,
        final_num=27,
    ),
    ModelSpec(
        name="BioViLT",
        dem_csv=BASE / "demographic_biovilt.csv",
        dis_csv=BASE / "disease_conditioned_biovilt.csv",
        layer_fn=layer_biovilt,
        final_num=7,
    ),
    ModelSpec(
        name="LLaVAMed",
        dem_csv=BASE / "demographic_llavamed1p5.csv",
        dis_csv=BASE / "disease_conditioned_llavamed1p5.csv",
        layer_fn=layer_llavamed,
        final_num=25,
    ),
    ModelSpec(
        name="CheXAgent",
        dem_csv=BASE / "demographic_chexagent.csv",
        dis_csv=BASE / "disease_conditioned_chexagent.csv",
        layer_fn=layer_chexagent,
        final_num=25,
    ),
]


def apply_layer_fn(df: pd.DataFrame, fn: Callable[[str], Optional[int]]) -> pd.Series:
    return df["layer"].astype(str).map(fn).astype("float")


# Auto-detect final_num for NULL entries
for idx, spec in enumerate(MODEL_REGISTRY):
    if spec.final_num is not None:
        continue
    try:
        dem_tmp = pd.read_csv(spec.dem_csv)
        nums = apply_layer_fn(dem_tmp, spec.layer_fn).dropna()
        final_num = int(nums.max()) if len(nums) else np.nan
        MODEL_REGISTRY[idx] = ModelSpec(
            name=spec.name,
            dem_csv=spec.dem_csv,
            dis_csv=spec.dis_csv,
            layer_fn=spec.layer_fn,
            final_num=final_num,
        )
        log("Auto final_num:", spec.name, "->", final_num)
    except Exception:
        MODEL_REGISTRY[idx] = ModelSpec(
            name=spec.name,
            dem_csv=spec.dem_csv,
            dis_csv=spec.dis_csv,
            layer_fn=spec.layer_fn,
            final_num=np.nan,
        )


# ==============================================================================
# STEP 1 - BUILD DATASET
# ==============================================================================


def build_data(registry: List[ModelSpec]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []

    for m in registry:
        try:
            dem = pd.read_csv(m.dem_csv)
            dem["layer_num"] = apply_layer_fn(dem, m.layer_fn)
            dem = dem.loc[dem["layer_num"].notna()].copy()
        except Exception:
            log("dem fail:", m.name)
            dem = None

        try:
            dis = pd.read_csv(m.dis_csv)
            dis["layer_num"] = apply_layer_fn(dis, m.layer_fn)
            dis = dis.loc[
                dis["layer_num"].notna() & (dis["disease"] == "No Finding")
            ].copy()
        except Exception:
            log("dis fail:", m.name)
            dis = None

        if dem is None or dis is None:
            continue

        enc = (
            dem.loc[(dem["level"] == "OVERALL") & (dem["metric"] == "auc"),
                    ["attribute", "layer_num", "value"]]
            .rename(columns={"value": "encoding_auc"})
            .copy()
        )

        for attr, (c1, c2) in DEMO_PAIRS.items():
            dis_sub = dis.loc[dis["condition"].isin([c1, c2]), ["layer_num", "condition", "fpr"]].copy()
            if dis_sub.empty:
                continue
            fpr_wide = dis_sub.pivot_table(
                index="layer_num",
                columns="condition",
                values="fpr",
                aggfunc="first",
            ).reset_index()
            if c1 not in fpr_wide.columns or c2 not in fpr_wide.columns:
                continue
            fpr_gap_df = fpr_wide.loc[fpr_wide[c1].notna() & fpr_wide[c2].notna(), ["layer_num", c1, c2]].copy()
            fpr_gap_df["fpr_gap"] = (fpr_gap_df[c1] - fpr_gap_df[c2]).abs()
            fpr_gap_df = fpr_gap_df[["layer_num", "fpr_gap"]]

            out = enc.loc[enc["attribute"] == attr].merge(fpr_gap_df, on="layer_num", how="inner")
            out["model"] = m.name
            out["attribute"] = attr
            rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["attribute", "layer_num", "encoding_auc", "fpr_gap", "model"])
    return pd.concat(rows, ignore_index=True)


dat = build_data(MODEL_REGISTRY)
final_df = pd.DataFrame(
    [{"model": m.name, "final_num": m.final_num} for m in MODEL_REGISTRY]
)
dat = dat.merge(final_df, on="model", how="left")
dat["layer_rel"] = dat["layer_num"] / dat["final_num"]
dat = dat.drop(columns=["final_num"])
dat["encoding_auc_c"] = dat.groupby("attribute")["encoding_auc"].transform(lambda x: x - x.mean(skipna=True))
dat = dat[["attribute", "layer_num", "encoding_auc", "fpr_gap", "model", "layer_rel", "encoding_auc_c"]]

log(
    "\nDataset:",
    len(dat),
    "rows |",
    dat["model"].nunique(),
    "models |",
    dat["attribute"].nunique(),
    "demographics",
)
if len(dat):
    log("AUC range:", np.round([dat["encoding_auc"].min(), dat["encoding_auc"].max()], 3))
    log("FPR gap range:", np.round([dat["fpr_gap"].min(), dat["fpr_gap"].max()], 3))
dat.to_csv(OUT / "01_dataset.csv", index=False)


# ------------------------------------------------------------------------------
# R setup for exact model estimation
# ------------------------------------------------------------------------------

pandas2ri.activate()

# Import R packages used by the original analysis.
base = importr("base")
stats_r = importr("stats")
nlme = importr("nlme")
mgcv = importr("mgcv")
emmeans_pkg = importr("emmeans")
car_pkg = importr("car")
utils = importr("utils")

ro.r("options(stringsAsFactors = FALSE)")
ro.r("suppressPackageStartupMessages(library(nlme))")
ro.r("suppressPackageStartupMessages(library(mgcv))")
ro.r("suppressPackageStartupMessages(library(emmeans))")
ro.r("suppressPackageStartupMessages(library(car))")


def py_to_r_df(df: pd.DataFrame):
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.py2rpy(df)


def r_to_py_df(obj) -> pd.DataFrame:
    with localconverter(ro.default_converter + pandas2ri.converter):
        return ro.conversion.rpy2py(obj)


# R helper functions. These preserve the nlme/emmeans/mgcv behavior of the R script.
ro.r(
    r'''
    fit_null_icc_r <- function(d) {
      fit <- lme(fpr_gap ~ 1, random = ~1|model, data = d, method = "REML")
      vc  <- as.numeric(VarCorr(fit)[, "Variance"])
      data.frame(
        ICC    = round(vc[1] / sum(vc), 3),
        tau2   = round(vc[1], 5),
        sigma2 = round(vc[2], 5)
      )
    }

    fit_model_selection_r <- function(d, attr) {
      d <- d[order(d$model, d$layer_num), ]

      fit_lin     <- lme(fpr_gap ~ encoding_auc_c,
                         random = ~1|model, data = d, method = "ML")

      fit_quad    <- lme(fpr_gap ~ encoding_auc_c + I(encoding_auc_c^2),
                         random = ~1|model, data = d, method = "ML")

      fit_quad_rs <- lme(fpr_gap ~ encoding_auc_c + I(encoding_auc_c^2),
                         random  = ~encoding_auc_c|model,
                         data = d, method = "ML",
                         control = lmeControl(opt = "optim", maxIter = 200))

      fit_ar1 <- tryCatch(
        lme(fpr_gap ~ encoding_auc_c + I(encoding_auc_c^2),
            random      = ~encoding_auc_c|model,
            correlation = corAR1(form = ~layer_num|model),
            data = d, method = "ML",
            control = lmeControl(opt = "optim", maxIter = 200)),
        error = function(e) { message("AR1 failed: ", attr); NULL }
      )

      fit_gamm <- tryCatch(
        gamm(fpr_gap ~ s(encoding_auc, k = 5, bs = "cr"),
             random = list(model = ~1), data = d, method = "ML"),
        error = function(e) NULL
      )

      lrt <- anova(fit_lin, fit_quad)
      edf <- if (!is.null(fit_gamm)) {
        summary(fit_gamm$gam)$s.table["s(encoding_auc)", "edf"]
      } else {
        NA_real_
      }

      data.frame(
        demographic       = attr,
        AIC_linear        = round(AIC(fit_lin),     1),
        AIC_quadratic     = round(AIC(fit_quad),    1),
        AIC_quad_RS       = round(AIC(fit_quad_rs), 1),
        AIC_quad_RS_AR1   = if (!is.null(fit_ar1)) round(AIC(fit_ar1), 1) else NA_real_,
        AIC_gamm_k5       = if (!is.null(fit_gamm)) round(AIC(fit_gamm$lme), 1) else NA_real_,
        dAIC_lin_vs_quad  = round(AIC(fit_lin)     - AIC(fit_quad),    1),
        dAIC_quad_vs_RS   = round(AIC(fit_quad)    - AIC(fit_quad_rs), 1),
        dAIC_RS_vs_AR1    = if (!is.null(fit_ar1)) round(AIC(fit_quad_rs) - AIC(fit_ar1), 1) else NA_real_,
        LRT_p             = round(lrt[["p-value"]][2], 4),
        GAMM_EDF          = round(edf, 2),
        EDF_verdict       = if (is.na(edf)) {
                              "GAMM failed"
                            } else if (edf < 1.5) {
                              "linear"
                            } else if (edf < 2.5) {
                              "quadratic confirmed"
                            } else {
                              "more complex"
                            }
      )
    }

    fit_final_r <- function(d) {
      d <- d[order(d$model, d$layer_num), ]
      fit <- tryCatch(
        lme(fpr_gap ~ encoding_auc_c + I(encoding_auc_c^2) + layer_rel,
            random      = ~encoding_auc_c|model,
            correlation = corAR1(form = ~layer_num|model),
            data        = d,
            method      = "REML",
            control     = lmeControl(opt = "optim", maxIter = 200)),
        error = function(e) {
          message("Full model singular, falling back to random-intercept + AR1: ", e$message)
          NULL
        }
      )
      if (!is.null(fit)) return(fit)
      # Fallback: random intercept only + AR1
      lme(fpr_gap ~ encoding_auc_c + I(encoding_auc_c^2) + layer_rel,
          random      = ~1|model,
          correlation = corAR1(form = ~layer_num|model),
          data        = d,
          method      = "REML",
          control     = lmeControl(opt = "optim", maxIter = 200))
    }

    extract_final_results_r <- function(fit, d, attr) {
      ar1_rho  <- coef(fit$modelStruct$corStruct, unconstrained = FALSE)
      tt       <- summary(fit)$tTable
      ci       <- intervals(fit, which = "fixed")$fixed
      mean_auc <- mean(d$encoding_auc[d$attribute == attr], na.rm = TRUE)
      b1 <- tt["encoding_auc_c",      "Value"]
      b2 <- tt["I(encoding_auc_c^2)", "Value"]
      vc            <- VarCorr(fit)
      var_intercept <- as.numeric(vc["(Intercept)",    "Variance"])
      # Handle fallback model without random slopes
      var_slope     <- if ("encoding_auc_c" %in% rownames(vc)) {
                         as.numeric(vc["encoding_auc_c", "Variance"])
                       } else { 0.0 }
      var_resid     <- as.numeric(vc["Residual",        "Variance"])

      data.frame(
        demographic    = attr,
        AR1_rho        = round(ar1_rho, 3),
        mean_auc       = round(mean_auc, 3),
        beta_linear    = round(b1, 4),
        beta_quadratic = round(b2, 4),
        ci_low_quad    = round(ci["I(encoding_auc_c^2)", "lower"], 4),
        ci_high_quad   = round(ci["I(encoding_auc_c^2)", "upper"], 4),
        p_linear       = round(tt["encoding_auc_c",      "p-value"], 4),
        p_quadratic    = round(tt["I(encoding_auc_c^2)", "p-value"], 4),
        beta_layer_rel = round(tt["layer_rel",            "Value"],   4),
        p_layer_rel    = round(tt["layer_rel",            "p-value"], 4),
        peak_auc       = round((-b1 / (2*b2)) + mean_auc, 3),
        var_intercept  = round(var_intercept, 6),
        var_slope      = round(var_slope,     6),
        var_resid      = round(var_resid,     6),
        ICC            = round(var_intercept / (var_intercept + var_slope + var_resid), 3),
        n_obs          = nrow(d[!is.na(d$encoding_auc_c) & !is.na(d$fpr_gap), ])
      )
    }

    residuals_fitted_r <- function(fit) {
      data.frame(
        fitted = fitted(fit),
        resid = residuals(fit, type = "normalized")
      )
    }

    shapiro_r <- function(fit) {
      r <- residuals(fit, type = "normalized")
      sw <- shapiro.test(r)
      data.frame(
        W = round(unname(sw$statistic), 4),
        p = round(sw$p.value, 4),
        normal = sw$p.value > 0.05
      )
    }

    fixed_effects_r <- function(fit) {
      fe <- fixef(fit)
      data.frame(term = names(fe), estimate = as.numeric(fe))
    }

    emm_means_r <- function(fit, attr, d, layer_refs, layer_labs) {
      mean_auc <- mean(d$encoding_auc, na.rm = TRUE)
      ref_raw  <- as.numeric(quantile(d$encoding_auc, c(0.10, 0.50, 0.90), na.rm = TRUE))
      ref_c    <- ref_raw - mean_auc
      out <- list()

      for (i in seq_along(layer_refs)) {
        lr  <- layer_refs[i]
        lab <- layer_labs[i]
        emm <- emmeans(fit,
                       ~encoding_auc_c,
                       at      = list(encoding_auc_c = ref_c,
                                      layer_rel      = lr),
                       mode    = "appx-satterthwaite",
                       pbkrtest.limit = 5000)
        df <- as.data.frame(emm)
        df$demographic  <- attr
        df$encoding_auc <- round(ref_raw, 3)
        df$label        <- c("Low (p10)", "Median (p50)", "High (p90)")
        df$layer_label  <- lab
        df$layer_rel    <- lr
        out[[i]] <- df[, c("demographic", "layer_label", "layer_rel", "label", "encoding_auc",
                           "emmean", "SE", "lower.CL", "upper.CL")]
      }
      do.call(rbind, out)
    }

    emm_contrasts_r <- function(fit, attr, d, layer_refs, layer_labs) {
      mean_auc <- mean(d$encoding_auc, na.rm = TRUE)
      ref_c    <- as.numeric(quantile(d$encoding_auc, c(0.10, 0.50, 0.90), na.rm = TRUE)) - mean_auc
      out <- list()

      for (i in seq_along(layer_refs)) {
        lr  <- layer_refs[i]
        lab <- layer_labs[i]
        emm <- emmeans(fit,
                       ~encoding_auc_c,
                       at      = list(encoding_auc_c = ref_c,
                                      layer_rel      = lr),
                       mode    = "appx-satterthwaite",
                       pbkrtest.limit = 5000)
        df <- as.data.frame(pairs(emm, adjust = "bonferroni"))
        df$demographic    <- attr
        df$layer_label    <- lab
        df$layer_rel      <- lr
        df$contrast_label <- c("Low vs Median", "Low vs High", "Median vs High")
        out[[i]] <- df[, c("demographic", "layer_label", "layer_rel", "contrast_label",
                           "estimate", "SE", "df", "t.ratio", "p.value")]
      }
      do.call(rbind, out)
    }
    '''
)


# ==============================================================================
# STEP 2 - PRE-MODEL CHECKS
# ==============================================================================

# 2a. VIF - justifies centring

def compute_vif_two_terms(x1: pd.Series, x2: pd.Series) -> Tuple[float, float]:
    X = pd.DataFrame({"x1": x1, "x2": x2}).dropna()
    X_const = sm.add_constant(X, has_constant="add")
    v1 = variance_inflation_factor(X_const.values, X_const.columns.get_loc("x1"))
    v2 = variance_inflation_factor(X_const.values, X_const.columns.get_loc("x2"))
    return float(v1), float(v2)


vif_rows = []
for attr in DEMO_PAIRS.keys():
    d = dat.loc[dat["attribute"] == attr].copy()
    v_raw_1, v_raw_2 = compute_vif_two_terms(d["encoding_auc"], d["encoding_auc"] ** 2)
    v_ctr_1, v_ctr_2 = compute_vif_two_terms(d["encoding_auc_c"], d["encoding_auc_c"] ** 2)
    vif_rows.append(
        {
            "demographic": attr,
            "VIF_uncentred": round(v_raw_1, 2),
            "VIF_uncentred_sq": round(v_raw_2, 2),
            "VIF_centred": round(v_ctr_1, 2),
            "VIF_centred_sq": round(v_ctr_2, 2),
            "centring_needed": bool(v_raw_1 > 10),
        }
    )
vif_check = pd.DataFrame(vif_rows)
log_df("\nVIF:", vif_check)
vif_check.to_csv(OUT / "02a_vif.csv", index=False)

# 2b. ICC - justifies random intercept
icc_rows = []
for attr in DEMO_PAIRS.keys():
    d = dat.loc[dat["attribute"] == attr].copy()
    r_df = ro.r["fit_null_icc_r"](py_to_r_df(d))
    out = r_to_py_df(r_df)
    out.insert(0, "demographic", attr)
    icc_rows.append(out)
icc_check = pd.concat(icc_rows, ignore_index=True)
log_df("\nNull-model ICCs:", icc_check)
icc_check.to_csv(OUT / "02b_icc_null.csv", index=False)


# 2c. Descriptive plot

def theme_classic_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_facecolor("white")


def add_strip(ax, text: str) -> None:
    ax.add_patch(Rectangle((0, 1.01), 1, 0.11, transform=ax.transAxes,
                           facecolor="#f2f2f2", edgecolor="none", clip_on=False))
    ax.text(0.5, 1.065, text, transform=ax.transAxes,
            ha="center", va="center", fontsize=10)


def savefig(path: Path, fig: plt.Figure, dpi: int = 300) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
for ax, attr in zip(axes, DEMO_PAIRS.keys()):
    d = dat.loc[dat["attribute"] == attr].copy()
    for model, g in d.groupby("model"):
        ax.scatter(
            g["encoding_auc"],
            g["fpr_gap"],
            s=22,
            marker=MODEL_MARKERS.get(model, "o"),
            facecolors=MODEL_COLORS.get(model, "gray"),
            edgecolors=MODEL_COLORS.get(model, "gray"),
            alpha=0.6,
            linewidths=0.4,
            label=model,
        )
    dd = d[["encoding_auc", "fpr_gap"]].dropna().sort_values("encoding_auc")
    if len(dd) >= 5:
        lo = lowess(dd["fpr_gap"], dd["encoding_auc"], frac=0.75, return_sorted=True)
        ax.plot(lo[:, 0], lo[:, 1], color="#262626", linewidth=1.0)
        # Approximate loess confidence ribbon is intentionally omitted because ggplot2's
        # loess SE calculation is not reproduced exactly by statsmodels.lowess.
    add_strip(ax, label_demo(attr))
    theme_classic_axes(ax)
    ax.set_xlabel("Encoding AUC")
    ax.set_ylabel("FPR gap (No Finding)" if ax is axes[0] else "")

handles = [
    Line2D([0], [0], marker=MODEL_MARKERS[m], color="none", label=m,
           markerfacecolor=MODEL_COLORS[m], markeredgecolor=MODEL_COLORS[m], markersize=6)
    for m in MODEL_COLORS.keys()
]
fig.legend(handles=handles, loc="center right", frameon=False)
fig.suptitle("Raw relationship: encoding AUC vs FPR gap", fontsize=12, fontweight="normal", y=1.05)
fig.text(0.5, 0.98, "Loess smoother - inspect shape before committing to quadratic", ha="center", fontsize=10)
fig.tight_layout(rect=[0, 0, 0.88, 0.94])
savefig(OUT / "02c_raw_loess.png", fig)
plt.close(fig)


# ==============================================================================
# STEP 3 - MODEL SELECTION (ML for valid AIC comparison)
# fpr_gap ~ encoding_auc_c + encoding_auc_c^2 + random effects
# No layer_rel - variation across layers absorbed by random slope
# ==============================================================================

model_selection_rows = []
for attr in DEMO_PAIRS.keys():
    d = dat.loc[
        (dat["attribute"] == attr)
        & dat["encoding_auc_c"].notna()
        & dat["fpr_gap"].notna()
    ].sort_values(["model", "layer_num"]).copy()
    r_df = ro.r["fit_model_selection_r"](py_to_r_df(d), attr)
    model_selection_rows.append(r_to_py_df(r_df))
model_selection = pd.concat(model_selection_rows, ignore_index=True)
log_df("\nModel selection:", model_selection)
model_selection.to_csv(OUT / "03_model_selection.csv", index=False)


# ==============================================================================
# UPDATED ANALYSIS - with layer_rel as covariate
# EMMs computed at encoding AUC p10/p50/p90 x layer_rel 0.1/0.5/1.0
# ==============================================================================

# -- STEP 4 - FINAL MODEL (with layer_rel) -------------------------------------

fits: Dict[str, object] = {}
r_data_by_attr: Dict[str, object] = {}
py_data_by_attr: Dict[str, pd.DataFrame] = {}

for attr in DEMO_PAIRS.keys():
    d = dat.loc[
        (dat["attribute"] == attr)
        & dat["encoding_auc_c"].notna()
        & dat["fpr_gap"].notna()
        & dat["layer_rel"].notna()
    ].sort_values(["model", "layer_num"]).copy()
    log(attr, "| n =", len(d))
    py_data_by_attr[attr] = d
    r_d = py_to_r_df(d)
    r_data_by_attr[attr] = r_d
    fits[attr] = ro.r["fit_final_r"](r_d)

result_rows = []
for attr, fit in fits.items():
    r_df = ro.r["extract_final_results_r"](fit, r_data_by_attr[attr], attr)
    result_rows.append(r_to_py_df(r_df))
results = pd.concat(result_rows, ignore_index=True)
results["p_quad_bonferroni"] = np.round(np.minimum(results["p_quadratic"] * len(results), 1.0), 4)

def shape_from_row(row) -> str:
    if row["p_quadratic"] >= 0.05:
        return "no significant curvature"
    if row["beta_quadratic"] < 0:
        return "inverted-U"
    return "U-shaped"


def sig_from_p(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

results["shape"] = results.apply(shape_from_row, axis=1)
results["sig"] = results["p_quad_bonferroni"].map(sig_from_p)

log_df("\n== FINAL RESULTS ==", results)
results.to_csv(OUT / "04_results.csv", index=False)


# ==============================================================================
# STEP 5 - DIAGNOSTICS
# ==============================================================================

# 5a. Shapiro-Wilk
sw_rows = []
for attr, fit in fits.items():
    out = r_to_py_df(ro.r["shapiro_r"](fit))
    out.insert(0, "demographic", attr)
    sw_rows.append(out)
sw = pd.concat(sw_rows, ignore_index=True)
log_df("\nShapiro-Wilk:", sw)
sw.to_csv(OUT / "05a_shapiro.csv", index=False)

# 5b. QQ plot
qq_rows = []
for attr, fit in fits.items():
    rf = r_to_py_df(ro.r["residuals_fitted_r"](fit))
    tmp = pd.DataFrame({"resid": rf["resid"], "demographic": DEMO_LABELS[attr]})
    qq_rows.append(tmp)
qq_data = pd.concat(qq_rows, ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=False)
for ax, (demo, g) in zip(axes, qq_data.groupby("demographic", sort=False)):
    resid = g["resid"].dropna().values
    osm, osr = stats.probplot(resid, dist="norm", fit=False)
    ax.scatter(osm, osr, s=14, color="steelblue", alpha=0.5)
    if len(resid) > 1:
        slope, intercept, r = stats.probplot(resid, dist="norm", fit=True)[1]
        xline = np.array([min(osm), max(osm)])
        ax.plot(xline, intercept + slope * xline, color="red", linewidth=0.8)
    add_strip(ax, demo)
    theme_classic_axes(ax)
    ax.set_xlabel("Theoretical")
    ax.set_ylabel("Sample" if ax is axes[0] else "")
fig.suptitle("QQ plot - normalised residuals", fontsize=12, y=1.04)
fig.tight_layout()
savefig(OUT / "05b_qq.png", fig)
plt.close(fig)

# 5c. Residuals vs fitted
resid_rows = []
for attr, fit in fits.items():
    d = dat.loc[
        (dat["attribute"] == attr)
        & dat["encoding_auc_c"].notna()
        & dat["fpr_gap"].notna()
    ].sort_values(["model", "layer_num"]).copy()
    rf = r_to_py_df(ro.r["residuals_fitted_r"](fit))
    tmp = pd.DataFrame(
        {
            "fitted": rf["fitted"].values,
            "resid": rf["resid"].values,
            "model": d["model"].values[: len(rf)],
            "demographic": DEMO_LABELS[attr],
        }
    )
    resid_rows.append(tmp)
resid_data = pd.concat(resid_rows, ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
for ax, (demo, gdemo) in zip(axes, resid_data.groupby("demographic", sort=False)):
    for model, g in gdemo.groupby("model"):
        ax.scatter(g["fitted"], g["resid"], s=18, color=MODEL_COLORS.get(model, "gray"), alpha=0.5)
    ax.axhline(0, color="#4d4d4d", linestyle="--", linewidth=0.8)
    dd = gdemo[["fitted", "resid"]].dropna().sort_values("fitted")
    if len(dd) >= 5:
        lo = lowess(dd["resid"], dd["fitted"], frac=0.75, return_sorted=True)
        ax.plot(lo[:, 0], lo[:, 1], color="red", linewidth=0.9)
    add_strip(ax, demo)
    theme_classic_axes(ax)
    ax.set_xlabel("Fitted")
    ax.set_ylabel("Normalised residuals" if ax is axes[0] else "")
fig.suptitle("Residuals vs fitted", fontsize=12, y=1.06)
fig.text(0.5, 0.98, "Loess should be flat at zero", ha="center", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
savefig(OUT / "05c_resid_vs_fitted.png", fig)
plt.close(fig)

# 5d. Post-AR1 ACF check
acf_rows = []
for attr, fit in fits.items():
    d = dat.loc[
        (dat["attribute"] == attr)
        & dat["encoding_auc_c"].notna()
        & dat["fpr_gap"].notna()
    ].sort_values(["model", "layer_num"]).copy()
    rf = r_to_py_df(ro.r["residuals_fitted_r"](fit))
    d = d.iloc[: len(rf)].copy()
    d["resid"] = rf["resid"].values
    for model, g in d.groupby("model"):
        if len(g) < 4:
            lag1 = np.nan
        else:
            lag1 = sm_acf(g["resid"].values, nlags=2, fft=False)[1]
        acf_rows.append(
            {"model": model, "n_layers": len(g), "lag1_acf": lag1, "demographic": attr}
        )
acf_post = pd.DataFrame(acf_rows)
log_df("\nPost-AR1 lag-1 ACF:", acf_post)
acf_post.to_csv(OUT / "05d_acf_post_ar1.csv", index=False)


# ==============================================================================
# STEP 6 - EMM at p10/p50/p90 x layer_rel 0.1/0.5/1.0
# ==============================================================================

layer_refs_r = ro.FloatVector(LAYER_REFS)
layer_labs_r = ro.StrVector(LAYER_LABS)

emm_mean_rows = []
emm_contrast_rows = []
for attr, fit in fits.items():
    d_r = r_data_by_attr[attr]
    mm = r_to_py_df(ro.r["emm_means_r"](fit, attr, d_r, layer_refs_r, layer_labs_r))
    cc = r_to_py_df(ro.r["emm_contrasts_r"](fit, attr, d_r, layer_refs_r, layer_labs_r))
    emm_mean_rows.append(mm)
    emm_contrast_rows.append(cc)

emm_means = pd.concat(emm_mean_rows, ignore_index=True).rename(columns={"emmean": "predicted_gap"})
emm_contrasts = pd.concat(emm_contrast_rows, ignore_index=True)

log_df("\nEMM means:", emm_means)
log_df("\nEMM contrasts:", emm_contrasts)
emm_means.to_csv(OUT / "06a_emm_means.csv", index=False)
emm_contrasts.to_csv(OUT / "06b_emm_contrasts.csv", index=False)


# ==============================================================================
# STEP 7 - FIGURE - one panel per demographic, three layer depth curves
# ==============================================================================


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "p<.001"
    if p < 0.05:
        return f"p={p:.3f}"
    return f"p={p:.3f}"


def get_fixef_dict(attr: str) -> Dict[str, float]:
    fe_df = r_to_py_df(ro.r["fixed_effects_r"](fits[attr]))
    return dict(zip(fe_df["term"], fe_df["estimate"]))


def get_direction(attr: str, res: pd.Series, d: pd.DataFrame) -> str:
    p10_val = d["encoding_auc"].quantile(0.10)
    p90_val = d["encoding_auc"].quantile(0.90)
    mid_low = p10_val + (p90_val - p10_val) * 0.33
    mid_high = p10_val + (p90_val - p10_val) * 0.67
    if res["shape"] == "no significant curvature":
        return "no significant curvature"
    if res["shape"] == "inverted-U" and res["peak_auc"] < mid_low:
        return "peaks at low encoding"
    if res["shape"] == "inverted-U" and res["peak_auc"] > mid_high:
        return "peaks at high encoding"
    if res["shape"] == "inverted-U":
        return "peaks at intermediate encoding"
    if res["shape"] == "U-shaped" and res["peak_auc"] < mid_low:
        return "minimum at low encoding"
    if res["shape"] == "U-shaped" and res["peak_auc"] > mid_high:
        return "minimum at high encoding"
    return "minimum at intermediate encoding"


def panel_border(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#999999")
        spine.set_linewidth(0.4)
    ax.grid(axis="y", color="#ededed", linewidth=0.3)
    ax.set_facecolor("white")


y_range = (0.0, float(dat["fpr_gap"].max() * 1.15))


def make_panel(ax, attr: str, show_legend: bool = False) -> None:
    d = dat.loc[
        (dat["attribute"] == attr)
        & dat["encoding_auc_c"].notna()
        & dat["fpr_gap"].notna()
        & dat["layer_rel"].notna()
    ].copy()
    res = results.loc[results["demographic"] == attr].iloc[0]
    fe = get_fixef_dict(attr)
    mean_auc = float(res["mean_auc"])

    x_min = d["encoding_auc"].min()
    x_max = d["encoding_auc"].max()
    x_pad = (x_max - x_min) * 0.02
    x_rng = (x_min - x_pad, x_max + x_pad)
    xseq = np.linspace(x_rng[0], x_rng[1], 300)
    xc = xseq - mean_auc

    for model, g in d.groupby("model"):
        ax.scatter(
            g["encoding_auc"],
            g["fpr_gap"],
            s=18,
            marker=MODEL_MARKERS.get(model, "o"),
            facecolors=MODEL_COLORS.get(model, "gray"),
            edgecolors="#1a1a1a",
            linewidths=0.2,
            alpha=0.25,
        )

    for lr, lab in zip(LAYER_REFS, LAYER_LABS):
        pred_y = (
            fe["(Intercept)"]
            + fe["encoding_auc_c"] * xc
            + fe["I(encoding_auc_c^2)"] * (xc ** 2)
            + fe["layer_rel"] * lr
        )
        mask = (pred_y >= 0) & (pred_y <= y_range[1])
        ax.plot(
            xseq[mask],
            pred_y[mask],
            color=LAYER_LINE_COLORS[lab],
            linestyle=LAYER_LINESTYLES[lab],
            linewidth=1.3,
            label=lab,
        )

    emm_d = emm_means.loc[emm_means["demographic"] == attr].copy()
    for lab, g in emm_d.groupby("layer_label", sort=False):
        color = LAYER_LINE_COLORS[lab]
        ax.errorbar(
            g["encoding_auc"],
            g["predicted_gap"],
            yerr=[g["predicted_gap"] - g["lower.CL"], g["upper.CL"] - g["predicted_gap"]],
            fmt="none",
            ecolor=color,
            elinewidth=0.7,
            capsize=3,
            capthick=0.7,
        )
        ax.scatter(
            g["encoding_auc"],
            g["predicted_gap"],
            s=34,
            marker="o",
            facecolors=color,
            edgecolors="white",
            linewidths=0.8,
        )

    ax.set_title(label_demo(attr), fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(x_rng)
    ax.set_ylim(y_range)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("Encoding AUC", fontsize=9.5)
    ax.set_ylabel("FPR disparity gap", fontsize=9.5)
    ax.tick_params(axis="both", labelsize=8.5)
    panel_border(ax)

    if show_legend:
        handles = [
            Line2D([0], [0], color=LAYER_LINE_COLORS[lab], linestyle=LAYER_LINESTYLES[lab],
                   linewidth=1.3, marker="o", markerfacecolor=LAYER_LINE_COLORS[lab],
                   markeredgecolor="white", label=lab)
            for lab in LAYER_LABS
        ]
        ax.legend(handles=handles, title="Layer depth", frameon=False,
                  fontsize=8, title_fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))


attrs_to_plot = list(results["demographic"].dropna().unique())

fig, axes = plt.subplots(
    1,
    len(attrs_to_plot),
    figsize=(5 * len(attrs_to_plot), 5.5),
    sharey=True,
)

if len(attrs_to_plot) == 1:
    axes = [axes]

for i, attr in enumerate(attrs_to_plot):
    make_panel(
        axes[i],
        attr,
        show_legend=(i == len(attrs_to_plot) - 1),
    )

fig.suptitle("Demographic encoding strength and FPR disparity across VLM layers",
             fontsize=13, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0.02, 0.94, 0.95])
savefig(OUT / "07_figure_layer_depth.png", fig)
savefig(OUT / "07_figure_layer_depth.pdf", fig)
plt.close(fig)

log("Saved: 07_figure_layer_depth")


# Individual Plots
# ==============================================================================
# THREE SEPARATE FIGURES - one per layer depth (0.1, 0.5, 1.0)
# ==============================================================================


def make_layer_panel(ax, attr: str, target_lr: float, layer_lab: str, show_legend: bool) -> None:
    d = dat.loc[
        (dat["attribute"] == attr)
        & dat["encoding_auc_c"].notna()
        & dat["fpr_gap"].notna()
        & dat["layer_rel"].notna()
    ].copy()
    res = results.loc[results["demographic"] == attr].iloc[0]
    fe = get_fixef_dict(attr)
    mean_auc = float(res["mean_auc"])

    x_min = d["encoding_auc"].min()
    x_max = d["encoding_auc"].max()
    x_pad = (x_max - x_min) * 0.02
    x_rng = (x_min - x_pad, x_max + x_pad)
    xseq = np.linspace(x_rng[0], x_rng[1], 300)
    xc = xseq - mean_auc

    pred_y = (
        fe["(Intercept)"]
        + fe["encoding_auc_c"] * xc
        + fe["I(encoding_auc_c^2)"] * (xc ** 2)
        + fe["layer_rel"] * target_lr
    )
    mask = (pred_y >= 0) & (pred_y <= y_range[1])

    peak_y = max(
        0.0,
        fe["(Intercept)"]
        + fe["encoding_auc_c"] * (float(res["peak_auc"]) - mean_auc)
        + fe["I(encoding_auc_c^2)"] * ((float(res["peak_auc"]) - mean_auc) ** 2)
        + fe["layer_rel"] * target_lr,
    )
    peak_in_data = bool((float(res["peak_auc"]) >= x_min) and (float(res["peak_auc"]) <= x_max))

    for model, g in d.groupby("model"):
        ax.scatter(
            g["encoding_auc"],
            g["fpr_gap"],
            s=20,
            marker=MODEL_MARKERS.get(model, "o"),
            facecolors=MODEL_COLORS.get(model, "gray"),
            edgecolors="#1a1a1a",
            linewidths=0.25,
            alpha=0.55,
            label=model,
        )

    ax.plot(xseq[mask], pred_y[mask], color="#1a1a1a", linewidth=1.4)

    emm_d = emm_means.loc[(emm_means["demographic"] == attr) & (emm_means["layer_rel"] == target_lr)].copy()
    emm_con = emm_contrasts.loc[(emm_contrasts["demographic"] == attr) & (emm_contrasts["layer_rel"] == target_lr)].copy()

    ax.errorbar(
        emm_d["encoding_auc"],
        emm_d["predicted_gap"],
        yerr=[emm_d["predicted_gap"] - emm_d["lower.CL"], emm_d["upper.CL"] - emm_d["predicted_gap"]],
        fmt="none",
        ecolor="#B2182B",
        elinewidth=0.85,
        capsize=3,
        capthick=0.85,
    )
    ax.scatter(
        emm_d["encoding_auc"],
        emm_d["predicted_gap"],
        s=42,
        marker="o",
        facecolors="#B2182B",
        edgecolors="white",
        linewidths=0.8,
    )
    for _, row in emm_d.iterrows():
        ax.text(
            row["encoding_auc"],
            row["predicted_gap"] + y_range[1] * 0.035,
            f"{row['predicted_gap']:.3f}",
            color="#B2182B",
            fontsize=7.2,
            fontweight="bold",
            ha="center",
            va="bottom",
        )

    if peak_in_data:
        ax.scatter([float(res["peak_auc"])], [peak_y], marker="v", s=40,
                   facecolors="#333333", edgecolors="#333333")

    ax.set_title(label_demo(attr), fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(x_rng)
    ax.set_ylim(y_range)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel("Encoding AUC", fontsize=9.5)
    ax.set_ylabel("FPR disparity gap", fontsize=9.5)
    ax.tick_params(axis="both", labelsize=8.5)
    panel_border(ax)

    if show_legend:
        handles = [
            Line2D([0], [0], marker=MODEL_MARKERS[m], color="none", label=m,
                   markerfacecolor=MODEL_COLORS[m], markeredgecolor="#1a1a1a",
                   markeredgewidth=0.3, markersize=5)
            for m in MODEL_COLORS.keys()
        ]
        ax.legend(handles=handles, title="Model", frameon=False, fontsize=7.5,
                  title_fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5))


def make_layer_figure(target_lr: float, layer_lab: str) -> None:
    attrs_to_plot = list(results["demographic"].dropna().unique())

    fig, axes = plt.subplots(
        1,
        len(attrs_to_plot),
        figsize=(5 * len(attrs_to_plot), 5.5),
        sharey=True,
    )

    if len(attrs_to_plot) == 1:
        axes = [axes]

    for i, attr in enumerate(attrs_to_plot):
        make_layer_panel(
            axes[i],
            attr,
            target_lr,
            layer_lab,
            show_legend=(i == len(attrs_to_plot) - 1),
        )
    fig.suptitle(
        f"Layer depth: {layer_lab} - Encoding strength and FPR disparity across VLM layers",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0.02, 0.94, 0.95])

    fname = "07_figure_layer_" + str(target_lr).replace(".", "")
    savefig(OUT / f"{fname}.png", fig)
    savefig(OUT / f"{fname}.pdf", fig)
    plt.close(fig)
    log("Saved:", fname)


# Save all three
make_layer_figure(0.1, "Early layers")
make_layer_figure(0.5, "Middle layers")
make_layer_figure(1.0, "Late layers")


# Check direction - which group has higher FPR on average
direction_rows = []
for attr, (c1, c2) in DEMO_PAIRS.items():
    dis_frames = []
    for m in MODEL_REGISTRY:
        try:
            dis = pd.read_csv(m.dis_csv)
            dis = dis.loc[(dis["disease"] == "No Finding") & (dis["condition"].isin([c1, c2])), ["condition", "fpr"]].copy()
            dis_frames.append(dis)
        except Exception:
            continue
    dis_all = pd.concat(dis_frames, ignore_index=True) if dis_frames else pd.DataFrame(columns=["condition", "fpr"])
    means = dis_all.groupby("condition", as_index=False)["fpr"].mean()
    means["mean_fpr"] = means["fpr"].round(4)
    m1 = means.loc[means["condition"] == c1, "mean_fpr"]
    m2 = means.loc[means["condition"] == c2, "mean_fpr"]
    mean_fpr_g1 = float(m1.iloc[0]) if len(m1) else np.nan
    mean_fpr_g2 = float(m2.iloc[0]) if len(m2) else np.nan
    higher = c1 if mean_fpr_g1 > mean_fpr_g2 else c2
    direction_rows.append(
        {
            "demographic": attr,
            "group_1": c1,
            "group_2": c2,
            "mean_fpr_g1": mean_fpr_g1,
            "mean_fpr_g2": mean_fpr_g2,
            "higher_fpr": higher,
        }
    )
direction_check = pd.DataFrame(direction_rows)
log(direction_check.to_string(index=False))
direction_check.to_csv(OUT / "direction_check.csv", index=False)


# -- Direction check across all layers and models ------------------------------
direction_by_layer_rows = []
for m in MODEL_REGISTRY:
    try:
        dis = pd.read_csv(m.dis_csv)
        dis["layer_num"] = apply_layer_fn(dis, m.layer_fn)
        dis = dis.loc[dis["layer_num"].notna() & (dis["disease"] == "No Finding")].copy()
    except Exception:
        continue

    for attr, (c1, c2) in DEMO_PAIRS.items():
        sub = dis.loc[dis["condition"].isin([c1, c2]), ["layer_num", "condition", "fpr"]].copy()
        wide = sub.pivot_table(index="layer_num", columns="condition", values="fpr", aggfunc="first").reset_index()
        if c1 not in wide.columns or c2 not in wide.columns:
            continue
        wide = wide.loc[wide[c1].notna() & wide[c2].notna()].copy()
        wide["model"] = m.name
        wide["attribute"] = attr
        wide["fpr_g1"] = wide[c1]
        wide["fpr_g2"] = wide[c2]
        wide["gap"] = wide[c1] - wide[c2]  # signed - not abs
        wide["direction"] = np.where(
            wide["gap"] > 0,
            c1 + " higher",
            np.where(wide["gap"] < 0, c2 + " higher", "equal"),
        )
        wide["layer_rel"] = wide["layer_num"] / float(m.final_num)
        direction_by_layer_rows.append(
            wide[["model", "attribute", "layer_num", "layer_rel", "fpr_g1", "fpr_g2", "gap", "direction"]]
        )

direction_by_layer = (
    pd.concat(direction_by_layer_rows, ignore_index=True)
    if direction_by_layer_rows
    else pd.DataFrame(columns=["model", "attribute", "layer_num", "layer_rel", "fpr_g1", "fpr_g2", "gap", "direction"])
)

# -- Summary: how often does direction flip? -----------------------------------
summary_rows = []
for attr, g in direction_by_layer.groupby("attribute"):
    n_obs = len(g)
    pct_g1_higher = round(100 * float((g["gap"] > 0).mean()), 1) if n_obs else np.nan
    pct_g2_higher = round(100 * float((g["gap"] < 0).mean()), 1) if n_obs else np.nan
    group_1, group_2 = DEMO_PAIRS[attr]
    if pct_g1_higher > 90:
        verdict = f"{group_1} consistently higher"
    elif pct_g2_higher > 90:
        verdict = f"{group_2} consistently higher"
    else:
        verdict = "direction flips - interpret with caution"
    summary_rows.append(
        {
            "attribute": attr,
            "n_obs": n_obs,
            "n_g1_higher": int((g["gap"] > 0).sum()),
            "n_g2_higher": int((g["gap"] < 0).sum()),
            "n_equal": int((g["gap"] == 0).sum()),
            "pct_g1_higher": pct_g1_higher,
            "pct_g2_higher": pct_g2_higher,
            "direction_consistent": bool((pct_g1_higher > 90) or (pct_g2_higher > 90)),
            "group_1": group_1,
            "group_2": group_2,
            "verdict": verdict,
        }
    )
direction_summary = pd.DataFrame(summary_rows)

log("\n== DIRECTION CONSISTENCY ==")
log(direction_summary.to_string(index=False))
direction_summary.to_csv(OUT / "direction_summary.csv", index=False)

# -- By model - does any specific model flip direction? ------------------------
model_rows = []
for (attr, model), g in direction_by_layer.groupby(["attribute", "model"]):
    n_layers = len(g)
    pct_g1_higher = round(100 * float((g["gap"] > 0).mean()), 1) if n_layers else np.nan
    pct_g2_higher = round(100 * float((g["gap"] < 0).mean()), 1) if n_layers else np.nan
    mean_gap = round(float(g["gap"].mean()), 4) if n_layers else np.nan
    min_gap = round(float(g["gap"].min()), 4) if n_layers else np.nan
    max_gap = round(float(g["gap"].max()), 4) if n_layers else np.nan
    mean_sign = np.sign(g["gap"].mean()) if n_layers else np.nan
    flips = int((np.sign(g["gap"]) != mean_sign).sum()) if n_layers else 0
    if pct_g1_higher > 90:
        verdict = "consistent"
    elif pct_g2_higher > 90:
        verdict = "consistent - opposite direction"
    else:
        verdict = "flips within model"
    model_rows.append(
        {
            "attribute": attr,
            "model": model,
            "n_layers": n_layers,
            "pct_g1_higher": pct_g1_higher,
            "pct_g2_higher": pct_g2_higher,
            "mean_gap": mean_gap,
            "min_gap": min_gap,
            "max_gap": max_gap,
            "flips": flips,
            "verdict": verdict,
        }
    )
direction_by_model = pd.DataFrame(model_rows)

log("\n== DIRECTION BY MODEL ==")
log(direction_by_model.to_string(index=False))
direction_by_model.to_csv(OUT / "direction_by_model.csv", index=False)

# -- Visual: signed gap across layer depth -------------------------------------
attrs_to_plot = list(direction_by_layer["attribute"].dropna().unique())

fig, axes = plt.subplots(
    1,
    len(attrs_to_plot),
    figsize=(4 * len(attrs_to_plot), 4),
    sharey=False,
)

if len(attrs_to_plot) == 1:
    axes = [axes]

for ax, attr in zip(axes, attrs_to_plot):
    gd = direction_by_layer.loc[direction_by_layer["attribute"] == attr].copy()
    for model, g in gd.groupby("model"):
        g = g.sort_values("layer_rel")
        ax.plot(g["layer_rel"], g["gap"], color=MODEL_COLORS.get(model, "gray"),
                alpha=0.6, linewidth=0.7, label=model)
    ax.axhline(0, linestyle="--", color="#333333", linewidth=0.8)
    add_strip(ax, label_demo(attr))
    theme_classic_axes(ax)
    ax.set_xlabel("Relative layer depth (0 = earliest, 1 = final)")
    ax.set_ylabel("Signed FPR gap (group 1 minus group 2)" if ax is axes[0] else "")
fig.suptitle("Signed FPR gap across layer depth", fontsize=12, y=1.06)
fig.text(0.5, 0.98,
         "Above zero = group 1 has higher FPR | Below zero = group 2 has higher FPR | Dashed = no gap",
         ha="center", fontsize=10)
handles = [Line2D([0], [0], color=MODEL_COLORS[m], linewidth=1, label=m) for m in MODEL_COLORS.keys()]
fig.legend(handles=handles, loc="center right", frameon=False)
fig.tight_layout(rect=[0, 0, 0.88, 0.94])
savefig(OUT / "direction_across_layers.png", fig)
plt.close(fig)
log("Saved: direction_across_layers.png")


if __name__ == "__main__":
    # The analysis runs top-to-bottom to mirror the original R script.
    # This block is intentionally minimal because all relevant print/log statements
    # are placed inline at the same analysis stages as in the original code.
    pass
