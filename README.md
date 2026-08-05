# Demographic Bias and Encoding in Medical Vision-Language Models

Code for the project:

> **Demographic Bias and Encoding in Medical Vision-Language Models**
> Ammar Ahmed Pallikonda Latheef, Paravreet Woodwal, Aditya Kulkarni, Jacob M. Luber,
> Pritam Mukherjee\*, Paul H. Yi\*†
> St. Jude Children's Research Hospital
> \*Co-Senior Authors · †Corresponding Author

This repository contains the code used to extract layer-wise vision/text embeddings from
eight medical vision-language models (VLMs), train linear probes for demographic encoding
(sex, age, race/ethnicity) and disease classification (with emphasis on "No Finding"), and
evaluate demographic underdiagnosis disparity across three chest x-ray datasets:
MIMIC-CXR, CheXpert, and ReXGradient-160K.

## Models evaluated

| Model | Type | Source |
|---|---|---|
| CheXzero | CXR-specific | Tiu et al., *Nat. Biomed. Eng.* 2022 |
| CheXagent | CXR-specific | [Stanford-AIMI/CheXagent](https://github.com/Stanford-AIMI/CheXagent) |
| NV-Reason-CXR | CXR-specific | [NVIDIA-Medtech/NV-Reason-CXR](https://github.com/NVIDIA-Medtech/NV-Reason-CXR) |
| BioViL-T | CXR-specific | Bannur et al., CVPR 2023 (Microsoft `hi-ml`/`health_multimodal`) |
| MedGemma 1.5 | Generalist | Sellergren et al., 2026 (Google) |
| RadFM | Generalist | [chaoyi-wu/RadFM](https://github.com/chaoyi-wu/RadFM) |
| LLaVA-Med 1.5 | Generalist | [microsoft/LLaVA-Med](https://github.com/microsoft/LLaVA-Med) |
| MedVersa | Generalist | Zhou et al., *NEJM AI* 2026 |

This repository does **not** vendor any of the above model repositories or their weights.
To reproduce feature extraction, clone/download each model from its official source above
(see full citation list in the manuscript) and point the paths in each model's `config_*.py`
to your local copy. `extraction/chexzero/model.py` and `extraction/chexzero/chexzero_modules/`
are small (MIT-licensed) architecture/tokenizer files derived from OpenAI CLIP via the
CheXzero codebase and are included because the extraction script imports them directly.

## Datasets

| Dataset | Access |
|---|---|
| MIMIC-CXR | PhysioNet (credentialed access) — Johnson et al., *Sci. Data* 2019 |
| CheXpert | Stanford AIMI (registration required) — Irvin et al., *AAAI* 2019 |
| ReXGradient-160K | Zhang et al., 2025 (see paper for access instructions) |

No raw images, reports, metadata CSVs, or model weights are included in this repository.
Demographic/disease labels were derived using the CheXpert-style NLP labeler. Analyses were
restricted to adult patients (age ≥ 18).

**Age-code convention used in our probes:**

| Dataset | Code 0 | Code 1 | Code 2 | Code 3 | Code 4 |
|---|---|---|---|---|---|
| MIMIC-CXR / CheXpert | 80+ | 60–79 | 40–59 | 18–39 | — |
| ReXGradient-160K | 0–17 | 18–39 | 40–59 | 60–79 | 80+ |

MIMIC-CXR/CheXpert age codes are in **descending** order; ReXGradient-160K is **ascending**.
The primary age contrast used throughout is 80+ vs. 18–39.

## Repository structure

```
extraction/                  Layer-wise vision/text embedding extraction (frozen models)
  chexzero/ biovilt/ radfm/ llavamed1p5/ nv_reason/ chexagent/ medversa/ medgemma_1p5/
      config_<model>.py            model/dataset paths, layer names
      <model>_extract_image_layers.py
      <model>_extract_text_layers.py
  rexgradient_image_utils.py       16-bit PNG -> 8-bit RGB fix used by all image
                                    extraction scripts when run on ReXGradient-160K
  orchestration/                   dataset-level shell drivers
      run_image_layer_extraction_mimic.sh
      run_text_layer_extraction_mimic.sh
      run_chexpert_extraction_all_models.sh
      run_rexgradient_fixed_final_pipeline.sh

probe_training/               Linear probe training (sex, age, ethnicity, No Finding, diseases)
  universal/                       shared probe-training scripts used by chexzero,
                                    radfm, nv_reason, and medversa
      universal_train_demographic_probe.py
      universal_train_disease_probe.py
  mimic/<model>/                   per-model training scripts/pipelines (MIMIC-CXR)
  chexpert/train_chexpert_probes_all_models.py
  rexgradient/                     vision (& text, via --modalities/shell flag) training
      train_rexgradient_nonpediatric_all_layers.py
      run_rexgradient_nonpediatric_all_layers.sh
      run_rexgradient_nonpediatric_text_train.sh

evaluation/                   Held-out test-set evaluation -> JSON results
  mimic/06_evaluate_demographics_full_test.py     sex/age/ethnicity AUC (+ bootstrap CI)
  mimic/07_evaluate_diseases_full_test.py         per-disease AUC/FPR incl. No Finding,
                                                   conditioned on demographics
  chexpert/evaluate_chexpert_full_test.py         demographics + No Finding
  rexgradient/evaluate_rexgradient_full_test_nonpediatric.py

analysis/                     Layer-wise linear mixed model (LMM) analysis
  build_lmm_csvs_from_jsons.py     converts evaluation JSONs -> per-layer CSVs
  linear_mixed_model_analysis.py  quadratic LMM (encoding AUC -> FPR gap), AR(1), emmeans
  run_lmm_all_datasets.py         orchestrates the above across MIMIC/CheXpert/ReXGradient
```

## Pipeline overview

1. **Extraction** — For each of the 8 models, extract frozen layer-wise vision embeddings
   (from chest x-rays) and text embeddings (from report impressions) using each model's
   native preprocessing/tokenization and pooling rule. Run per-dataset via the scripts in
   `extraction/orchestration/`.
2. **Probe training** — Train independent L2-logistic-regression probes per
   model/dataset/modality/layer/target (`probe_training/`). Demographic probes: sex
   (female vs. male), age (80+ vs. 18–39), race/ethnicity (Black vs. White, MIMIC/CheXpert
   only). Disease probes: CheXpert-style multi-label disease classification, with "No
   Finding" used for underdiagnosis disparity. Hyperparameters and F1-optimal thresholds
   are selected on the validation split only.
3. **Evaluation** — Apply validation-selected thresholds to the held-out test split and
   compute AUC (demographic encoding) and FPR gaps between demographic subgroups
   (underdiagnosis disparity) for each model/modality/layer (`evaluation/`).
4. **Analysis** — Aggregate per-layer evaluation JSONs and fit the quadratic linear mixed
   model relating vision-layer demographic encoding AUC to No Finding FPR gap, with model
   identity as a random effect and an AR(1) correlation across ordered layers
   (`analysis/`). This step requires R (`nlme`, `emmeans`) via `rpy2`.

## Output locations (original experiment layout)

```
other_models/<model>/probe_experiment_outputs/<dataset>/
  trained_probes_vision_only/            MIMIC/CheXpert demographic probes (vision)
  trained_probes_text_only/              MIMIC/CheXpert demographic probes (text)
  trained_probes_vision_only_nonpediatric/  ReXGradient demographic probes (vision, adults)
  trained_probes_text_only_nonpediatric/    ReXGradient demographic probes (text, final layer)
  trained_probes_image_diseases/         MIMIC multi-label disease probes (vision)
  trained_probes_text_diseases/          MIMIC multi-label disease probes (text)

other_models/evaluation_results_optimized_thresholds/<model>/   MIMIC-CXR
  evaluation_results_vision_full_test/
  evaluation_results_text_full_test/
  evaluation_results_vision_disease_full_test/

other_models/evaluation_results/<model>/                        CheXpert & ReXGradient
  evaluation_results_vision_full_test_chexpert/
  evaluation_results_text_full_test_chexpert/
  evaluation_results_vision_nofinding_full_test_chexpert/
  evaluation_results_vision_full_test_rexgradient_nonpediatric/
  evaluation_results_text_full_test_rexgradient_nonpediatric/
  evaluation_results_vision_nofinding_full_test_rexgradient_nonpediatric/
```

## Environment

Each model has substantially different dependencies (e.g., `torch`, `transformers`,
`health_multimodal`/`hi-ml`, `safetensors`) and, for RadFM/MedVersa/LLaVA-Med, requires the
official upstream repository on `sys.path` (see `config_*.py` / extraction scripts for the
expected local paths). Probe training/evaluation additionally require `scikit-learn`,
`pandas`, `numpy`, and `joblib`. The LMM analysis (`analysis/linear_mixed_model_analysis.py`)
requires R with the `nlme`, `emmeans`, and `mgcv` packages, accessed from Python via `rpy2`.
We recommend a separate conda environment per model, following that model's own
installation instructions, plus one shared environment for probing/evaluation/analysis.

## License

Code in this repository is released under the MIT License (see `LICENSE`), except where a
file retains an upstream license header (e.g., `extraction/chexzero/model.py` and
`extraction/chexzero/chexzero_modules/`, which are MIT-licensed OpenAI CLIP-derived code
distributed with CheXzero). Third-party models and datasets referenced above are governed
by their own respective licenses/data-use agreements.

## Citation

If you use this code, please cite the manuscript above. A publication DOI/citation entry
will be added here upon publication.
