# Understanding and Mitigating Gender Bias in Stable Diffusion 3: An Analysis of Embedding-Level Intervention

This repository contains the code developed for the project "Understanding and Mitigating Gender Bias in Stable Diffusion 3: An Analysis of Embedding-Level Intervention", a UCL BSc Computer Science dissertation. 

## Contents
- [Content Overview](#context-overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Data Notes](#data-notes)


## Context Overview

Gender bias in text-to-image (T2I) models manifests when a neutral occupational prompt (e.g. "a photo of a software engineer") consistently generates images skewed toward one gender. 

This project addresses that bias at the embedding level, analysing gender bias across different stages of the Stable Diffusion 3 (SD3) pipeline from the prompt embeddings, generated images, and the automated tools used to label gender in generated outputs. We develop **FairEmbed**, a surrogate-assisted Genetic Algorithm that optimise embedding-level interventions to mitigate gender bias in SD3.

---

## Repository Structure

This repository is organised into four folders containing the code for the three main stages of the project:

- **Stage 1 Empirical analysis of automated labelling tools (RQ1)**
- **Stage 1 Empirical analysis of embeddings (RQ2)**
- **Stage 2 Surrogate modelling (RQ3)**
- **Stage 3 Bias Mitigation (RQ4)**



```text
FINAL-YEAR_PROJECT/
├── 1_Empirical_Analysis_Automated_Labelling_Tools/
│   ├── model_helpers/
│   ├── labels/
│   ├── analysis_results/
│   ├── gender_stats/
│   ├── labelling_tools_analysis.ipynb
│   └── calc_gender_stats_tools.ipynb
│ 
├── 1_Empirical_Analysis_Embeddings/
│   ├── embeddings/
│   ├── generated_images/
│   ├── ground_truth/
│   └── embeddings_analysis.ipynb
│ 
├── 2_Surrogate_Modelling/
│   ├── data/
│   ├── model/
│   ├── results/
│   └── analyse_model.ipynb
│ 
├── 3_Bias_Mitigation/
│   ├── moea/
│   ├── exp1_results/
│   ├── exp2_results/
│   └── exp2_scripts/
│ 
├── requirements.txt
└── README.md
```

### Folder/File Descriptions

#### `1_Empirical_Analysis_Automated_Labelling_Tools/`
Contains the implementations and evaluation outputs for automated gender-labelling tool experiments, including CLIP (its variants), BLIP-2, MiVOLO, and FairFace (RQ2)

- `model_helpers/` – helper scripts to collect labels from tools
    > **Note:**  For CLIP-enhance, FairFace, MiVOLO please refer to the original authors package repositories in order to use their tools. For Fairface, `fairface_paths.csv` can be used when running FairFace on dataset.
- `labels/` – raw and cleaned label outputs, along with converter scripts used to clean them
- `analysis_results/` – results from the `labelling_tools_analysis` notebook
- `gender_stats/` – per-category gender distribution statistics for each tool
- `labelling_tools_analysis.ipynb`-  helps compute the gender stats, which are stored in `gender_stats/`
- `calc_gender_stats_tools.ipynb` - contains code for the plots and tables for the automated labelling tool empirical analysis results

#### `1_Empirical_Analysis_Embeddings/`
Contains the code and results for the embedding-based analysis of gender bias (RQ1)

- `embeddings/` – raw embeddings, embeddings extraction scripts and cosine-similarity bias scores
    > **Note:** `LATEST_rp_updated_embeddings.pkl` contains main prompt embeddings, and `LATEST_extra_data_embeddings.pkl` is for the supplementary data.
- `generated_images/` – labels for generated images, with bias analysis and scores for output images
- `ground_truth/` – ground-truth label analysis and gender stats
- `embeddings_analysis.ipynb` - contains code for the plots used in the report for the embedding analysis results 

#### `2_Surrogate_Modelling/`
Contains the data preparation and surrogate modelling pipeline code for the surrogate model experiments (RQ3)

- `data/` – final merged dataset and preprocessing outputs, along with their scripts
    > **Note:**  `LATEST_all_data_merged_for_regression` is the final dataset used to train models. `LATEST_all_embeddings_by_category` only includes embedding data, `LATEST_merged_for_regression` enriches these with cosine similarities (there are seperate files for the supplementary dataset for embedding data and cosine similarities)

- `model/` – model training and fitting scripts that evaluate different modelling families, and train the final model
    > **Note:**  `evaluate_nested_cv_shufflesplit.py` is for stage 1 of the modelling pipeline described in the report, `train_all_models.py` is used for stage 2, and `train_final_model.py` trains XGBoost for the GA. 

- `results/` - includes trained models, and other statistics gathered during training
- `analyse_model.ipynb` - contains code to decide which modelling family is best for this problem

#### `3_Bias_Mitigation/`
Contains the results and implementation of FairEmbed (RQ4)

- `moea/` – FairEmbed algorithm implementation
- `exp1_results/` – results from 50-run configuration search experiments
- `exp2_results/` – results from generalisation experiments on unseen prompts
- `exp2_scripts/` – scripts used for generalisation experiments on unseen prompts
> **Note:**  There is a seperate README provided for FairEmbed in `moea/`.

---

## Setup

```bash
pip install -r requirements.txt
```

**Hardware**: Surrogate-based GA runs on CPU. Embedding extraction and image generation are better run on a CUDA-capable GPU.

---

## Data Notes
> `extra_data.csv` — supplementary occupational category data; reference: 
> SD3 images from Lyu et al. are referenced in the report but are not included in this repository. Please see their repository package here: 
> Raw label outputs per tool are in `labels/`; aligned/cleaned versions are in `labels/cleaned_results/`



