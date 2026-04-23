# Understanding and Mitigating Gender Bias in Stable Diffusion 3: An Analysis of Embedding-Level Intervention

This repository contains the code developed for the project "Understanding and Mitigating Gender Bias in Stable Diffusion 3: An Analysis of Embedding-Level Intervention", a UCL BSc Computer Science dissertation. 

## Contents
- [Content Overview](#context-overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Acknowledgements](#acknowledgements)


## Context Overview

Gender bias in text-to-image (T2I) models manifests when a neutral occupational prompt (e.g. "a photo of a software engineer") consistently generates images skewed toward one gender. 

This project addresses that bias at the embedding level, analysing gender bias across different stages of the Stable Diffusion 3 (SD3) pipeline from the prompt embeddings, generated images, and the automated tools used to label gender in generated outputs. We develop **FairEmbed**, a surrogate-assisted Genetic Algorithm that optimise embedding-level interventions to mitigate gender bias in SD3.

---

## Repository Structure

This repository is organised into four folders containing the code for the three main stages of the project:

- **Stage 1 Empirical analysis**
    - Prompt Embedding-Level Gender Bias (RQ1)
    - Influence of Automated Labelling Tools on Measured Bias (RQ2)
- **Stage 2 Surrogate Modelling (to predict bias)** (RQ3)
- **Stage 3 Bias Mitigation (GA Optimisation)** (RQ4)



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
    > **Note:** The labelling tool experiments in this project 
    > reproduce and extend the methodology of Lyu et al. (2025) 
    > ("Do Existing Testing Tools Really Uncover Gender Bias in 
    > Text-to-Image Models?", https://arxiv.org/abs/2501.15775).
    > For CLIP-Enhance, FairFace, and MiVOLO, please refer to 
    > the original authors' repositories to set up their tools
    > and for further instructions:
    > - FairFace: https://github.com/joojs/fairface
    > - MiVOLO: https://github.com/WildChlamydia/MiVOLO (Note: we use this [model](https://drive.google.com/file/d/11i8pKctxz3wVkDBlWKvhYIh7kpVFXSZ4/view) trained on the IMDB-cleaned dataset.)
    > - CLIP-Enhance: https://arxiv.org/abs/2501.15775 (the provided `clip_enhance.py` is named as `main.py` in the authors repository)
    >
    > `fairface_paths.csv` can be used when running FairFace 
    > on the dataset.
    > `runmivolo.py` can be used when running MiVOLO on the dataset.
    
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
## Acknowledgements
### Data Sources

- `extra_data.csv` contains supplementary data extracted from 
  d'Aloisio et al. (2025). The original dataset is available at:
  https://github.com/giordanoDaloisio/image-generation-bias
  Paper: https://arxiv.org/pdf/2501.09014

- The SD3 images (`sd3_label_image.zip`) forming the main dataset 
are from Lyu et al. (2025). 
Paper: https://arxiv.org/abs/2501.15775

### Reused and Adapted Code
The following files were originally developed as part of 
Lyu et al. (2025), "Do Existing Testing Tools Really 
Uncover Gender Bias in Text-to-Image Models?"  
(https://doi.org/10.1145/3746027.3755748).  
These files were obtained from the authors' replication package,
and has been reused or adapted in this project:

- `clip_enhance.py` — reused from the original implementation (as `main.py` in `clip_enhance` folder their code)
- `clip_prob.py` — adapted from the original implementation (we track skipped images)
- `clip_uncertain.py` — adapted from the original implementation (we track skipped images)
- `clip.py` — reused from the original implementation



