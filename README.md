# Understanding and Mitigating Gender Bias in Stable Diffusion 3: An Analysis of Embedding-Level Intervention

This repository contains the code developed for the project "Understanding and Mitigating Gender Bias in Stable Diffusion 3: An Analysis of Embedding-Level Intervention", a UCL BSc Computer Science dissertation. 

## Contents
- [Content Overview](#context-overview)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Reproducing Results](#reproducing-results)
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


## Reproducing Results

Results in the report can be reproduced by running the 
following files in order. Each stage depends on outputs 
from the previous one.

### Stage 1: Empirical Analysis (Report Section 4.1)

**RQ1 – Embedding bias analysis:**
1. Extract embeddings: run scripts in 
   `1_Empirical_Analysis_Embeddings/embeddings/collecting_emb/`
2. Compute cosine similaritiy scores between gender variants of prompts:
   run `bias_calc_embeddings.py`
3. Label generated images, and compute gender statisitics for them:
   run `blip_labeller.py` then `bias_calc_generated_img.ipynb`
4. Compute gender statisitics for ground truh (manual labels):
   run `bias_calc_ground_truth.ipynb`
5. Generate RQ1 results (Figures 4.1–4.3, Table 4.1): 
   run `embeddings_analysis.ipynb`

**RQ2 – Labelling tool comparison:**
1. Follow the notes above and set up FairFace, MiVOLO, Clip-Enhance according to the authors' instructions.
2. Run each labelling tool using the helper scripts in 
   `1_Empirical_Analysis_Automated_Labelling_Tools/model_helpers/`
   (Ensure the path to the images is correct. Refer again to the authors' instructions on how to use each tool.)
4. Clean outputs from each tool using the converter scripts in `labels/`
5. Compute gender stats for each tool, changing the CSV path to the cleaned results obtained in step 4:
   run `labelling_tools_analysis.ipynb` (Instructions in notebook)
7. Generate RQ2 results (Figure 4.4, Tables 4.2–4.4, Tables A.4-A.6 in Appendix): 
   run `calc_gender_stats_tools.ipynb`

### Stage 2: Surrogate Modelling (Report Section 4.2) - RQ3

1. Prepare dataset: run scripts in `2_Surrogate_Modelling/data/`
    run `extract_emb_by_category.py`, -> `create_extradata_dataset.py` -> `create_final_dataset.py`
3. Model family comparison - Stage 1 of Surrogate Modelling Pipeline (Table 4.5): 
   run `evaluate_nested_cv_shufflesplit.py` 
4. Full model training - Stage 2 of Surrogate Modelling Pipeline (Table 4.5): run `train_all_models.py`
5. Train final XGBoost for GA - Stage 3 of Surrogate Modelling Pipeline: run `train_final_model.py`
6. Generate RQ3 results (Figure 4.5, Table 4.5, Table A.7): 
   run `analyse_model.ipynb`

### Stage 3: Bias Mitigation (Report Section 4.3) - RQ4

1. Run FairEmbed: see separate README in `3_Bias_Mitigation/moea/`
   run `runs.sh` to help complete the 50 runs
   Note: Change arguments to `main.py` to test each configuration listed in table 3.3)
3. RQ4 Experiment 1 results (Statistical tests in Section 4.3.1, Figure 4.6, Table A.8-A.11): in `exp1_results/`
    run `solution_helper.py` -> `50_runs_helper.py` with the relevant paths to the results you're trying to recreate
    then use `50_runs_analysis.pynb` 
4. RQ4 Experiment 2 generalisation (Table 4.6, Figure 4.7, Table A.12-A.13): 
   run `other_prompts.py` then use `50_runs_analysis.pynb` 
5. RQ4 Comparison with existing methods analysis (Results in Section 4.3.2, Table 4.7)
   run `runtime_analysis.py` and use `image_generation_analysis.pynb` 
   Note: You must first follow steps 1-2 using the parameters specified in the report.
   
---
## Acknowledgements
### Data Sources

- `extra_data.csv` contains supplementary data extracted from 
  d'Aloisio et al. (2025). The original dataset is available at:
  https://github.com/giordanoDaloisio/image-generation-bias
  Paper: https://arxiv.org/pdf/2501.09014

- The SD3 images (`sd3_label_image.zip`) forming the main dataset 
are from Lyu et al. (2025) licensed under CC BY 4.0.  
Paper: https://doi.org/10.1145/3746027.3755748  
Replication package: https://figshare.com/articles/software/T2IReplication-ISSTA25/27377649/1


### Reused and Adapted Code
The following files were originally developed as part of 
Lyu et al. (2025), "Do Existing Testing Tools Really 
Uncover Gender Bias in Text-to-Image Models?" (https://doi.org/10.1145/3746027.3755748).  
Replication package: https://figshare.com/articles/software/T2IReplication-ISSTA25/27377649/1

These files were obtained from the authors' replication package,
and has been reused or adapted in this project:

- `clip_enhance.py` — reused from the original implementation (as `main.py` in `clip_enhance` folder their code)
- `clip_prob.py` — adapted from the original implementation (we track skipped images)
- `clip_uncertain.py` — adapted from the original implementation (we track skipped images)
- `clip.py` — reused from the original implementation



