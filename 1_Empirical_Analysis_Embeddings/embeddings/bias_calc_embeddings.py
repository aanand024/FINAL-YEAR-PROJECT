import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import json, re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import argparse

"""
This script computes cosine similarity between the male, female, and neutral variants of prompt embeddings
to quantify gender bias in generated images. It the calculates cosine similarities, and outputs a CSV file
with our bias metrics (difference and ratio).

Usage:
    python bias_calc_embeddings.py
        [--input PATH_TO_EMBEDDINGS]
        [--output1 OUTPUT_CSV_1]
        [--output2 OUTPUT_CSV_2]

To process a different dataset (e.g., extra data), provide the appropriate input and output paths as arguments:
    python bias_calc_embeddings.py --input 1_Empirical_Analysis/embeddings/raw_data/LATEST_extra_data_embeddings.pkl \
        --output1 LATEST_extra_data_emb_cosine_simil.csv \
        --output2 LATEST_extra_data_emb_cosine_simil_2.csv
"""
# Load data 
parser = argparse.ArgumentParser(description="Compute cosine similarity for bias analysis.")
parser.add_argument('--input', type=str, default=os.path.join('1_Empirical_Analysis', 'embeddings', 'raw_data', 'LATEST_rp_updated_embeddings.pkl'),
                    help='Path to the input embeddings pickle file.')
parser.add_argument('--output1', type=str, default=os.path.join('1_Empirical_Analysis', 'LATEST_rp_updated_emb_cosine_simil.csv'),
                    help='Path to the first output CSV file.')
parser.add_argument('--output2', type=str, default=os.path.join('1_Empirical_Analysis', 'LATEST_rp_updated_emb_cosine_simil_2.csv'),
                    help='Path to the second output CSV file.')
args = parser.parse_args()

path = args.input
output_path1 = args.output1
output_path2 = args.output2

try:
    emb_results = pd.read_pickle(path)
except Exception:
    with open(path, 'rb') as f:
        emb_results = pickle.load(f)

if not isinstance(emb_results, pd.DataFrame):
    try:
        emb_results = pd.DataFrame(emb_results)
    except Exception:
        pass

if isinstance(emb_results['Embedding1'].iloc[0], str):
    emb_results['Embedding1'] = emb_results['Embedding1'].apply(lambda x: np.array(eval(x)))


# Pivot the data: one row per category with columns for each gender
pivot = emb_results.pivot(index='Category', columns='Gender', values='Embedding1')

similarities = []
for base_prompt in pivot.index:
    neutral_emb = pivot.loc[base_prompt, 'Neutral']
    male_emb = pivot.loc[base_prompt, 'Male']
    female_emb = pivot.loc[base_prompt, 'Female']
    
    # Flatten embeddings if multi-dim
    neutral_flat = neutral_emb.flatten().reshape(1, -1)
    male_flat = male_emb.flatten().reshape(1, -1)
    female_flat = female_emb.flatten().reshape(1, -1)
    
    simil_neutral_male = cosine_similarity(neutral_flat, male_flat)[0, 0]
    simil_neutral_female = cosine_similarity(neutral_flat, female_flat)[0, 0]
    
    similarities.append({
        'Category': base_prompt,
        'cosine_sim_neutral_male': simil_neutral_male,
        'cosine_sim_neutral_female': simil_neutral_female,
        'difference': simil_neutral_male - simil_neutral_female,
        'ratio': simil_neutral_male / simil_neutral_female if simil_neutral_female != 0 else None

    })

simil_df = pd.DataFrame(similarities)


simil_df.to_csv(output_path1, index=False)

pivot = emb_results.pivot(index='Category', columns='Gender', values='Embedding2')

similarities = []
for base_prompt in pivot.index:
    neutral_emb = pivot.loc[base_prompt, 'Neutral']
    male_emb = pivot.loc[base_prompt, 'Male']
    female_emb = pivot.loc[base_prompt, 'Female']
    
    # Flatten embeddings if multi-dim
    neutral_flat = neutral_emb.flatten().reshape(1, -1)
    male_flat = male_emb.flatten().reshape(1, -1)
    female_flat = female_emb.flatten().reshape(1, -1)
    
    simil_neutral_male = cosine_similarity(neutral_flat, male_flat)[0, 0]
    simil_neutral_female = cosine_similarity(neutral_flat, female_flat)[0, 0]
    
    similarities.append({
        'Category': base_prompt,
        'cosine_sim_neutral_male': simil_neutral_male,
        'cosine_sim_neutral_female': simil_neutral_female,
        'difference': simil_neutral_male - simil_neutral_female,
        'ratio': simil_neutral_male / simil_neutral_female if simil_neutral_female != 0 else None

    })
simil_df = pd.DataFrame(similarities)

simil_df.to_csv(output_path2, index=False)