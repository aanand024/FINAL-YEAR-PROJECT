'''
This script is used to extract and save embeddings by category.

For the extra data, use: 
INPUT = '1_Empirical_Analysis_Embeddings/embeddings/raw_data/LATEST_extra_data_embeddings.pkl'
OUTPUT = 'LATEST_extra_data_all_embeddings_by_category.csv'
'''

import pandas as pd
import numpy as np
import json
import pickle


INPUT = '1_Empirical_Analysis_Embeddings/embeddings/raw_data/LATEST_rp_updated_embeddings.pkl'
OUTPUT = 'LATEST_all_embeddings_by_category.csv'

try:
    emb_results = pd.read_pickle(INPUT)
except Exception:
    with open(INPUT, 'rb') as f:
        emb_results = pickle.load(f)

if not isinstance(emb_results, pd.DataFrame):
    emb_results = pd.DataFrame(emb_results)

# Convert string embeddings to numpy arrays if needed
for col in ['Embedding1', 'Embedding2']:
    if col in emb_results.columns and isinstance(emb_results[col].iloc[0], str):
        emb_results[col] = emb_results[col].apply(lambda x: np.array(eval(x)))

def emb_to_str(emb):
    return json.dumps(emb.tolist()) if isinstance(emb, np.ndarray) else json.dumps(emb)

# Pivot and convert to string for Embedding1
pivot1 = emb_results.pivot_table(index='Category', columns='Gender', values='Embedding1', aggfunc='first')
pivot1 = pivot1.apply(lambda col: col.map(emb_to_str))
pivot1.columns = [f'Embedding1_{col}' for col in pivot1.columns]

# Pivot and convert to string for Embedding2 (if present)
if 'Embedding2' in emb_results.columns:
    pivot2 = emb_results.pivot_table(index='Category', columns='Gender', values='Embedding2', aggfunc='first')
    pivot2 = pivot2.apply(lambda col: col.map(emb_to_str))
    pivot2.columns = [f'Embedding2_{col}' for col in pivot2.columns]
    final_df = pd.concat([pivot1, pivot2], axis=1)
else:
    final_df = pivot1

final_df.reset_index(inplace=True)
final_df.to_csv(OUTPUT, index=False)
