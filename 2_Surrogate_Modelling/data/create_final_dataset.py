'''
This Script creates the final dataset for the surrogate models. 
'''

import pandas as pd

# prompt embeddings - only neutral prompts 'Embedding1_Neutral', 'Embedding2_Neutral'
embedds = pd.read_csv('surrogate_models/data/LATEST_all_embeddings_by_category.csv')

#cosine similarities 
cosine1 = pd.read_csv('analysis/embeddings/latest/LATEST_rp_updated_emb_cosine_simil.csv')
cosine2 = pd.read_csv('analysis/embeddings/latest/LATEST_rp_updated_emb_cosine_simil_2.csv')

#output image bias score - to label 
output_bias = pd.read_csv('analysis/manual_l/prompt_bias_score_manual_category_gender_stats.csv')

# Merge on 'Category'
df = embedds[['Category', 'Embedding1_Neutral', 'Embedding2_Neutral']].merge(
    cosine1[['Category', 'difference']], on='Category', how='inner', suffixes=('', '_cos1')
).merge(
    cosine2[['Category', 'difference']], on='Category', how='inner', suffixes=('', '_cos2')
).merge(
    output_bias[['Category', 'prompt_bias_score']], on='Category', how='inner'
)

# Rename columns for clarity
df.rename(columns={'difference': 'difference_cos1', 'difference_cos2': 'difference_cos2'}, inplace=True)
df.to_csv('LATEST_merged_for_regression.csv', index=False)


# MERGE FOR FINAL DATASET
# Combine the two CSV files by concatenating rows
merged_extra = pd.read_csv('2_Surrogate_Modelling/data/LATEST_merged_extra_data.csv')
merged = pd.read_csv('2_Surrogate_Modelling/data/LATEST_merged_for_regression.csv')

combined_df = pd.concat([merged, merged_extra], ignore_index=True)
combined_df.to_csv('LATEST_all_data_merged_for_regression.csv', index=False)


