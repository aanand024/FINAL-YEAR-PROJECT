import pandas as pd

'''
This script  creates extra data dataset for surrogate models.
'''

# prompt embeddings - only neutral prompts 'Embedding1_Neutral', 'Embedding2_Neutral'
embedds = pd.read_csv('2_Surrogate_Modelling/data/LATEST_extra_data_all_embeddings_by_category.csv')

#cosine similarities
cosine1 = pd.read_csv('1_Empirical_Analysis_Embeddings/embeddings/cosine_sim/LATEST_extra_data_emb_cosine_simil.csv')
cosine2 = pd.read_csv('1_Empirical_Analysis_Embeddings/embeddings/cosine_sim/LATEST_extra_data_emb_cosine_simil_2.csv')

#output image bias score - to label 
output_bias = pd.read_csv('1_Empirical_Analysis_Embeddings/ground_truth/prompt_bias_score_extra_data_stats.csv')

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
df.to_csv('LATEST_merged_extra_data.csv', index=False)


