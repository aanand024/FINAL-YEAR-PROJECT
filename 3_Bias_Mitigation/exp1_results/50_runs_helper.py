import pandas as pd

'''This script compares each optimal solution's gender bias to the ground truth.'''

INPUT_CSV = "3_Bias_Mitigation/exp1_results/comparison/runtime_comparison_optimal_solutions_from_results.csv"
OUTPUT_CSV = "3_Bias_Mitigation/exp1_results/comparison/swe_prompt_aggregated_results.csv"

ORIGINAL_GENDER_BIAS = 1.0 # banker prompt
df = pd.read_csv(INPUT_CSV)

df = df.rename(columns={
    "round": "run_id",
    "fitness": "modified_gender_bias",
})

df["original_gender_bias"] = ORIGINAL_GENDER_BIAS
df["improvement"] = df["original_gender_bias"] - df["modified_gender_bias"]

df.to_csv(OUTPUT_CSV, index=False)
print(df.head())
print(f"\nCollected {len(df)} runs")