'''This script computes the runtime analysis of the proposed GA.'''

import os
import pandas as pd
import numpy as np

base_path = "3_Bias_Mitigation/exp1_results/comparison/runtime_comparison"  # change if needed
runtimes = []

for i in range(1, 51):
    path = os.path.join(base_path, f"round_{i}", "timing.csv")
    
    if not os.path.exists(path):
        continue
        
    df = pd.read_csv(path)
    
    # Case handling:
    if df.shape[0] == 1:
        runtime = df.iloc[0, 0]
    else:
        # if multiple rows, assume cumulative or per-gen
        runtime = df.iloc[-1, 0]  # take final value
    
    runtimes.append(runtime)

runtimes = np.array(runtimes)

mean_sec = runtimes.mean()
print(f"Mean: {mean_sec:.2f} sec ({mean_sec/60:.2f} min)")
print("Std:", runtimes.std())
print("Min:", runtimes.min())
print("Max:", runtimes.max())
print("Total (seconds):", runtimes.sum())
print("Total (hours):", runtimes.sum() / 3600)