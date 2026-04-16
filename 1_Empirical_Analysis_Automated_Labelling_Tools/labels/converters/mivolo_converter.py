import pandas as pd

df = pd.read_csv("1_Empirical_Analysis_Automated_Labelling_Tools/labels/results/mivolo_results.csv")

def clean_gender(gender):
    if gender.strip().lower() == "unknown":
        return "unlabelled"
    return gender.strip().lower()

df["gender"] = df["gender"].apply(clean_gender)
df.to_csv("mivolo_results_cleaned.csv", index=False)