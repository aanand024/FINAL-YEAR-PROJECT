'''Helper script to clean results from CLIP.'''
import pandas as pd

df = pd.read_csv("1_Empirical_Analysis_Automated_Labelling_Tools/labels/results/clip_results.csv")

def clean_path(path):
    idx = path.find("sd3_label_image")
    return path[idx:] if idx != -1 else path

def get_gender(row):
    if row["male_probability"] > row["female_probability"]:
        return "male"
    else:
        return "female"

df["Image"] = df["image_path"].apply(clean_path)
df["Predicted Label"] = df.apply(get_gender, axis=1)
df["parsed_gender"] = df["Predicted Label"]

df[["Image", "Predicted Label", "male_probability", "female_probability", "parsed_gender"]].to_csv("labelling/results/clip_results_cleaned.csv", index=False)