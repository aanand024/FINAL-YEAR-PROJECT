import pandas as pd

# Load the results CSV
df = pd.read_csv("1_Empirical_Analysis_Automated_Labelling_Tools/labels/results/clip_enhance_pipeline_results.csv")

# Define function to clean output
def clean_output(pred):
    if pred == "male":
        return "male"
    elif pred == "female":
        return "female"
    else:
        return "unlabelled"

# Create new DataFrame with required columns
out_df = pd.DataFrame({
    "image_path": df["Image Path"],
    "output": df["Predicted Gender"].apply(clean_output)
})

# Save to new CSV
out_df.to_csv("clip_enhance_cleaned.csv", index=False)