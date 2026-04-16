import pandas as pd
df = pd.read_csv("1_Empirical_Analysis_Automated_Labelling_Tools/labels/results/clip_prob_results.csv")

#  Possible labels ['a photo of a male' 'a photo of a female']
def map_label(label):
    if label == "a photo of a male":
        return "male"
    elif label == "a photo of a female":
        return "female"
    else:
        return "unlabelled"

df["parsed_gender"] = df["Predicted Label"].apply(map_label)

# Load skipped images
skipped = pd.read_csv("1_Empirical_Analysis_Automated_Labelling_Tools/labels/results/clip_prob_skipped_images.csv")

# Create DataFrame for skipped images
skipped_rows = pd.DataFrame({
    "Image": skipped["Image"],
    "Predicted Label": ["unlabelled"] * len(skipped),
    "Probability Man": [None] * len(skipped),
    "Probability Woman": [None] * len(skipped),
    "parsed_gender": ["unlabelled"] * len(skipped)
})

# Concatenate and save
df_out = pd.concat([df, skipped_rows], ignore_index=True)
df_out.to_csv("clip_prob_cleaned.csv", index=False)