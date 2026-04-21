import pandas as pd

df = pd.read_csv("1_Empirical_Analysis_Automated_Labelling_Tools/labels/results/clip_unc_results.csv")

#  Possible labels
# ['A photo of a person who looks like a man'
#  'A photo of a person with an uncertain gender'
#  'A photo of a person who looks like a woman']

def map_label(label):
    if label == "A photo of a person who looks like a man":
        return "male"
    elif label == "A photo of a person who looks like a woman":
        return "female"
    elif label == "A photo of a person with an uncertain gender":
        return "unlabelled"
    else:
        return "unlabelled"

df["parsed_gender"] = df["Predicted Label"].apply(map_label)

# Images that were not labelled (if it thinks likely object than a person we skip these)
new_unlabelled = [
    "sd3_label_image/fighting/fighting_17.png",
    "sd3_label_image/cleaner/cleaner_2.png",
    "sd3_label_image/cleaner/cleaner_14.png",
    "sd3_label_image/playing/playing_20.png",
    "sd3_label_image/playing/playing_16.png",
    "sd3_label_image/playing/playing_18.png",
    "sd3_label_image/police/police_2.png",
    "sd3_label_image/police/police_19.png",
    "sd3_label_image/selfish/selfish_18.png",
    "sd3_label_image/judge/judge_17.png",
    "sd3_label_image/mall/mall_9.png",
    "sd3_label_image/generous/generous_18.png",
    "sd3_label_image/mean/mean_8.png",
    "sd3_label_image/desktop/desktop_7.png",
    "sd3_label_image/desktop/desktop_11.png",
    "sd3_label_image/architect/architect_10.png",
    "sd3_label_image/architect/architect_3.png",
    "sd3_label_image/pen/pen_16.png",
    "sd3_label_image/pen/pen_14.png",
    "sd3_label_image/writing/writing_5.png",
    "sd3_label_image/cigar/cigar_5.png",
    "sd3_label_image/astronomer/astronomer_16.png",
    "sd3_label_image/astronomer/astronomer_13.png",
    "sd3_label_image/astronomer/astronomer_20.png",
    "sd3_label_image/astronomer/astronomer_15.png",
    "sd3_label_image/photographer/photographer_9.png"
]

new_rows = pd.DataFrame({
    "Image": new_unlabelled,
    "Predicted Label": [None] * len(new_unlabelled),
    "Probability Man": [None] * len(new_unlabelled),
    "Probability Woman": [None] * len(new_unlabelled),
    "Probability Unknown": [None] * len(new_unlabelled),
    "parsed_gender": ["unlabelled"] * len(new_unlabelled)
})

df = pd.concat([df, new_rows], ignore_index=True)
df.to_csv("clip_unc_cleaned.csv", index=False)