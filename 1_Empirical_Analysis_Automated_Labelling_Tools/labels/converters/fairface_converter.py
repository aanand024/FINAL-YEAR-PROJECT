import pandas as pd
import re

# Load the CSV
df = pd.read_csv('1_Empirical_Analysis_Automated_Labelling_Tools/labels/results/fairface_results.csv')

# Load all image paths
all_imgs = pd.read_csv("1_Empirical_Analysis_Automated_Labelling_Tools/labels/converters/sd3_images.csv")["img_path"].tolist()


def construct_img_path(face_name_align):
    # Extract the part after 'detected_faces/' and before '_face'
    match = re.match(r"detected_faces/([a-zA-Z0-9_]+)_face\d+\.png", face_name_align)
    if match:
        base = match.group(1)  # e.g., 'pen_2' or 'eye_glasses_11'
        # Split at the last underscore to get folder and filename
        folder, filename = base.rsplit('_', 1)
        return f"sd3_label_image/{folder}/{folder}_{filename}.png"
    return ""


df['img_path'] = df['face_name_align'].apply(construct_img_path)
new_df = df[['img_path', 'gender']]

# Find missing images
labelled_imgs = set(new_df["img_path"])
missing_imgs = [img for img in all_imgs if img not in labelled_imgs]

# Create DataFrame for missing images with unlabelled gender
missing_df = pd.DataFrame({
    "img_path": missing_imgs,
    "gender": ["unlabelled"] * len(missing_imgs)
})

# Combine and save
out_df = pd.concat([new_df, missing_df], ignore_index=True)
out_df.to_csv("new_fairface_results_with_unlabelled.csv", index=False)

duplicates = new_df['img_path'][new_df['img_path'].duplicated(keep=False)]
if not duplicates.empty:
    print("Images with multiple faces detected (multiple entries):")
    print(duplicates.value_counts())