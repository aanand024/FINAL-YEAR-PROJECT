import os
import csv
from tqdm import tqdm
from pipeline_classes import Pipeline
import numpy as np

# Initialise all pipelines
pipelines = [
    Pipeline(),
]

# Ensure output directory exists
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# CSV filenames corresponding to each pipeline
csv_filenames = [
    os.path.join(output_dir, "clip_enhance_pipeline_results.csv"),
]

# Directory to scan for images
data_dir = "sd3_label_image"

# Iterate over all pipelines
for i, pipeline in enumerate(pipelines):
    results = []
    # Walk through all subdirectories and images
    image_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    for image_path in tqdm(image_paths, desc=f"Processing Pipeline {i+7}"):
        try:
            # Run the pipeline on the image
            predicted_gender, confidence, probs = pipeline.process(image_path)
            if predicted_gender is None or confidence is None or probs is None:
                raise ValueError("Pipeline returned None for one of the outputs")
            
            # Store the result
            results.append([image_path, predicted_gender, confidence, probs.tolist()])
        except Exception as e:
            print(f"Error processing {image_path} with Pipeline {i+7}: {e}")
            results.append([image_path, "Error", None, None])

    # Save results to CSV in the output directory
    with open(csv_filenames[i], mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Predicted Gender", "Confidence", "Probabilities"])
        writer.writerows(results)

    print(f"Results for Pipeline saved to {csv_filenames[i]}")

print("Processing completed for all pipelines.")
