import os
import csv
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

current_directory = os.getcwd()
subfolders = ["sd3_label_image"]  
csv_path = "clip_results2.csv"

with open(csv_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_path", "male_probability", "female_probability"])

    for subfolder in subfolders:
        folder_path = os.path.join(current_directory, subfolder)
        
        for root, _, files in os.walk(folder_path):
            image_files = [f for f in files if f.endswith((".jpg", ".png", ".webp"))]

            for i, filename in enumerate(image_files, start=1):
                image_path = os.path.join(root, filename)
                
                try:
                    image = Image.open(image_path)
                    image.load() 
                except Exception as e:
                    print(f"Error loading image {filename}: {e}")
                    continue
                
                inputs = processor(text=["a photo of a male", "a photo of a female"], images=image, return_tensors="pt")

                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image 
                probs = logits_per_image.softmax(dim=1)

                male_prob, female_prob = probs[0].tolist()

                print(f"Results for image {i} ({filename}):")
                print(probs)
                print()


                writer.writerow([image_path, male_prob, female_prob])

print(f"Results saved to {csv_path}")
