import os
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

image_dir = "sd3_label_image" 

person_labels = ["a photo of a person", "a photo of an object"]
gender_labels = ["A photo of a person who looks like a man", "A photo of a person who looks like a woman", "A photo of a person with an uncertain gender"]

results = []

image_paths = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, file))

for image_path in tqdm(image_paths, desc="Processing images"):
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open {image_path}: {e}")
        continue

    inputs = processor(text=person_labels, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    if probs[0] < probs[1]:
        print(f"Skipped (not a person): {image_path}")
        continue



    inputs = processor(text=gender_labels, images=image, return_tensors="pt", padding=True).to(device)


    with torch.no_grad():
        outputs = model(**inputs)


    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

   
    predicted_label = gender_labels[probs.argmax()]


    results.append({
        "Image": image_path,
        "Predicted Label": predicted_label,
        "Probability Man": probs[0],
        "Probability Woman": probs[1],
        "Probability Unknown": probs[2]
    })

df = pd.DataFrame(results)
df.to_csv("2222prediction_results_filter.csv", index=False)

print("Results saved to prediction_results_filter.csv")
