import os
from PIL import Image
import torch
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import cv2
import mediapipe as mp
import numpy as np


device = "cpu"


model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)


image_dir = "sd3_label_image"

labels = ["a photo of a male", "a photo of a female"]


results = []

image_paths = []
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, file))

skipped = []
for image_path in tqdm(image_paths, desc="Processing images"):

    image = Image.open(image_path).convert("RGB")
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results_mp = face_detection.process(image_cv)

    if not results_mp.detections:
        skipped.append({"Image": image_path, "Reason": "No face detected"})
        continue
    inputs = processor(text=labels, images=image, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    predicted_label = labels[probs.argmax()]
    predicted_probability = probs.max()
    
    if predicted_probability < 0.9:
        skipped.append({"Image": image_path, "Reason": "Low confidence"})

    if predicted_probability >= 0.9:
        results.append({
            "Image": image_path,
            "Predicted Label": predicted_label,
            "Probability Man": probs[0],
            "Probability Woman": probs[1]
        })
        print('u')

df = pd.DataFrame(results)
df.to_csv("clip_prob_results.csv", index=False)
df_skipped = pd.DataFrame(skipped)
df_skipped.to_csv("clip_prob_skipped_images.csv", index=False)

print("Results saved for CLIP-Prob.")