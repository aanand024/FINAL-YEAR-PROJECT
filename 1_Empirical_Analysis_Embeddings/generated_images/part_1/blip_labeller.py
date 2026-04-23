'''
This script is used to automatically label the gender of the set of generated images, 
using Blip (https://huggingface.co/Salesforce/blip-vqa-base). 
'''

import os
import csv
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

    model.to(device)
    model.eval()

    base_directory = "sd3_label_image"

    results = []

    for root, _, files in os.walk(base_directory):
        for file in files:
            if file.lower().endswith(".png"):
                path = os.path.join(root, file)
                image = Image.open(path).convert("RGB")
                prompt = "Question: the person is a male or a female? Answer:"
                inputs = processor(image, prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=10)
                answer = processor.decode(out[0], skip_special_tokens=True)
                results.append((path, answer))
                print(path, "->", answer)

    with open("output_images_labels.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Path", "Output"])
        writer.writerows(results)

if __name__ == "__main__":
    main()
