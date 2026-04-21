#pylint: skip-file
import os
import csv
import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipForQuestionAnswering

def main():
    # device = "mps" if torch.backends.mps.is_available() else "cpu"
    device = "cpu"    
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
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
                out = model.generate(**inputs)
                answer = processor.decode(out[0], skip_special_tokens=True).strip()
                results.append((path, answer))
                print(path, "->", answer)

    with open("blip2_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Path", "Output"])
        writer.writerows(results)

if __name__ == "__main__":
    main()