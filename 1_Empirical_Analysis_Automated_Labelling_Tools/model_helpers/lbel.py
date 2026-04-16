#pylint: skip-file
import os
import csv
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# Initialize the BLIP2 model and processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")

# Function to get all image files in the specified directory
def get_image_files(directory):
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    return image_files

# Function to process each image and get the gender
def process_image(image_path):
    try:
        raw_image = Image.open(image_path).convert('RGB')
        question = "Question: the person is a male or a female? Answer:"
        inputs = processor(raw_image, question, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        answer = processor.decode(out[0], skip_special_tokens=True).strip()
        return answer
    except Exception as e:
        return f"Error: {str(e)}"

# Main function to iterate over images and save results to CSV
def main():
    base_directory = "sd3_label_image"
    image_files = get_image_files(base_directory)
    
    with open('results_t2i.csv', mode='w', newline='') as csv_file:
        fieldnames = ['Image Path', 'Gender']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        
        for image_file in image_files:
            gender = process_image(image_file)
            writer.writerow({'Image Path': image_file, 'Gender': gender})
            print(f"Processed {image_file}: {gender}")

if __name__ == "__main__":
    main()
