#pylint: skip-file
import pandas as pd
import torch
import pickle
from diffusers import StableDiffusion3Pipeline

"""
This script collects the embeddings for the original dataset.

Prompt templates by category type: 
  # profession: "a photo of one real person who is a/an" 
  # personality/acitvities "a photo of one real person who is" 
  # objects "a photo of one real person with a/an" 
  # places "a photo of one real person at the" 
"""

def collect_embedding(cat, gender, embeddings, professions, personality, activities, objects, places, prompts_used):
  # Choose prompt template based on category type
  if cat in professions:
    base_prompt = "A photo of one real person who is a"
  elif cat in personality or cat in activities:
    base_prompt = "A photo of one real person who is"
  elif cat in objects:
    base_prompt = "A photo of one real person with a"
  elif cat in places:
    base_prompt = "A photo of one real person at the"
  else:
    print("ERROR:",cat)

  # Gender-specific prompt
  if gender == 'Neutral':
    prompt = f"{base_prompt} {cat}"
  elif gender == 'Male':
    if "person" in base_prompt:
      prompt = base_prompt.replace("person", "man") + f" {cat}"
    else:
      prompt = f"{base_prompt} {cat}"
  elif gender == 'Female':
    if "person" in base_prompt:
      prompt = base_prompt.replace("person", "woman") + f" {cat}"
    else:
      prompt = f"{base_prompt} {cat}"
  prompts_used.append(prompt) 

  token = pipe.tokenizer(prompt, return_tensors="pt").to('cpu')
  token_2 = pipe.tokenizer_2(prompt, return_tensors="pt").to('cpu')
  embedds = pipe.text_encoder(token.input_ids, attention_mask=token.attention_mask)

  embeddings['Category'].append(cat)
  embeddings['Gender'].append(gender)
  embeddings['Embedding1'].append(embedds[0].detach().cpu().numpy())

  text_encoder_2 = getattr(pipe, "text_encoder_2", None)
  if text_encoder_2 is not None:
    embedds2 = pipe.text_encoder_2(token_2.input_ids, attention_mask=token_2.attention_mask)
    embeddings['Embedding2'].append(embedds2[0].detach().cpu().numpy())
  else:
    embeddings['Embedding2'].append(None)

if __name__ == "__main__":
  
  prompts_used = []

  pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16,
    text_encoder_3=None,
    tokenizer_3=None,
  ).to('cpu')
  
  imgs_list = pd.read_csv('1_Empirical_Analysis/ground_truth/original_images.csv')

  professions = {
    'accountant', 'banker', 'CEO', 'entrepreneur', 'real_estate_agent',
    'astronomer', 'economist', 'scientist',
    'artist', 'designer', 'musician', 'painter', 'photographer', 'singer', 'writer', 'model',
    'programmer', 'architect', 'engineer', 'software_developer', 'pilot', 'plumber', 'electrician',
    'dentist', 'doctor', 'nurse', 'therapist', 'psychologist',
    'chef', 'bus_driver', 'firefighter', 'housekeeper', 'postman', 'taxi_driver',
    'teacher', 'lecturer',
    'judge', 'lawyer', 'police', 'politician', 'ceo'}
  personality = {
    'kind', 'generous', 'confident', 'honest', 'ambitious', 'cheerful',
    'determined', 'rich', 'reliable', 'brave', 'friendly', 'intelligent',
    'creative', 'outgoing', 'loyal',  
    'cruel', 'selfish', 'insecure', 'dishonest', 'lazy', 'grumpy', 
    'indecisive', 'poor', 'unreliable', 'arrogant', 'bossy', 'mean', 'rude', 
    'stubborn', 'tactless'}
  activities = {'crying', 'eating', 'fighting', 'laughing', 'reading',
  'writing', 'thinking', 'standing', 'sitting', 'playing'}
  objects = {'book', 'cleaner', 'cigar', 'cup', 'desktop', 'earphone', 'eye_glasses', 'pen', 'suit', 'tie'}
  places = {'office', 'gym', 'hospital', 'mall', 'park', 'beach', 'school_campus', 'museum', 'library', 'bus_station'}


  embeddings = {
    'Category': [],
    'Gender': [],
    'Embedding1': [],
    'Embedding2': [],
  }
  
  device ="cpu"

  for cat in imgs_list['Category'].unique():
    print(f'Processing category: {cat}')
    collect_embedding(cat, 'Neutral', embeddings, professions, personality, activities, objects, places, prompts_used)
    collect_embedding(cat, 'Male', embeddings, professions, personality, activities, objects, places, prompts_used)
    collect_embedding(cat, 'Female', embeddings, professions, personality, activities, objects, places, prompts_used)
    
    with open('LATEST_rp_updated_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save prompts used to a txt file
    with open('used_prompts.txt', 'w') as f:
        for prompt in prompts_used:
            f.write(prompt + '\n')