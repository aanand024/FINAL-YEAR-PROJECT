import pandas as pd
import torch
import pickle
from diffusers import StableDiffusion3Pipeline

"""This script collects the embeddings for the extra dataset."""

# Note for extra data, category is not as well-defined
def collect_embedding(prompt, gender, embeddings, prompts_used):
  # Extract category as the part after 'that'
  if 'that' in prompt:
      cat = prompt.split('that', 1)[1].strip().rstrip('.')
  else:
      cat = prompt
      print("error")

  # Gender-specific prompt
  if gender == 'Neutral':
      gendered_prompt = prompt
  elif gender == 'Male':
      gendered_prompt = prompt.replace("person", "man")
  elif gender == 'Female':
      gendered_prompt = prompt.replace("person", "woman")
  else:
      gendered_prompt = prompt

  prompts_used.append(gendered_prompt)

  token = pipe.tokenizer(gendered_prompt, return_tensors="pt").to('cpu')
  token_2 = pipe.tokenizer_2(gendered_prompt, return_tensors="pt").to('cpu')
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
  
  imgs_list = pd.read_csv('1_Empirical_Analysis/embeddings/collecting_emb/extra_data.csv')

  embeddings = {
    'Category': [],
    'Gender': [],
    'Embedding1': [],
    'Embedding2': [],
  }
  
  device ="cpu"

  for idx, row in imgs_list.iterrows():
    prompt = row['Prompt']
    print(f'Processing prompt: {prompt}')
    collect_embedding(prompt, 'Neutral', embeddings, prompts_used)
    collect_embedding(prompt, 'Male', embeddings, prompts_used)
    collect_embedding(prompt, 'Female', embeddings, prompts_used)

    
    with open('LATEST_extra_data_embeddings.pkl', 'wb') as f:
        pickle.dump(embeddings, f)

    # Save prompts used to a txt file
    with open('extra_data_used_prompts.txt', 'w') as f:
        for prompt in prompts_used:
            f.write(prompt + '\n')