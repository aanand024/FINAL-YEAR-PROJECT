import csv
import glob
import os
import random
import ast
import json
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity

from gao import pipe, _get_surrogate_model, construct_gendered_prompts

"""
This script evaluates and generates images for a set of neutral prompts using the previously optimised
individuals (solutions) from the GA.  For each prompt, it predicts gender bias, generates mutated images 
by manipulating prompt embeddings, and saves both the results and generated images for further analysis. 

The script is designed to test the generalisation of bias mitigation strategies across diverse prompts.

Results can be found in 3_Bias_Mitigation/exp2_results.
"""


# CHANGE the following according to preference:
DIRECTORY_NAME = "prompt_only_outputs"
NUM_RUNS_PER_PROMPT = 5
"""
MODE Settings
prompt = modifying prompt embeddings only
pooled = modifying pooled embeddings only
both = modifying both prompt and pooled embeddings 
"""
MODE = "prompt"


def get_encoder_embeddings(prompt: str):
    with torch.no_grad():
        text_inputs = pipe.tokenizer(prompt, return_tensors="pt").to(pipe.device)
        text_inputs_2 = pipe.tokenizer_2(prompt, return_tensors="pt").to(pipe.device)

        emb1 = pipe.text_encoder(
            text_inputs.input_ids,
            attention_mask=text_inputs.attention_mask,
        )[0].detach().cpu().numpy().flatten()

        emb2 = pipe.text_encoder_2(
            text_inputs_2.input_ids,
            attention_mask=text_inputs_2.attention_mask,
        )[0].detach().cpu().numpy().flatten()

    return emb1, emb2


def predict_bias_for_prompt(prompt: str, individual, xgb_pipe):
    gendered_prompts = construct_gendered_prompts(prompt)
    neutral, male, female = gendered_prompts

    embeddings = {}
    for gp in gendered_prompts:
        embeddings[gp] = get_encoder_embeddings(gp)

    emb1, emb2 = embeddings[neutral]

    individual = np.array(individual, dtype=np.float32)
    if individual.shape[0] != 2048:
        raise ValueError(f"Expected individual of length 2048, got {individual.shape[0]}")

    ind_emb1 = individual[:768]
    ind_emb2 = individual[768:2048]

    mutated_emb1 = emb1 * ind_emb1
    mutated_emb2 = emb2 * ind_emb2

    mutated_emb1_norm = np.linalg.norm(mutated_emb1)
    mutated_emb2_norm = np.linalg.norm(mutated_emb2)
    mutated_emb1_mean = mutated_emb1.mean()
    mutated_emb2_mean = mutated_emb2.mean()
    mutated_emb1_std = mutated_emb1.std()
    mutated_emb2_std = mutated_emb2.std()
    mutated_emb1_max = mutated_emb1.max()
    mutated_emb2_max = mutated_emb2.max()
    mutated_emb1_min = mutated_emb1.min()
    mutated_emb2_min = mutated_emb2.min()

    emb1_male, emb2_male = embeddings[male]
    emb1_female, emb2_female = embeddings[female]

    cos1_male = cosine_similarity(mutated_emb1.reshape(1, -1), emb1_male.reshape(1, -1))[0, 0]
    cos1_female = cosine_similarity(mutated_emb1.reshape(1, -1), emb1_female.reshape(1, -1))[0, 0]
    diff_cos1 = cos1_male - cos1_female

    cos2_male = cosine_similarity(mutated_emb2.reshape(1, -1), emb2_male.reshape(1, -1))[0, 0]
    cos2_female = cosine_similarity(mutated_emb2.reshape(1, -1), emb2_female.reshape(1, -1))[0, 0]
    diff_cos2 = cos2_male - cos2_female

    features = np.concatenate([
        mutated_emb1,
        mutated_emb2,
        np.array([
            mutated_emb1_norm, mutated_emb2_norm,
            mutated_emb1_mean, mutated_emb2_mean,
            mutated_emb1_std, mutated_emb2_std,
            mutated_emb1_max, mutated_emb2_max,
            mutated_emb1_min, mutated_emb2_min,
            diff_cos1, diff_cos2
        ], dtype=np.float32)
    ]).reshape(1, -1)

    gender_bias = xgb_pipe.predict(features)[0]
    return float(gender_bias)


def expand_individual_with_zeros(individual, target_width=4096, used_width=2048):
    """
    Padding: Append zeros to the 2048 individual to make it 4096
    """
    individual = np.array(individual, dtype=np.float32)

    if individual.shape[0] != used_width:
        raise ValueError(f"Expected individual of length {used_width}, got {individual.shape[0]}")

    if target_width < used_width:
        raise ValueError(f"target_width {target_width} must be >= used_width {used_width}")

    pad_len = target_width - used_width
    zero_pad = np.zeros(pad_len, dtype=np.float32)
    expanded = np.concatenate([individual, zero_pad], axis=0)

    return expanded


def generate_image_for_prompt(prompt: str, individual, output_path: str, mode):
    """
    Mutate prompt_embeds
    """
    individual = np.array(individual, dtype=np.float32)
    with torch.no_grad():
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            prompt_3=None,
            device=pipe.device,
            do_classifier_free_guidance=True,
        )
        
        mutated_prompt_embeds = prompt_embeds
        mutated_pooled_prompt_embeds = pooled_prompt_embeds
        
        if mode in {"prompt", "both"}:
            gene_4096 = expand_individual_with_zeros(
                individual,
                target_width=prompt_embeds.shape[-1],
                used_width=2048,
            )
            gene_4096 = torch.from_numpy(gene_4096).to(pipe.device).view(1, 1, -1)

            if gene_4096.shape[-1] != prompt_embeds.shape[-1]:
                raise ValueError(
                    f"Expanded gene width {gene_4096.shape[-1]} does not match prompt_embeds width {prompt_embeds.shape[-1]}"
                )

            mutated_prompt_embeds = prompt_embeds * gene_4096

        if mode in {"pooled", "both"}:
            gene_2048 = torch.from_numpy(individual).to(pipe.device).unsqueeze(0)  # [1, 2048]
            if gene_2048.shape[1] != pooled_prompt_embeds.shape[1]:
                raise ValueError(
                    f"Genome length {gene_2048.shape[1]} does not match pooled_prompt_embeds width {pooled_prompt_embeds.shape[1]}"
                )

            mutated_pooled_prompt_embeds = pooled_prompt_embeds * gene_2048


        image = pipe(
            prompt_embeds=mutated_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=mutated_pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        ).images[0]

    image.save(output_path)

def load_best_individuals(csv_path):
    all_individuals = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            individual = np.array(ast.literal_eval(row["individual"]), dtype=np.float32)
            all_individuals.append(individual)
    return all_individuals


def load_prompts(csv_path):
    prompts = []
    with open(csv_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                prompts.append(row[0].strip())
    print(f"Number of prompts: {prompts.__len__()}")

    return prompts


def safe_filename(text: str, max_len: int = 100):
    text = text.replace(" ", "_").replace("/", "_").replace("\\", "_")
    text = "".join(ch for ch in text if ch.isalnum() or ch in {"_", "-"})
    return text[:max_len] if len(text) > max_len else text


if __name__ == "__main__":
    prompts = load_prompts("3_Bias_Mitigation/exp2_scripts/neutral_prompts.csv")
    all_individuals = load_best_individuals("3_Bias_Mitigation/exp1_results/config3/crossover_0.2_fixed_aggregated_results.csv")
    xgb_pipe = _get_surrogate_model()

    print(f"Loaded {len(prompts)} prompts")
    print(f"Loaded {len(all_individuals)} best individuals")

    os.makedirs(DIRECTORY_NAME, exist_ok=True)

    results = []

    for i, prompt in enumerate(prompts):
        idx = random.randint(0, len(all_individuals) - 1)
        individual = all_individuals[idx]
        
        for run_idx in range(NUM_RUNS_PER_PROMPT):
            try:
                gender_bias = predict_bias_for_prompt(prompt, individual, xgb_pipe)

                file_stub = safe_filename(prompt)
    
                img_path = os.path.join(f"{DIRECTORY_NAME}_outputs", f"{i:03d}_{file_stub}_run{run_idx+1}.png")

                generate_image_for_prompt(prompt, individual, img_path, MODE)

                results.append({
                    "prompt": prompt,
                    "run": run_idx+1, 
                    "individual_id": idx,
                    "individual": json.dumps(individual.tolist()),
                    "gender_bias": gender_bias,
                    "image_path": img_path,
                })

                print(f"[{i+1}/{len(prompts)}] Done: {prompt}")

            except Exception as e:
                print(f"[{i+1}/{len(prompts)}] Failed on prompt: {prompt}")
                print(f"Error: {e}")
                results.append({
                    "prompt": prompt,
                    "run":run_idx+1,
                    "individual_id": "",
                    "individual":"",
                    "gender_bias": "",
                    "image_path": "",
                })

    # Optional, if you wish to save results to CSV for bookeeping (currently only images are generated)
    # with open(f"mutated_prompt_results_{DIRECTORY_NAME}.csv", "w", newline="") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["prompt", "run", "individual_id", "individual", "gender_bias", "image_path"])
    #     for res in results:
    #         writer.writerow([
    #             res["prompt"],
    #             res["run"],
    #             res["individual_id"],
    #             res["individual"],
    #             res["gender_bias"],
    #             res["image_path"],
    #         ])

    # print("Saved results to mutated_prompt_results.csv")