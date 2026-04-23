"""
This file implements the core GA optimiser for bias mitigation using prompt embedding manipulation and a 
surrogate model. Handles population initialisation, fitness evaluation, crossover, mutation, and optimisation 
loop, leveraging Stable Diffusion and XGBoost for fitness assessment.
"""

import calendar
import copy
import os
import random
import time
import logging
from PIL import Image
import numpy
import torch
from deap import algorithms, base, creator, tools
from diffusers import (
    StableDiffusion3Pipeline,
)
import json
import csv
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



_XGB_PIPE = None

def _get_surrogate_model():
    global _XGB_PIPE
    if _XGB_PIPE is None:
        _XGB_PIPE = joblib.load(os.path.join(os.path.dirname(__file__), "LATEST_final_xgb_full_data_pipeline.joblib"))
    return _XGB_PIPE

print('!!! START OF FILE')
device = "cpu"

# Uncomment the below if using cuda
# device = "cuda" if torch.cuda.is_available() else "cpu"
# if torch.cuda.is_available():
#     torch.cuda.empty_cache()


print('!!! HERE 1')
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    # torch_dtype=torch.float16, # for cuda
    torch_dtype=torch.float32, # for cpu 
    text_encoder_3=None,
    tokenizer_3=None,
)

pipe.to(device)

# For cuda
# pipe.enable_xformers_memory_efficient_attention()
# if device == "cuda":
#     pipe.enable_sequential_cpu_offload()


def construct_gendered_prompts(prompt):
    prompts = []
    prompts.append(prompt) # neutral prompt

    if 'person' in prompt:
        prompts.append(prompt.replace('person','man'))
        prompts.append(prompt.replace('person','woman'))
    else: 
        raise ValueError("Warning (Incorrect prompt format): the prompt does not contain the word 'person")
    return prompts

def collect_embeddings_and_evaluate_fitness(prompt, individual):
    """
    Function to collect the embeddings for a given prompt and compute the bias fitness and the image quality fitness.

    1- Collect the embeddings from the prompt
    2- Multiply the prompt embeddings times the individual
    3- Add the features needed as input for the ML model to the mutated embeddings
    4- Feed the XGBoost Regressor with the new features
    5- Return the prediction as output
    
    :param prompt: the prompt to generate the image
    :param individual: the individual to evaluate
    """
   
    xgb_pipe = _get_surrogate_model()
    print("Individual dimension:", len(individual))

    # Step 0: Create 3 gendered prompts 
    gendered_prompts = construct_gendered_prompts(prompt)
    neutral, male, female = gendered_prompts

    # Step 1: Collect prompt embeddings 
    embeddings = {}
    for gp in gendered_prompts:
        with torch.no_grad():
            text_inputs = pipe.tokenizer(gp, return_tensors="pt").to(device)
            text_inputs_2 = pipe.tokenizer_2(gp, return_tensors="pt").to(device)
        
            emb1 = pipe.text_encoder(text_inputs.input_ids, attention_mask=text_inputs.attention_mask)[0].cpu().numpy().flatten()
            text_encoder_2 = getattr(pipe, "text_encoder_2", None)
            if text_encoder_2 is not None:
                emb2 = pipe.text_encoder_2(text_inputs_2.input_ids, attention_mask=text_inputs_2.attention_mask)[0].cpu().numpy().flatten()
            else:
                emb2 = None
        embeddings[gp] = (emb1, emb2)
    
    # Step 2: Mutate only neutral prompt
    emb1, emb2 = embeddings[neutral]
    print(emb2.shape[0])
    ind_emb1 = individual[:768]
    ind_emb2 = individual[768:2048]
    mutated_emb1 = emb1 * ind_emb1
    mutated_emb2 = emb2 * ind_emb2 if emb2 is not None else None

    # Step 3: Prepare features for surrogate model (Compute stats for neutral prompt)
    mutated_emb1_norm = np.linalg.norm(mutated_emb1)
    mutated_emb1_mean = mutated_emb1.mean()
    mutated_emb1_std = mutated_emb1.std()
    mutated_emb1_max = mutated_emb1.max()
    mutated_emb1_min = mutated_emb1.min()

    if mutated_emb2 is not None:
        mutated_emb2_norm = np.linalg.norm(mutated_emb2)
        mutated_emb2_mean = mutated_emb2.mean()
        mutated_emb2_std = mutated_emb2.std()
        mutated_emb2_max = mutated_emb2.max()
        mutated_emb2_min = mutated_emb2.min()
    else: 
        mutated_emb2_norm = mutated_emb2_mean = mutated_emb2_std = mutated_emb2_max = mutated_emb2_min = 0.0

    # Compute cosine similarities 
    emb1_male, emb2_male = embeddings[male]
    emb1_female, emb2_female = embeddings[female]

    cos1_male = cosine_similarity(mutated_emb1.reshape(1, -1), emb1_male.reshape(1, -1))[0, 0]
    cos1_female = cosine_similarity(mutated_emb1.reshape(1, -1), emb1_female.reshape(1, -1))[0, 0]
    diff_cos1 = cos1_male - cos1_female

    if mutated_emb2 is not None and emb2_male is not None and emb2_female is not None:
        cos2_male = cosine_similarity(mutated_emb2.reshape(1, -1), emb2_male.reshape(1, -1))[0, 0]
        cos2_female = cosine_similarity(mutated_emb2.reshape(1, -1), emb2_female.reshape(1, -1))[0, 0]
        diff_cos2 = cos2_male - cos2_female
    else:
        cos2_male = cos2_female = diff_cos2 = 0.0
    

    features = np.concatenate([
        mutated_emb1, 
        mutated_emb2 if mutated_emb2 is not None else np.zeros_like(mutated_emb1),
        np.array([
            mutated_emb1_norm, mutated_emb2_norm,
            mutated_emb1_mean, mutated_emb2_mean,
            mutated_emb1_std, mutated_emb2_std,
            mutated_emb1_max, mutated_emb2_max,
            mutated_emb1_min, mutated_emb2_min,
            diff_cos1, diff_cos2
        ])
    ]).reshape(1, -1)

    # print("Calling surrogate model for inference with features:", features.shape)

    # Step 4: Predict gender bias
    np.set_printoptions(threshold=np.inf)
    gender_bias = xgb_pipe.predict(features)[0]

    return gender_bias



print('!!! JUST BEFORE GAO')
class GAOptimizer:
    """
    Main class for the genetic optimiser.
    The population is a list of individuals.
    Each individual is a list of floats between 0 and 1, of the same length as the prompt embeddings, which is a multiplier of the prompt embeddings.
    The fitness function evaluates the gender bias using the surrogate model and the image quality using a similarity with the original embeddings.

    TODO: the individual can be a list of float values, each one is a multiplier of a specific part of the prompt embeddings.

    """

    def __init__(self, attributes={}) -> None:
        self.number_of_generations = int(attributes["number_of_generations"])
        self.mutation_probability = float(attributes["mutation_probability"])
        self.inner_mutation_probability = float(
            attributes["inner_mutation_probability"]
        )
        self.population_size = int(attributes["population_size"])
        self.crossover_probability = float(attributes["crossover_probability"])
        self.img_num = attributes["img_num"]
        self.mu = attributes["mu"]
        self.lambda_ = attributes["lambda"]
        self.prompt = attributes["prompt"]
        self.folder_name = attributes["folder_name"]
        self.round = attributes["round"]
        self.fitness = attributes["fitness"]
        self.genome_length = int(attributes.get("genome_length", 1))
        self.logger = logging.getLogger(__name__)
        self.setup_deap()
    
    # After image generation
    def eval_fitness(self, individual):
        """
        Function to evaluate the fitness of an individual.
        It collects the embeddings for a given prompt and compute the gender bias fitness and the image quality fitness.
        
        :param self: Descrizione
        :param individual: Descrizione
        """

        self.logger.info(f"Generating Images for: \\n {individual}")

        gender_bias = collect_embeddings_and_evaluate_fitness(self.prompt, individual)

        self.logger.info(
            f"Individual: {individual} \n  Gender Fitness: {gender_bias}"
        )

        # For saving fitness values to csv files
        fieldnames = [
            "Individual",
            "Gender Fitness"
        ]

        row_dict = {
            "Individual": json.dumps(individual),
            "Gender Fitness": gender_bias
        }

        csv_file = f"{self.folder_name}/fitness.csv"
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write the header if the file is empty
            if file.tell() == 0:
                writer.writeheader()

            # Write the row
            writer.writerow(row_dict)
  
        return gender_bias,

    def crossover(self, parent1, parent2):
        """
        Function to perform one-point crossover between two parents.
        It creates two children by combining the genes of the parents.
        
        :param parent1: the first parent
        :param parent2: the second parent
        """
        child1 = copy.deepcopy(parent1)
        child2 = copy.deepcopy(parent2)
  
        self.logger.info("Performing crossover")
        max_length = min(len(parent1), len(parent2))
        i = random.randint(1, max_length - 1)
        child1[:i], child2[:i] = parent2[:i], parent1[:i]

        return child1, child2
    
    def mutate(self, individual, mu, sigma):
        """
        Function to perform mutation on an individual.
        It randomly changes one gene of the individual with a certain probability.
        
        :param individual: the individual to mutate
        :param mu: the mean of the Gaussian noise
        :param sigma: the standard deviation of the Gaussian noise
        """

        self.logger.info("Performing mutation")
        new_individual = copy.deepcopy(individual)
        for i in range(len(new_individual)):
                new_individual[i] += random.gauss(mu, sigma)  # Add Gaussian noise
                new_individual[i] = min(max(new_individual[i], 0), 1)  # Keep within [0, 1]
        return new_individual,

    def setup_deap(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", random.uniform, 0.1, 1)
        self.toolbox.register(
            "individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=self.genome_length
        )
        self.toolbox.register(
            "population", tools.initRepeat, list, self.toolbox.individual, n=self.population_size
        )

        self.toolbox.register("evaluate", self.eval_fitness)
        self.toolbox.register("mate", self.crossover)
        self.toolbox.register("mutate", self.mutate, mu=0.0, sigma=1.0)
        self.toolbox.register("select", tools.selTournament, tournsize=5)

    def optimization(self):
        # collect statistsics for the individuals in population
        hof = tools.ParetoFront()
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("avg", numpy.mean, axis=0)
        stats.register("std", numpy.std, axis=0)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        population = self.toolbox.population()

        self.logger.info("Running GAO ...")
        _, logbook = algorithms.eaMuCommaLambda(
            population,
            self.toolbox,
            mu=self.mu,
            lambda_=self.lambda_,
            cxpb=self.crossover_probability,
            mutpb=self.mutation_probability,
            ngen=self.number_of_generations,
            stats=stats,
            halloffame=hof
        )

        # Select the best individual
        best = tools.selBest(population, k=1)
        return best[0], hof, logbook
