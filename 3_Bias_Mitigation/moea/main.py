from gao import GAOptimizer
from argparse import ArgumentParser
import os
from objectives import Fitness
import pandas as pd
import csv
import time
import logging
import numpy as np
import torch
from gao import pipe  


"""
This script runs a GA-based optimisation for bias mitigation experiments. 
Configures and executes the evolutionary process, logs results, and saves experiment 
outputs including best individuals, Pareto front, and timing information.
"""


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger()
    parser = ArgumentParser()
    parser.add_argument("--num_gen", type=int, default=30) 
    parser.add_argument("--pop_size", type=int, default=50)
    parser.add_argument("--imgs_to_generate", type=int, default=20)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument('--fitness', type=str, choices=[
        Fitness.NSGAII.value,
        Fitness.NSGAIII.value,
        Fitness.WEIGHT.value
    ], default=Fitness.NSGAII.value)

    args = parser.parse_args()
    folder_name = f"results/round_{args.round}" 
    os.makedirs(folder_name, exist_ok=True)

    log_path = os.path.join(folder_name, "main.log")
    if not any(
        isinstance(handler, logging.FileHandler)
        and handler.baseFilename == log_path
        for handler in logger.handlers
    ):
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    logger.info("Starting the experiment")

    # Chosen prompt for our experiment is the banker prompt
    prompt = "A photo of one real person who is a banker"

    # Currently using the optimum configuration 3 (mutation-heavy mixed setting)
    attributes = {
        "number_of_generations": args.num_gen,
        "mutation_probability": 0.8, 
        "inner_mutation_probability": 0.2,
        "population_size": args.pop_size,
        "selection_size": 5,
        "crossover_probability": 0.2,
        "img_num": args.imgs_to_generate,
        "mu": 5,
        "lambda": 5,
        "prompt": prompt,
        "round": args.round,
        "folder_name": folder_name,
        "fitness": args.fitness,
        "genome_length": 2048,  
    }

    ga = GAOptimizer(attributes)
    logger.info(f"Beginning GAO round {args.round}")
    # To record time taken to reach optimum
    start = time.time()
    best_individual, pareto_front, logbook = ga.optimization()
    end = time.time()

    logger.info(f"Best Individual:, {best_individual}")
    logger.info(f"Best Fitness:, {best_individual.fitness.values}")
    logger.info(f"Offspring:, {pareto_front}")
    logger.info(f"Logbook:, {logbook}")
    logger.info(f"Time taken: {end - start} seconds")

    # Write the lobgook to a CSV file
    logdf = pd.DataFrame(logbook)
    logdf.to_csv(f"{folder_name}/logbook.csv")

    with open(f"{folder_name}/results.csv", "a", newline="") as csvfile:
        # Create a CSV writer object
        writer = csv.writer(csvfile)
        # Write the dictionary string to the CSV file
        writer.writerow([ind for ind in pareto_front])
        writer.writerow([ind.fitness.values for ind in pareto_front])
    
    with open(f"{folder_name}/best_individual.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["individual", "fitness"])
        writer.writerow([list(best_individual), list(best_individual.fitness.values)])
    
    with open(f"{folder_name}/timing.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time_seconds"])
        writer.writerow([end - start])

    # Optionally, generate the image (we do not do this for our study)
    # image_path = generate_mutated_image(prompt, best_individual, folder_name)
    # logger.info("Saved mutated image to %s", image_path)

    logger.info("Done")
