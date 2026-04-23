'''
This script gathers all optimal solutions from the runs produced by the GA algorithm.
'''
import os
import re
import csv
import ast
import glob
from typing import List, Tuple, Optional

def parse_results_csv(results_csv_path: str) -> Tuple[List[List[float]], List[float]]:
    with open(results_csv_path, "r", newline="") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) < 2:
        raise ValueError(f"{results_csv_path} does not contain two non-empty rows")

    individuals_line = lines[0]
    fitnesses_line = lines[1]

    individual_strings = re.findall(r'"(\[.*?\])"', individuals_line)
    if not individual_strings:
        individual_strings = re.findall(r'(\[.*?\])', individuals_line)

    individuals = [ast.literal_eval(ind_str) for ind_str in individual_strings]

    fitness_matches = re.findall(r'np\.float32\(([-+eE0-9\.]+)\)', fitnesses_line)

    if fitness_matches:
        fitnesses = [float(x) for x in fitness_matches]
    else:
        tuple_matches = re.findall(r'\(([-+eE0-9\.]+),?\)', fitnesses_line)
        if tuple_matches:
            fitnesses = [float(x) for x in tuple_matches]
        else:
            parsed = next(csv.reader([fitnesses_line]))
            fitnesses = []
            for item in parsed:
                item = item.strip()
                item = re.sub(r'np\.float32\(([-+eE0-9\.]+)\)', r'\1', item)
                val = ast.literal_eval(item)
                if isinstance(val, tuple):
                    fitnesses.append(float(val[0]))
                else:
                    fitnesses.append(float(val))

    if len(individuals) != len(fitnesses):
        raise ValueError(
            f"Mismatch in {results_csv_path}: "
            f"{len(individuals)} individuals vs {len(fitnesses)} fitnesses"
        )

    return individuals, fitnesses


def get_round_number(path: str) -> Optional[int]:
    match = re.search(r'round_(\d+)', path)
    return int(match.group(1)) if match else None


def collect_best_solutions(
    base_dir: str,
    output_csv: str
) -> None:
    pattern = os.path.join(base_dir, "round_*", "results.csv")
    result_files = sorted(glob.glob(pattern), key=lambda p: get_round_number(p) or 10**9)

    if not result_files:
        raise FileNotFoundError(f"No files found matching: {pattern}")

    rows = []

    for results_path in result_files:
        round_num = get_round_number(results_path)
        individuals, fitnesses = parse_results_csv(results_path)

        best_idx = min(range(len(fitnesses)), key=lambda i: fitnesses[i])
        best_individual = individuals[best_idx]
        best_fitness = fitnesses[best_idx]

        rows.append({
            "round": round_num,
            "fitness": best_fitness,
            "individual": repr(best_individual),
        })

        print(
            f"Round {round_num}: selected archived solution "
            f"{best_idx + 1}/{len(individuals)} with fitness {best_fitness:.8f}"
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "round",
                "fitness",
                "individual",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {len(rows)} best-per-round solutions to: {output_csv}")


if __name__ == "__main__":
    # Fix directories for your setup if reproducing results 
    # BASE_DIR is directory with results of 50 runs produced by moea/main.py
    BASE_DIR = "3_Bias_Mitigation/exp1_results/comparison/software_eng_prompt2" 
    OUTPUT_CSV = "swe_prompt_optimal_solutions_from_results.csv"
    collect_best_solutions(BASE_DIR, OUTPUT_CSV)