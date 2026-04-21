# Bias Mitigation via Genetic Algorithm (GA) on Prompt Embeddings

This folder contains the source code for FairEmbed. FairEmbed is a surrogate-assissted GA algorithm that optimises the prompt embeddings of Stable Diffusion 3 to minimise gender bias. It utilises a pre-trained XGBoost surrgoate model (`LATEST_final_xgb_full_data_pipeline.joblib`).


## Files

| File | Description |
|------|-------------|
| `main.py` | Entry point — configures and runs the GA, saves results |
| `gao.py` | Core `GAOptimizer` class using DEAP (`eaMuCommaLambda`), and the surrogate model |
| `objectives.py` | `Fitness` enum defining supported search strategies |
| `LATEST_final_xgb_full_data_pipeline.joblib` | Pre-trained XGBoost surrogate pipeline for gender bias prediction |

## Usage

```bash
python main.py \
  --num_gen <num_gen> \
  --pop_size <pop_size> \
  --imgs_to_generate <imgs_to_generate> \
  --round <round> \
  --fitness <fitness>
```

### Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--num_gen` | Number of generations | 30 |
| `--pop_size` | Population size | 50 |
| `--imgs_to_generate` | Number of images to generate (post-optimisation) | 20 |
| `--round` | Experiment round number (used for output folder) | 1 |
| `--fitness` | Search strategy: `nsgaii`, `nsgaiii`, or `weight` | `nsgaii` |


## Output

Results are saved to `results/round_<round>/`:

| File | Contents |
|------|----------|
| `logbook.csv` | Per-generation statistics (avg, std, min, max fitness) |
| `results.csv` | Full Pareto front — individuals and their fitness values |
| `best_individual.csv` | Best individual genome and fitness |
| `timing.csv` | Total optimisation time in seconds |
| `fitness.csv` | Per-evaluation log of individual and gender bias score |
| `main.log` | Full run log |

## Dependencies

- `torch`, `diffusers` (Stable Diffusion 3 — `stabilityai/stable-diffusion-3-medium-diffusers`)
- `deap` (genetic algorithm framework)
- `joblib`, `scikit-learn` (surrogate model)
- `pandas`, `numpy`, `Pillow`

> **Note:** SD3 is loaded on CPU by default (`torch.float32`). To use CUDA, uncomment the relevant lines in `gao.py` and update `device`.

> **Note:** The current experiment also uses the fixed prompt: `"A photo of one real person who is a banker"`.

