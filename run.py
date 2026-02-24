"""
Script runs evolutionary process.
"""


import os
import yaml
from aesthetic_evolution.generate_designs import aesthetic_evolution

# check confif file exists
assert os.path.exists("config/experiment_config.yaml"), "Config file 'experiment_config.yaml' not found."

# load config file
with open("config/experiment_config.yaml", 'r') as file:
    configuration = yaml.safe_load(file)

#### LOAD JOB CONFIGURATION #####

job_config = configuration.get('job', None)
assert job_config is not None, "Job configuration not detected, validate format for experiment_config.yaml."

# check experiment name does not already exist, let user know and ask if they want to overwrite
experiment_name = job_config['experiment_name']
if os.path.exists(os.path.join("Experiments", experiment_name)):
    response = input(f"Experiment '{experiment_name}' already exists. Do you want to overwrite it? (y/n): ")
    if response.lower() != 'y':
        print("\nExiting without overwriting. Please choose a different experiment name.")
        exit(0)

# load prompt from file if specified
prompt_file = job_config.get('prompt_filepath', None)
assert prompt_file is not None, "Prompt file not specified in job configuration."
assert os.path.exists(prompt_file), f"Prompt file '{prompt_file}' not found."
with open(prompt_file, 'r') as file:
    prompt = file.read()


#### LOAD EVOLUTION CONFIGURATION #####
evo_config = configuration.get('evo', None)
assert evo_config is not None, "Evolution configuration not detected, validate format for experiment_config.yaml."

# validate alpha values if present
assert evo_config["alpha_mode"] in ["random", "fixed", "biased"], "Alpha mode must be one of 'random', 'fixed', or 'biased'."
if evo_config["alpha_mode"] == "fixed":
    alpha = evo_config.get("alpha", None)
    assert alpha is not None, "Alpha value must be provided for fixed alpha mode."
    assert 0 <= alpha <= 1, "Alpha must be between 0 and 1."

# retrieve check mutation rate
mutation_rate = evo_config.get("mutation_rate", 0.1)
assert 0 <= mutation_rate <= 1, "Mutation rate must be between 0 and 1."

# check mutation sigma
mutation_sigma = evo_config.get("mutation_sigma", 0.1)
assert mutation_sigma > 0, "Mutation sigma must be positive."

# retrieve and check competing parents parameters
parents_compete = evo_config.get("parents_compete", None)
competing_parents_rate = evo_config.get("competing_parents_rate", None)
assert parents_compete is not None, "Parents compete parameter must be specified in evolution configuration."
if parents_compete:
    assert competing_parents_rate is not None, "Competing parents rate must be specified if parents compete is True."
    assert 0 <= competing_parents_rate <= 1, "Competing parents rate must be between 0 and 1."

# check ranking method
ranking_method = evo_config.get('ranking_method', None)
assert ranking_method in ['glicko', 'simple', "CLIP-IQA"], f"Ranking method must be either 'glicko', 'simple', or 'CLIP-IQA', but got '{ranking_method}'."


##### START EXPERIMENT #####

print(f"Starting experiment '{experiment_name}' with configuration:")
for key, value in job_config.items():
    print(f"  {key}: {value}")

# run aesthetic evolution with loaded configuration
aesthetic_evolution(
    experiment_name=job_config['experiment_name'],
    runs=evo_config['runs'],
    param_spec_filepath=job_config['param_spec_file'],
    sketch_dir=job_config['sketch_dir'],
    prompt=prompt,
    alpha_mode=evo_config['alpha_mode'],
    alpha=evo_config["alpha"],
    mutation_rate=mutation_rate,
    mutation_sigma=mutation_sigma,
    k=evo_config["k"],
    ranking_method=ranking_method,
    population_size=evo_config["population_size"],
    processing=job_config["processing"],
    screen=job_config.get('screen', False),
    workers=job_config.get('workers', 8),
    parents_compete=parents_compete,
    competing_parents_rate=competing_parents_rate
)

# save configuration to experiment directory
experiment_dir = os.path.join("Experiments", experiment_name)

with open(os.path.join(experiment_dir, "experiment_config.yaml"), 'w') as file:
    yaml.dump(configuration, file)