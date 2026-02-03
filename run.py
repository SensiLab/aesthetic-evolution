"""
Script runs evolutionary process.
"""


import os
import yaml
from generate_designs import aesthetic_evolution

# check confif file exists
assert os.path.exists("experiment_config.yaml"), "Config file 'experiment_config.yaml' not found."

# load config file
with open("experiment_config.yaml", 'r') as file:
    configuration = yaml.safe_load(file)

# check experiment name does not already exist, let user know and ask if they want to overwrite
experiment_name = configuration['experiment_name']

if os.path.exists(os.path.join("Experiments", experiment_name)):

    response = input(f"Experiment '{experiment_name}' already exists. Do you want to overwrite it? (y/n): ")
    if response.lower() != 'y':
        print("Exiting without overwriting. Please choose a different experiment name.")
        exit(0)

print(f"Starting experiment '{experiment_name}' with configuration:")
for key, value in configuration.items():
    print(f"  {key}: {value}")

# run aesthetic evolution with loaded configuration
aesthetic_evolution(
    experiment_name=configuration['experiment_name'],
    runs=configuration['runs'],
    param_spec_filepath=configuration['param_spec_filepath'],
    sketch_dir=configuration['sketch_dir'],
    prompt=configuration['prompt'],
    population_size=configuration.get('population_size', 20),
    processing=configuration.get('processing', 'serial'),
    screen=configuration.get('screen', False),
    workers=configuration.get('workers', 8)
)

# save configuration to experiment directory
experiment_dir = os.path.join("Experiments", experiment_name)

with open(os.path.join(experiment_dir, "experiment_config.yaml"), 'w') as file:
    yaml.dump(configuration, file)