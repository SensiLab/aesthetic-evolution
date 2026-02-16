"""
Script generates random designs.
"""

import os
import yaml
from aesthetic_evolution.generate_designs import DesignEvolver, DesignGenerator


# initialise directories
if not os.path.exists("Data"):
    os.makedirs("Data")
if not os.path.exists("Data/benchmark"):
    os.makedirs("Data/benchmark")
if not os.path.exists("Data/benchmark/Params"):
    os.makedirs("Data/benchmark/Params")
if not os.path.exists("Data/benchmark/Images"):
    os.makedirs("Data/benchmark/Images")


# load generation config
generation_config_path = "config/generation_config.yaml"
assert os.path.exists(generation_config_path), f"Generation config file not found at {generation_config_path}"

with open(generation_config_path, 'r') as file:
    generation_config = yaml.safe_load(file)

# load param spec
param_spec_filepath = generation_config.get("param_spec_file", None)
assert param_spec_filepath is not None, "Parameter specification file path not specified in generation config."
assert os.path.exists(param_spec_filepath), f"Parameter specification file not found at {param_spec_filepath}"

spec_file = DesignEvolver.read_param_spec(param_spec_filepath)

# generate params
population = DesignEvolver.generate_initial_params(generation_config["population_size"],
                                                  spec_file,
                                                  "generation_experiment",
                                                  mutation_sigma=0.1) # mutation rate irrelevant for initial generation, but required for function signature



# initialise generator
generator = DesignGenerator(experiment_name="generation_experiment",
                            sketch_dir=generation_config["sketch_dir"],
                            processing=generation_config["processing"],
                            screen=generation_config["screen"],
                            workers=generation_config["workers"])

# generate designs
generator.generate_population(pop_name="benchmark", 
                              params=population, 
                              base_filepath="Data/benchmark")



