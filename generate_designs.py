
import os
import json
import time
import math
import yaml
import random
import subprocess
from dataclasses import dataclass
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple
from utils import build_messages

# currently not in use, would be good to fix types
@dataclass()
class Params:
    steps: int = 25
    sampleRate: float = 50.0
    ab: int = 300
    ampSkew: float = 0.0
    fb: int = 2
    frx: float = 1.0
    fry: float = 1.0
    px: float = 0.0
    py: float = math.pi / 2
    dx: float = 0.1
    dy: float = 0.1
    dt: float = 1.0/(2 * 1.0 * 50.0)
    radial: int = 0
    no: int = 1
    lac: float = 2.0
    fo: float = 0.5
    mix: float = 0.1
    spaceD: float = 0.0
    spaceP: float = 1
    algorithm: str = "SINE"

    def __getitem__(self, item):
        """Allows accessing fields using subscript notation (e.g., product['name'])."""
        return getattr(self, item)

def write_json(params: dict,
               filepath: str,
               filename: str,
               population_name: str) -> None:
    """
    Function writes parameters to a JSON file.
    
    :param params: parameters in dictionary format
    :type params: dict
    :param filepath: Path to the directory where the JSON file will be saved
    :type filepath: str
    :param filename: Name of the JSON file to be created
    :type filename: str
    :param population_name: Name of the population being generated
    :type population_name: str
    @author: Stephen Krol
    @date: Jan 2026

    :returns: None
    :rtype: None
    """

    full_path = f"{filepath}/{filename}"

    with open(f'{full_path}.json', 'w') as f:
        json.dump({"parameters": params, 
                   "filename": filename, 
                   "population_name": population_name,
                   "base" : os.getcwd()}, f, indent=4)


def generate_rand_params(config: dict) -> dict:
    """
    Function generates random parameters for sample.
    @author: Stephen Krol
    @date: Jan 2026
    
    :param config: A dictionary defining parameter ranges and types.
    :type config: dict

    :return: Dictionary of generated random parameters.
    :rtype: dict
    """

    params = {}

    # sample params using ranges from config
    for param in config:
        param_config = config[param]
        if param_config["type"] == "int":
            sample = random.randint
        else:
            sample = random.uniform

        params[param] = sample(param_config["min"], param_config["max"])

    # calculate remaining params
    params["dt"] = 1.0 / (params["fb"] * max(params["frx"], params["fry"] * params["sampleRate"]))

    # spaceP
    params["spaceP"] = 10**params["spaceD"]

    # algorithm
    params["algorithm"] = random.choice(["SINE", "POW", "SINEOSC", "NOISE", "FRACTAL", "OSC", "TURB", "PHASE"])

    # radial set to 0
    params["radial"] = 0
    

    return params

def read_param_spec(spec_filepath: str) -> dict:
    """
    Function reads parameter specification from a YAML file.
    @author: Stephen Krol
    @date: Jan 2026
    
    :param spec_filepath: Filepath to the YAML specification file.
    :type spec_filepath: str

    :return: Parameter specification as a dictionary.
    :rtype: dict
    """

    # Use a context manager to open and automatically close the file
    with open(spec_filepath, 'r') as file:
        # Use safe_load to safely parse the YAML content
        configuration = yaml.safe_load(file)
    
    return configuration


def initialise() -> None:
    """
    Function to initialise the Designs directory structure.
    @author: Stephen Krol
    @date: Jan 2026
    """

    if not os.path.exists("Designs"):
        os.mkdir("Designs")

    if not os.path.exists("Designs/Images"):
        os.mkdir("Designs/Images")
    
    if not os.path.exists("Designs/Params"):
        os.mkdir("Designs/Params")

def initialise_design(name: str):
    """
    Function initialises directories for a specific design population.
    @author: Stephen Krol
    @date: Jan 2026
    
    :param name: Name of population
    :type name: str
    """

    if not os.path.exists(f"Designs/Images/{name}"):
        os.mkdir(f"Designs/Images/{name}")
    
    if not os.path.exists(f"Designs/Params/{name}"):
        os.mkdir(f"Designs/Params/{name}")

def generate_image(jobs: Tuple[str, str, str, bool]) -> None:
    """
    Generate image calls processing to generate an image from parameters.
    Arguments are passed as a tuple for compatibility with ProcessPoolExecutor.
    @author: Stephen Krol
    @date: Jan 2026
    
    :param jobs: A tuple containing the arguments for generation. 
        Arg1 is population name, Arg2 is filename, Arg3 is sketch directory,
        Arg4 is screen boolean.
    :type jobs: Tuple[str, str, str, bool]

    :return: None
    :rtype: None
    """

    population_name, filename, sketch_dir, screen = jobs

    cwd = os.getcwd()

    # if screen is available
    if screen:
        subprocess.run([
            "processing-java",
            f"--sketch={sketch_dir}",
            "--run",
            f"{cwd}/Designs/Params/{population_name}/{filename}.json"], 
            check=True,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)
    else:
        subprocess.run([
            "xvfb-run", "-a",
            "--server-args=-screen 0 1024x768x24 -nolisten tcp",
            "processing-java",
            f"--sketch={sketch_dir}",
            "--run",
            f"{cwd}/Designs/Params/{population_name}/{filename}.json"], 
            check=True,
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

def generate_population(n: int,
                        name: str,
                        config: dict,
                        sketch_dir: str, 
                        processing: str="serial",
                        screen: bool=False,
                        workers: int=8) -> list:
    """
    Function generates a population of designs.
    @author: Stephen Krol
    @date: Jan 2026
    
    :param n: Number of designs to generate.
    :type n: int
    :param name: Name of the population.
    :type name: str
    :param config: Configuration parameters for the generation.
    :type config: dict
    :param sketch_dir: Directory containing the Processing sketch.
    :type sketch_dir: str
    :param processing: Either 'serial' or 'parallel' processing.
    :type processing: str
    :param screen: Whether processing has access to a display.
    :type screen: bool
    :param workers: Number of workers for parallel processing.
    :type workers: int
    
    :return: List of generated image filenames.
    :rtype: list
    """

    initialise_design(name)

    start = time.time()
    jobs = []

    # write JSON
    for i in tqdm(range(n)):
        params = generate_rand_params(config)
        filename = f"{name}_{i}"
        write_json(params, f"Designs/Params/{name}", f"{filename}", name)
        jobs.append((name, filename, sketch_dir, screen))

    # generate images serially
    if processing == "serial":
        for job in tqdm(jobs):
            generate_image(job)
    elif processing == "parallel":
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pool.map(generate_image, jobs)
    else:
        raise ValueError("Processing must be 'serial' or 'parallel'")
    
    end = time.time()

    print(f"Took {end - start:.2f} seconds to generate {n} designs.")


    return os.listdir(f"Designs/Images/{name}")

if __name__ == "__main__":


    # Example usage
    initialise()

    config = read_param_spec("param_spec.yaml")

    # for _ in range(10):
    #     generate_population(n=10, name="test",config=config)

    # generate_population(n=20,
    #                     name="test",
    #                     config=config,
    #                     sketch_dir="/home/sjkro1/ARC-Discovery/Harmonograph",
    #                     processing="parallel",
    #                     screen=False,
    #                     workers=8)

    # filename = "design0"
    # initialise_design(filename)

    # params = dict(
    #     steps=21,
    #     sampleRate=278.17144775390625,
    #     ab=300,
    #     ampSkew=0.0,
    #     fb=2.0,
    #     frx=1.0,
    #     fry=1.9776785373687744,
    #     px=-2.5132739543914795,
    #     py=1.3015170097351074,
    #     dx=0.10000000149011612,
    #     dy=0.10000000149011612,
    #     dt=9.088699007406831E-4,
    #     radial=0,
    #     no=1,
    #     lac=2,
    #     fo=0.48571428656578064,
    #     mix=0.39642858505249023,
    #     spaceD=0.014285683631896973,
    #     spaceP=1.0334409475326538,
    #     algorithm="SINE"
    # )


    # config = read_param_spec("param_spec.yaml")

    # params = generate_rand_params(config)

    # write_json(params, f"Designs/Params/{filename}", filename)
    # generate_image(filename)