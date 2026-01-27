
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


class DesignGenerator:

    def __init__(self, 
                 spec_filepath: str,
                 sketch_dir: str,
                 processing: str = "serial",
                 screen: bool = False,
                 workers: int = 8) -> None:
        """
        Constructor for DesignGenerator class.
        
        :param self: Current instance of the class. 
        :param spec_filepath: Description
        :type spec_filepath: str
        :param sketch_dir: Directory containing the Processing sketch.
        :type sketch_dir: str
        :param processing: Either 'serial' or 'parallel' processing.
        :type processing: str
        :param screen: Whether processing has access to a display.
        :type screen: bool
        :param workers: Number of workers for parallel processing.
        :type workers: int

        :return: None
        :rtype: None
        """

        # initialise folder structure
        self._initialise()

        # read param spec
        self.param_spec = self._read_param_spec(spec_filepath)

        # set processing directory
        self.sketch_dir = sketch_dir

        # set processing type
        assert processing in ["serial", "parallel"], "Processing must be 'serial' or 'parallel'"
        self.processing = processing

        #  set screen boolean
        assert isinstance(screen, bool), "Screen must be a boolean value"
        self.screen = screen

        # set number of workers
        assert isinstance(workers, int) and workers > 0, "Workers must be a positive integer"
        self.workers = workers

    def generate_population(self, n: int, name: str) -> list:
        """
        Method generates a population of designs.
        @author: Stephen Krol
        @date: Jan 2026
        
        :param self: Current instance of the class.
        :param n: Number of designs to generate.
        :type n: int
        :param name: Name of the population.
        :type name: str

        :return: List of generated image filenames.
        :rtype: list
        """

        # initialise design directories
        self._initialise_design(name)

        start = time.time()
        jobs = []

        # write JSON parameters for processing
        for i in tqdm(range(n)):
            params = self._generate_initial_params(self.param_spec)
            filename = f"{name}_{i}"
            self._write_json(params, f"Designs/Params/{name}", f"{filename}", name)
            jobs.append((name, filename, self.sketch_dir, self.screen))
        
        if self.processing == "serial": # generate images serially
            for job in tqdm(jobs):
                self.generate_image(job)

        else: # generate images in parallel
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                pool.map(self.generate_image, jobs)
        
        end = time.time()

        print(f"Took {end - start:.2f} seconds to generate {n} designs.")

        return os.listdir(f"Designs/Images/{name}")

    @staticmethod
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
                timeout=20,
                check=True,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)

    def _initialise(self) -> None:
        """
        Method to initialise the Designs directory structure.
        @author: Stephen Krol
        @date: Jan 2026

        :param self: Current instance of the class.

        :return: None
        :rtype: None
        """

        if not os.path.exists("Designs"):
            os.mkdir("Designs")

        if not os.path.exists("Designs/Images"):
            os.mkdir("Designs/Images")
        
        if not os.path.exists("Designs/Params"):
            os.mkdir("Designs/Params")
    
    def _initialise_design(self, name: str):
        """
        Method initialises directories for a specific design population.
        @author: Stephen Krol
        @date: Jan 2026
        
        :param self: Current instance of the class.
        :param name: Name of population
        :type name: str

        :return: None
        :rtype: None
        """

        if not os.path.exists(f"Designs/Images/{name}"):
            os.mkdir(f"Designs/Images/{name}")
        
        if not os.path.exists(f"Designs/Params/{name}"):
            os.mkdir(f"Designs/Params/{name}")

    def _read_param_spec(self, spec_filepath: str) -> dict:
        """
        Method reads parameter specification from a YAML file.
        @author: Stephen Krol
        @date: Jan 2026
        
        :param self: Current instance of the class.
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
    

    def _generate_initial_params(self, config: dict) -> dict:
        """
        Method generates inital random parameters for sample.
        @author: Stephen Krol
        @date: Jan 2026
        
        :param self: Current instance of the class.
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

    def _write_json(self, 
                   params: dict,
                   filepath: str,
                   filename: str,
                   population_name: str) -> None:
        """
        Method writes parameters to a JSON file.
        @author: Stephen Krol
        @date: Jan 2026
        
        :param self: Current instance of the class.
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


# def evaluate_population(filenames: list,
#                         root: str,
#                         prompt: str) -> list:

#     jobs = build_messages(filenames,
#                           root,
#                           prompt)

#     return

if __name__ == "__main__":


    # Example usage
    generator = DesignGenerator(
        spec_filepath="param_spec.yaml",
        sketch_dir="/home/sjkro1/ARC-Discovery/Harmonograph",
        processing="parallel",
        screen=False,
        workers=8
    )

    designs = generator.generate_population(n=20, name="test2")
    
    # evaluate_population(designs,)

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