

from fileinput import filename
import json
import math
import numpy as np
import os
import random
import subprocess
import time
import torch
import yaml

from utils import build_messages, calc_ranks, plot_image_grid
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from process_batch import Qwen3VLBatchProcessor
from typing import List, Tuple
from tqdm import tqdm


class Params:

    def __init__(self, 
                 parameters: dict, 
                 params_config: dict,
                 experiment_name: str, 
                 name: str) -> None:

        # check filepath exists
        self.params = parameters
        self.params_config = params_config

        # assign name and experiment
        self.name = name
        self.experiment_name = experiment_name

        # categorise parameters by type
        self.float_params = [param for param in self.params_config if self.params_config[param]["type"] == "float"]
        self.int_params = [param for param in self.params_config if self.params_config[param]["type"] == "int"]
        self.categorical_params = [param for param in self.params_config if self.params_config[param]["type"] == "categorical"]

    # TODO: could be a static method
    def breed(self, other: "Params", alpha: float, child_name: str) -> dict:

        assert 0 <= alpha <= 1, "Alpha must be between 0 and 1"

        # TODO: arthimitic mean currently does not consider ranking between parents,
        # making it harder to weight higher ranked parents more heavily.
        child = Params({}, self.params_config, self.experiment_name, child_name)

        # calculate arthimitic mean of float parameters
        for param in self.float_params:
            child.params[param] = alpha * self.params[param] + (1 - alpha) * other.params[param]
        
        # calculate arthimitic mean of int parameters and round
        for param in self.int_params:
            child.params[param] = round(alpha * self.params[param] + (1 - alpha) * other.params[param])
        
        # TODO: option to not cross over categorical parameters and instead inherit from one parent
        # randomly select categorical parameters from either parent
        for param in self.categorical_params:
            child.params[param] = random.choice([self.params[param], other.params[param]])

        # calculate dependent parameters
        child.params["dt"] = self._calculate_dt(child.params)
        child.params["spaceP"] = self._calculate_spaceP(child.params)
        child.params["radial"] = 0

        return child

    def mutate(self, mutation_rate: float) -> None:

        # float and int mutation
        for param in (self.float_params + self.int_params):

            if param in ["dt", "spaceP"]:
                continue

            if random.random() < mutation_rate:
                
                # gaussian mutation
                a, b = self.params_config[param]["min"], self.params_config[param]["max"]
                z = (self.params[param] - a) / (b - a)
                z += random.gauss(0, 0.1)
                z = min(max(z, 0), 1)
                x = a + z * (b - a)

                if self.params_config[param]["type"] == "int":
                    self.params[param] = round(x)
                else:   
                    self.params[param] = x
        
        # categorical mutation
        for param in self.categorical_params:
            if random.random() < mutation_rate:
                self.params[param] = random.choice(self.params_config[param]["values"])

        # recalculate dependent parameters
        self.params["dt"] = self._calculate_dt(self.params)
        self.params["spaceP"] = self._calculate_spaceP(self.params)


    def write_json(self, 
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
        image_path = f"{os.getcwd()}/Experiments/{self.experiment_name}/{population_name}/Images/{filename}.png"

        with open(f'{full_path}.json', 'w') as f:
            json.dump({"parameters": self.params, 
                       "filepath": image_path}, f, indent=4)
    
    def print_params(self) -> None:
        """
        Method prints the parameters to the console.
        """
        for key, value in self.params.items():
            print(f"{key}: {value}")

    def _calculate_dt(self, params) -> float:
        return 1.0 / (params["fb"] * max(params["frx"], params["fry"] * params["sampleRate"]))
    
    def _calculate_spaceP(self, params) -> float:
        return 10**params["spaceD"]

class DesignGenerator:
    """
    DesignGenerator class to generate design images using Processing. Can generate populations
    in both serial and parallel modes.
    """

    def __init__(self, 
                 experiment_name: str,
                 sketch_dir: str,
                 processing: str = "serial",
                 screen: bool = False,
                 workers: int = 8) -> None:
        """
        Constructor for DesignGenerator class.
        
        :param self: Current instance of the class. 
        :param experiment_name: Name of the experiment.
        :type experiment_name: str
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
        self.experiment_name = experiment_name
        self._initialise()

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


    def generate_population(self, 
                            pop_name: str,
                            params: list) -> None:
        """
        Method generates a population of designs.
        @author: Stephen Krol
        @date: Jan 2026
        
        :param self: Current instance of the class.
        :param pop_name: Name of the population.
        :type pop_name: str
        :param params: List of Params objects for the population.
        :type params: list like

        :return: None
        :rtype: None
        """

        n = len(params)
        assert n % 2 == 0, "Number of designs 'n' must be even."

        start = time.time()
        jobs = []

        base_filepath = f"Experiments/{self.experiment_name}/{pop_name}"
        
        self._initialise_design(pop_name)

        # build jobs for generation
        for i in tqdm(range(n)):
            
            # write JSON parameters for initial population
            individual = params[i]
            filename = f"{individual.name}"
            individual.write_json(f"{base_filepath}/Params", f"{filename}", pop_name)
            jobs.append((pop_name, filename, self.sketch_dir, self.experiment_name, self.screen))

        if self.processing == "serial": # generate images serially
            for job in tqdm(jobs):
                self.generate_image(job)

        else: # generate images in parallel
            results = []
            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                
                futures = [
                    pool.submit(DesignGenerator.generate_image, job)
                    for job in jobs
                ]

            for f in as_completed(futures):
                try:
                    results.append(f.result())
                except Exception as e:
                    
                    results.append({
                        "status": "crashed",
                        "error": str(e)
                    })
        
        end = time.time()

        print(f"Took {end - start:.2f} seconds to generate {n} designs.")

        return results


    @staticmethod
    def generate_image(jobs: Tuple[str, str, str, str, bool]) -> None:
        """
        Generate image calls processing to generate an image from parameters.
        Arguments are passed as a tuple for compatibility with ProcessPoolExecutor.
        @author: Stephen Krol
        @date: Jan 2026
        
        :param jobs: A tuple containing the arguments for generation. 
            Arg1 is population name, Arg2 is filename, Arg3 is sketch directory,
            Arg4 is experiment name, Arg5 is screen boolean.
        :type jobs: Tuple[str, str, str, str, bool]

        :return: None
        :rtype: None
        """

        population_name, filename, sketch_dir, experiment_name, screen = jobs

        cwd = os.getcwd()
        filepath = f"{cwd}/Experiments/{experiment_name}/{population_name}/Params/{filename}.json"

        # if screen is available
        if screen:
            subprocess.run([
                "processing-java",
                f"--sketch={sketch_dir}",
                "--run",
                filepath], 
                check=True,
                cwd=cwd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        else:
            try:
                subprocess.run([
                    "xvfb-run", "-a",
                    "--server-args=-screen 0 1024x768x24 -nolisten tcp",
                    "processing-java",
                    f"--sketch={sketch_dir}",
                    "--run",
                    filepath], 
                    timeout=20,
                    check=True,
                    cwd=cwd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
            except subprocess.TimeoutExpired:
                print(f"Timeout expired for design: {filename}")

    def _initialise(self) -> None:
        """
        Method to initialise the Designs directory structure.
        @author: Stephen Krol
        @date: Jan 2026

        :param self: Current instance of the class.

        :return: None
        :rtype: None
        """

        if not os.path.exists("Experiments"):
            os.mkdir("Experiments")

        if not os.path.exists(f"Experiments/{self.experiment_name}"):
            os.mkdir(f"Experiments/{self.experiment_name}")

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

        if not os.path.exists(f"Experiments/{self.experiment_name}/{name}"):
            os.mkdir(f"Experiments/{self.experiment_name}/{name}")

        if not os.path.exists(f"Experiments/{self.experiment_name}/{name}/Images"):
            os.mkdir(f"Experiments/{self.experiment_name}/{name}/Images")

        if not os.path.exists(f"Experiments/{self.experiment_name}/{name}/Params"):
            os.mkdir(f"Experiments/{self.experiment_name}/{name}/Params")

            
class DesignEvolver:

    def __init__(self, 
                 spec_filepath: str,
                 experiment_name: str,
                 prompt: str,
                 n: int, 
                 k: int,
                 plot_pop: bool = False) -> None:
                 
        # verify design path exists and assign
        self.experiment_name = experiment_name
        self.design_path = f"Experiments/{experiment_name}"
        assert os.path.exists(self.design_path), f"Design path: f{self.design_path} does not exist"

        # read param spec
        assert os.path.exists(spec_filepath), f"Spec filepath: f{spec_filepath} does not exist"
        self.param_spec = self._read_param_spec(spec_filepath)

        # set population size
        self.population_size = n

        # assign prompt
        self.prompt = prompt

        # initialise processor
        self.processor = Qwen3VLBatchProcessor(
                model_name="Qwen/Qwen3-VL-7B-Instruct",
                device="cuda" if torch.cuda.is_available() else "cpu"
                )

        # set plot population boolean
        self.plot_pop = plot_pop

        # set initial population parameters
        self.population_params = self._generate_initial_params()
        self.current_population = 0

        # initialise ranking probabilities
        self.ranking_probabilities = self._calc_ranking_probabilities(len(self.population_params), k)

    def evaluate_population(self,
                            plot: bool = True) -> None:
        """
        Evaluate population ranks images using LLM comparisons. This updates the population_params
        attribute to be sorted in order of highest ranked phenotype.
        @author: Stephen Krol
        @Date: Jan 2026
        
        :param self: Current instance of the class.
        :param plot: Whether to plot the images
        :type plot: bool

        :return: None
        :rtype: None
        """

        population_image_fileapath = f"{self.design_path}/run{self.current_population}/Images"

        # retrieve filenames
        filenames = np.array([f"{param.name}.png" for param in self.population_params])

        # build comparison jobs
        jobs = build_messages(filenames, population_image_fileapath, self.prompt)

        # rank images using LLM
        results = self.processor.process_batch_chunked(jobs, chunk_size=32)

        # calculate ranks
        ranks = calc_ranks(results, len(filenames))
        sorted_idx = np.argsort(ranks)[::-1]

        self.population_params = [self.population_params[i] for i in sorted_idx]

        # plot images
        if plot:
            plot_image_grid(filenames[sorted_idx], nrows=5, ncols=4, filepath=population_image_fileapath, ranks=ranks[sorted_idx], save_path=population_image_fileapath, image_name=f'Population {self.current_population} Rankings')

    
    def evolve_population(self):

        # update current population count
        self.current_population += 1

        # sample new population based on ranking probabilities derived from tournament selection
        sampled_pop = random.choices(self.population_params, weights=self.ranking_probabilities, k=len(self.population_params))

        # select N couples for crossover
        sampled_couples = [random.choices(sampled_pop, k=2) for _ in range(len(sampled_pop))]

        # perform crossover to create new population
        children = []
        for i, (parent1, parent2) in enumerate(sampled_couples):
            alpha = random.uniform(0, 1) # TODO: might be a better method
            child_params = parent1.breed(other=parent2, alpha=alpha, child_name=f"run{self.current_population}_{i}")
            children.append(child_params)
         
        # mutate children
        mutation_rate = 0.1 # TODO: make this a parameter
        for child in children:
            child.mutate(mutation_rate)
        
        self.population_params = children


    def _plot_population(self):

        filenames = np.array([f"{param.name}.png" for param in self.population_params])
        population_image_fileapath = f"{self.design_path}/run{self.current_population}/Images"
        plot_image_grid(filenames, nrows=5, ncols=4, filepath=population_image_fileapath, save_path=population_image_fileapath, image_name=f'Population {self.current_population} Designs')

    def _calc_ranking_probabilities(self, N: int, k: int) -> np.ndarray:

        r = np.arange(1, N+1)

        # Probability of r-th item being the highest ranked in radomly selected k items
        return (1 - (r - 1) / N) ** k - (1 - r / N) ** k 

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
    
    def _generate_initial_params(self) -> List[Params]:
        """
        Method generates inital random parameters for sample.
        @author: Stephen Krol
        @date: Jan 2026
        
        :param self: Current instance of the class.

        :return: List of generated random Params objects.
        :rtype: List[Params]
        """

        param_list = []

        for i in range(self.population_size):

            params = {}

            # sample params using ranges from config
            for param in self.param_spec:
                param_config = self.param_spec[param]
                if param_config["calc"] == True:
                    continue
                elif param_config["type"] == "categorical":
                    params[param] = random.choice(param_config["values"])
                    continue
                elif param_config["type"] == "int":
                    sample = random.randint
                else:
                    sample = random.uniform

                params[param] = sample(param_config["min"], param_config["max"])

            # calculate remaining params
            params["dt"] = 1.0 / (params["fb"] * max(params["frx"], params["fry"] * params["sampleRate"]))

            # spaceP
            params["spaceP"] = 10**params["spaceD"]
        
            filename = f"run0_{i}"
            param_list.append(Params(params, self.param_spec, self.experiment_name, filename))
        
        return param_list
    
def main(experiment_name: str,
         runs :int,
         param_spec_filepath: str,
         sketch_dir: str,
         prompt: str,
         population_size: int = 20,
         processing: str = "serial",
         screen: bool = False,
         workers: int = 8):

    # initialise design generator
    generator = DesignGenerator(
            experiment_name=experiment_name,
            sketch_dir=sketch_dir,
            processing=processing,
            screen=screen,
            workers=workers
        )

    # initialise evolver
    evolver = DesignEvolver(
        spec_filepath=param_spec_filepath,
        experiment_name=experiment_name,
        prompt=prompt,
        n=population_size,
        k=5,
        plot_pop=True
    )

    for _ in range(runs):

        # generate population
        generator.generate_population(pop_name=f"run{evolver.current_population}", params=evolver.population_params)

        # plot population
        evolver._plot_population()

        # evaluate population
        evolver.evaluate_population(plot=True)

        # evolve population
        evolver.evolve_population()



if __name__ == "__main__":

    experiment_name = "initial_test_1"
    param_spec_filepath = "param_spec.yaml"
    sketch_dir = "/home/sjkro1/ARC-Discovery/Harmonograph"
    prompt = """
        You will be given two images and you need to output either '1' or '2' based on which image is more aesthetically pleasing.
        Designs with noticable patterns should be rated higher than those which are messy or noisy. Also penalise images that are
        just dark shapes. Focus on ranking patterns that have complex structures that are visible as higher.
        """
    population_size = 20
    processing = "parallel"
    screen = False
    workers = 8
    runs = 100

    main(
        experiment_name=experiment_name,
        runs=runs,
        param_spec_filepath=param_spec_filepath,
        sketch_dir=sketch_dir,
        prompt=prompt,
        population_size=population_size,
        processing=processing,
        screen=screen,
        workers=workers
    )

    # # Example usage
    # generator = DesignGenerator(
    #     experiment_name="test_experiment",
    #     spec_filepath="param_spec.yaml",
    #     sketch_dir="/home/sjkro1/ARC-Discovery/Harmonograph",
    #     processing="parallel",
    #     screen=False,
    #     workers=8
    # )

    # designs, params_json = generator.generate_population(n=20, pop_name="run0", inital_populaion=True)

    ############# START Evolver Example Usage #############

    # designs = np.array(os.listdir("Experiments/test_experiment/run0/Images"))
    # params = [f"{design.strip('.png')}.json" for design in designs]

    # evolver = DesignEvolver(
    #     design_path="/home/sjkro1/ARC-Discovery/aesthetic-evolution/Experiments/test_experiment",
    #     prompt="""
    #     You will be given two images and you need to output either '1' or '2' based on which image is more aesthetically pleasing.
    #     Designs with interesting patterns should be rated higher. Penalise designs that are too noisy and messy or dark blops. Focus on ranking patterns
    #     that have complex structures that are visible as higher.
    #     """,
    #     initial_pop_param_list=params,
    #     k=5
    # )

    # evolver.evaluate_population(designs, "run0", plot=True)

    # selected = evolver.evolve_population()
    # filenames = [design.name + ".png" for design in selected]
    # plot_image_grid(filenames, nrows=5, ncols=4, filepath="/home/sjkro1/ARC-Discovery/aesthetic-evolution/Experiments/test_experiment/run0/Images", image_name='Selected Designs')


    ############# END Evolver Example Usage #############


    ############# TEST MUTATION AND BREEDING #############

    # p1 = {
    #     "steps": 47,
    #     "sampleRate": 286.8727917356732,
    #     "ab": 167,
    #     "as": 0.6994946486325484,
    #     "fb": 17,
    #     "frx": 1.7260021760064077,
    #     "fry": 2.795336520899456,
    #     "px": 2.231274673480362,
    #     "py": -5.086525350173667,
    #     "dx": 0.2410312493937298,
    #     "dy": 0.006891169323498286,
    #     "lac": 3.305891216047324,
    #     "fo": 1.5091465083145679,
    #     "mix": 0.6364773573733347,
    #     "spaceD": -0.25884961893939207,
    #     "no": 2,
    #     "algorithm": "POW",
    #     "radial": 0,
    #     "dt": 7.335464509380872e-05,
    #     "spaceP": 0.5509984549608025
    # }

    # p2 = {
    #     "steps": 96,
    #     "sampleRate": 191.902874002995,
    #     "ab": 152,
    #     "as": 0.3989198683872677,
    #     "fb": 10,
    #     "frx": 3.682748974321429,
    #     "fry": 3.7008720162228705,
    #     "px": 1.2206917814245841,
    #     "py": 5.3443282627665525,
    #     "dx": 0.1805668344718995,
    #     "dy": 0.22006003144015002,
    #     "lac": 9.120395325539539,
    #     "fo": 1.880744852023576,
    #     "mix": 0.5776249438474294,
    #     "spaceD": 0.22630952557407635,
    #     "no": 3,
    #     "algorithm": "NOISE",
    #     "radial": 0,
    #     "dt": 0.00014080382556497074,
    #     "spaceP": 1.6838737454341606
    # }


    # with open("param_spec.yaml", 'r') as file:
    #     # Use safe_load to safely parse the YAML content
    #     configuration = yaml.safe_load(file)

    # parameter1 = Params(p1, configuration, "test_experiment", "test_design")
    # parameter2 = Params(p2, configuration, "test_experiment", "test_design")

    # child = parameter1.breed(parameter2, alpha=0.5, child_name="child_design")

    # print(child.write_json(".", "child_design", "test_design"))


    # original = parameter1.params.copy()

    # # mutate params
    # parameter1.mutate(mutation_rate=0.5)

    # for param in original:
    #     if original[param] != parameter1.params[param]:
    #         print(f"Parameter {param} changed from {original[param]} to {parameter1.params[param]}")

    ############# END TEST MUTATION AND BREEDING #############

    # sorted_idx = evolver.evaluate_population(designs, "test2")


    # evaluate_population(designs,)

    # filename = "design0"
    # initialise_design(filename)


    # config = read_param_spec("param_spec.yaml")

    # params = generate_rand_params(config)

    # write_json(params, f"Designs/Params/{filename}", filename)
    # generate_image(filename)