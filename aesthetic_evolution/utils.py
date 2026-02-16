"""
Utility functions for image comparison and ranking.
"""


from dataclasses import dataclass
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import torch
from transformers import PreTrainedModel
from typing import List


@dataclass
class ComparisonJob:
    """Represents a single image comparison job"""
    job_id: str
    image1_path: str
    image2_path: str
    system_prompt: str

def build_messages(data: np.ndarray,
                   root: str,
                   prompt: str) -> List[ComparisonJob]:
    """
    Function takes a list of images and builds comparison jobs
    for all unique pairs. Assumes images are in the order of their indices.
    @author: Stephen Krol
    @Date: Jan 2026
    
    :param data: Array containing filenames for images.
    :type data: np.array
    :param root: Root directory for image paths
    :type root: str
    :param prompt: System prompt for the comparison task
    :type prompt: str

    :return: List of ComparisonJob instances for all unique image pairs.
    :rtype: List[ComparisonJob]
    """
    # build messages
    jobs = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            jobs.append(
                ComparisonJob(
                    job_id=f"comparison_{i}_{j}",
                    image1_path=f"{root}/{data[i]}",
                    image2_path=f"{root}/{data[j]}",
                    system_prompt=prompt
                )
            )
    
    return jobs

def calc_ranks(results: List[dict], n: int):
    """
    Function calculates the rank scores for each image based on comparison results.
    @atuthor: Stephen Krol
    @Date: Jan 2026
    
    :param results: List of result dictionaries from comparison jobs.
    :type results: List[Dict]
    :param n: Number of images being compared.
    :type n: int

    :return: Array of rank scores for each image.
    :rtype: np.ndarray
    """
    ranks = np.zeros((n, n))
    for result in results:
        i, j = map(int, result["job_id"].split("_")[1:])

        # if the model is reasoning step-by-step, the result will be in the last character of the output
        if len(result["result"]) > 1:
            try:
                rank = int(result["result"][-1].strip())
            except Exception as e:
                # TODO: deal with this more robustly, maybe by re-prompting the model or using a default value
                print(result["result"])
                raise(e)
                
        else:
            rank = int(result["result"].strip())

        if rank == 1:
            ranks[i, j] += 1
        elif rank == 2:
            ranks[j, i] += 1

    return ranks.sum(axis=1)

def update_glicko_scores(players: list, results: List[dict]) -> None:
    """
    Function updates Glicko scores for each player based on comparison results.
    Designed to work with batched generation where one batch is a rating period and
    all comparisons in that batch are used to update scores at the end of the period.
    @author: Stephen Krol
    @Date: Jan 2026

    :param players: List of Param instances representing each image's Glicko scores. 
        Assumes players are indexed in the same order as their IDs in the comparison jobs.
    :type players: list
    :param results: List of result dictionaries from comparison jobs.
    :type results: List[dict]

    :return: None
    :rtype: None
    """

    games = {}

    # retrieve active players
    for result in results:
        i, j = map(int, result["job_id"].split("_")[1:])
        if i not in games:
            games[i] = {
                "opponent_ratings": [],
                "opponent_deviations": [],
                "outcomes": []
            }
        if j not in games:
            games[j] = {
                "opponent_ratings": [],
                "opponent_deviations": [],
                "outcomes": []
            }
    
    # pre rating update
    # for player_id in games:
    #     players[player_id].pre_rating_period_update(c=63.2)

    snapshot_ratings = {player_id: players[player_id].rating for player_id in games}
    snapshot_deviations = {player_id: players[player_id].deviation for player_id in games} 

    # retrieve outcomes for each player and their opponents from results
    for result in results:
        i, j = map(int, result["job_id"].split("_")[1:])

        if len(result["result"]) > 1:
            try:
                rank = int(result["result"][-1].strip())
            except Exception as e:
                print(result["result"])
                raise(e)
        else:
            rank = int(result["result"].strip())

        # score i
        score_i = 1 if rank == 1 else 0.5 if rank == 3 else 0
        games[i]["opponent_ratings"].append(snapshot_ratings[j])
        games[i]["opponent_deviations"].append(snapshot_deviations[j])
        games[i]["outcomes"].append(score_i)


        # score j
        score_j = 1 if rank == 2 else 0.5 if rank == 3 else 0
        games[j]["opponent_ratings"].append(snapshot_ratings[i])
        games[j]["opponent_deviations"].append(snapshot_deviations[i])
        games[j]["outcomes"].append(score_j)

    
    # update ratings
    for player_id, game in games.items():
        players[player_id].update_rating(game["opponent_ratings"], game["opponent_deviations"], game["outcomes"])




def plot_image_grid(filenames: list,
                    nrows: int,
                    ncols: int,
                    population_size: int,
                    filepath:str,
                    ranks:list=None,
                    plot:bool=False,
                    save_path:str=".",
                    image_name="Rankings"):
    """
    Function plots a grid of images with an option to display ranks.
    @author: Stephen Krol
    @Date: Jan 2026

    :param filenames: List of image filenames to plot.
    :type filenames: list like
    :param nrows: Number of rows in the grid.
    :type nrows: int
    :param ncols: Number of columns in the grid.
    :type ncols: int
    :param population_size: Total number of images in the population.
    :type population_size: int
    :param filepath: Path to the directory containing the images.
    :type filepath: str
    :param ranks: Optional list of rank scores for each image.
    :type ranks: list like or None
    :param plot: Whether to display the plot or save image.
    :type plot: bool
    :param save_path: Directory to save the plotted image.
    :type save_path: str
    :param image_name: Filename for the saved image.
    :type image_name: str

    :return: None
    :rtype: None
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            if idx >= population_size:
                ax[i, j].axis('off')
                continue
            img = plt.imread(os.path.join(filepath, filenames[idx]))
            ax[i, j].imshow(img)
            if ranks is not None:
                ax[i, j].set_title(f"{filenames[idx].split('_')[-1].strip('.png')} : {round(ranks[idx])}")
            else:
                ax[i, j].set_title(filenames[idx].split('_')[-1].strip('.png'))
            ax[i, j].axis('off')
    plt.tight_layout()

    if plot:
        plt.show()
    else:
        plt.savefig(f"{save_path}/{image_name}.png")


def prob(k: int, N: int) -> np.ndarray:
    """
    Function to calculate the probablity of the r-th item being the highest ranked
    among k samples drawn with replacement from N unique items.
    @author: Stephen Krol
    @Date: Jan 2026
    
    :param k: Number of samples drawn with replacement.
    :type k: int
    :param N: Population size
    :type N: int

    :return: Array of probabilities for each item being the highest ranked.
    :rtype: np.ndarray
    """

    r = np.arange(1, N+1)

    return (1 - (r - 1) / N) ** k - (1 - r / N) ** k # Probability of r-th unique item being the k-th unique item observed


def simulated(k: int, N: int) -> np.ndarray:
    """
    Simulation of tournament selection probabilities to validate prob function.
    @author: Stephen Krol
    @Date: Jan 2026
    
    :param k: Number of samples drawn with replacement.
    :type k: int
    :param N: Population size
    :type N: int

    :return: Array of simulated probabilities for each item being the highest ranked.
    :rtype: np.ndarray
    """

    steps = 1000000

    counts = np.zeros(N)
    values = np.arange(1, N+1)

    for _ in range(steps):

        observed = np.min(np.random.choice(values, size=k, replace=True))
        counts[observed - 1] += 1

    return counts / steps



def timed_inference(model: PreTrainedModel, inputs: torch.Tensor) -> tuple[torch.Tensor, float]:
    """
    Docstring for timed_inference function, which measures the time taken for model inference.
    Import that cuda is synchronized before and after inference to get accurate timing due
    to asynchronous nature of GPU operations.
    @author: sjkro1

    :param model: transformer model to be evaluated
    :type model: PreTrainedModel
    :param inputs: input tensor for the model
    :type inputs: torch.Tensor

    :return: tuple of (model outputs, time taken in seconds)
    :rtype: tuple[torch.Tensor, float]
    """
    torch.cuda.synchronize()
    start = time.perf_counter()

    outputs = model.generate(**inputs)

    torch.cuda.synchronize()
    end = time.perf_counter()

    return outputs, end - start

def benchmark(model: PreTrainedModel, inputs: torch.Tensor, runs: int = 10, warmup: int = 3) -> dict[str, float]:
    """
    Docstring for benchmark function that runs multiple inferences on a model and computes
    performance metrics such as mean, p50, and p95 inference times. It includes a warm-up phase
    to stabilize performance before measurements.
    @author: sjkro1

    :param model: transformer model to be evaluated
    :type model: PreTrainedModel
    :param inputs: input tensor for the model
    :type inputs: torch.Tensor
    :param runs: number of inference runs to perform
    :type runs: int
    :param warmup: number of warm-up runs to perform
    :type warmup: int

    :return: dictionary with mean, p50, and p95 inference times in seconds
    :rtype: dict[str, float]
    """
    times = []
    peak_allocated = []

    # Warm-up
    for _ in range(warmup):
        _ = model.generate(**inputs)
    torch.cuda.synchronize()

    for _ in range(runs):
        torch.cuda.reset_peak_memory_stats()


        _, t = timed_inference(model, inputs)
        times.append(t)

    return {
        "mean_s": sum(times) / len(times),
        "p50_s": sorted(times)[len(times)//2],
        "p95_s": sorted(times)[int(len(times)*0.95)],
    }

# Benchmarking latency and VRAM usage
def benchmark_latency_and_vram(
    model: PreTrainedModel,
    inputs: torch.Tensor,
    runs: int = 10,
    warmup: int = 3,
    ) -> dict[str, float]:

    """
    Docstring for benchmark_latency_and_vram. This function benchmarks a model's 
    inference latency and VRAM usage over multiple runs, including a warm-up phase.
    It returns latency percentiles (p50, p95, mean) and VRAM  allocation statistics
    (p50, p95, peak).
    @author: sjkro1
    
    :param model: Model to benchmark
    :type model: PreTrainedModel
    :param inputs: Input tensor for the model
    :type inputs: torch.Tensor
    :param runs: Number of inference runs to perform
    :type runs: int
    :param warmup: Number of warm-up runs to perform
    :type warmup: int

    :return: Dictionary with latency and VRAM usage metrics
    :rtype: dict[str, float]
    """

    times = []
    peak_allocated = []

    # Warm-up (not measured)
    for _ in range(warmup):
        _ = model.generate(**inputs)
    torch.cuda.synchronize()

    for _ in range(runs):
        # Reset per-run memory peak
        torch.cuda.reset_peak_memory_stats()

        # CUDA-safe timing
        torch.cuda.synchronize()
        start = time.perf_counter()

        _ = model.generate(**inputs)

        torch.cuda.synchronize()
        end = time.perf_counter()

        times.append(end - start)
        peak_allocated.append(
            torch.cuda.max_memory_allocated() / 1024**2
        )

    # Sort for percentile calculation
    times.sort()
    peak_allocated.sort()

    return {
        # Latency
        "latency_p50_s": times[len(times) // 2],
        "latency_p95_s": times[int(len(times) * 0.95)],
        "latency_mean_s": sum(times) / len(times),

        # VRAM (true memory cost)
        "vram_alloc_p50_MB": peak_allocated[len(peak_allocated) // 2],
        "vram_alloc_p95_MB": peak_allocated[int(len(peak_allocated) * 0.95)],
        "vram_alloc_peak_MB": max(peak_allocated)
    }