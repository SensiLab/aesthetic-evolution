"""
Utility functions for image comparison and ranking.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List
from matplotlib import pyplot as plt


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
    for all unique pairs.
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
        rank = int(result["result"].strip())
        if rank == 1:
            ranks[i, j] += 1
        else:
            ranks[j, i] += 1

    return ranks.sum(axis=1)

def plot_image_grid(filenames: list,
                    nrows: int,
                    ncols: int,
                    filepath=str,
                    ranks=None,
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
    :param filepath: Path to the directory containing the images.
    :type filepath: str
    :param ranks: Optional list of rank scores for each image.
    :type ranks: list like or None
    """
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))
    for i in range(nrows):
        for j in range(ncols):
            idx = i * ncols + j
            img = plt.imread(os.path.join(filepath, filenames[idx]))
            ax[i, j].imshow(img)
            if ranks is not None:
                ax[i, j].set_title(f"{filenames[idx].split('_')[-1].strip('.png')} : {ranks[idx]}")
            else:
                ax[i, j].set_title(filenames[idx].split('_')[-1].strip('.png'))
            ax[i, j].axis('off')
    plt.tight_layout()
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