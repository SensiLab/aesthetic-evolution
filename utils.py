"""
Utility functions for image comparison and ranking.
"""


import numpy as np
from dataclasses import dataclass
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
    Docstring for build_messages. Function takes a list of images
    and builds comparison jobs for all unique pairs.
    
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
    Docstring for calc_ranks.
    
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