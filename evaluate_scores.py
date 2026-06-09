'''
Evaluate model predictions against human rankings using Glicko scores and rank correlation metrics. This is designed to run over a fixed benchmark.
However, the individual functions, not the script after `if __name__ == "__main__":`, can be used for other purposes.
'''

from email.policy import default
import os
import cv2
import json
import math
import torch
import random
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from collections.abc import Sequence

from aesthetic_evolution.glicko import Player
from aesthetic_evolution.process_batch import Qwen3VLBatchProcessor
from aesthetic_evolution.CLIP_IQA import CLIP_IQA
from aesthetic_evolution.utils import build_messages
import matplotlib.pyplot as plt
from scipy.stats import kendalltau, spearmanr


def get_id(name: str) -> int:
    """
    Extract the player ID from the image filename.
    @author: Stephen Krol
    @date: Feb 2026

    :param name: The filename of the image (e.g., "run_n.jpg").
    :type name: str

    :return: The extracted player ID.
    :rtype: int
    """

    return int(name.strip(".png").split("_")[1])

def plot_top_players(n: int,
                     players: list[Player], 
                     image_names: dict[int, str],
                     image_dir: str,
                     plot_name: str) -> None:
    """
    Plot the images of top n players using matplotlib.
    @author: Stephen Krol
    @date: Feb 2026

    :param n: The number of top players to plot.
    :type n: int
    :param players: A list of sorted Player objects.
    :type players: list[Player]
    :param image_names: A dictionary mapping player IDs to image filenames.
    :type image_names: dict[int, str]
    :param image_dir: The directory where the images are stored.
    :type image_dir: str

    :return: None
    :rtype: None
    """

    grid_estimate = math.sqrt(n)
    rows, cols = math.ceil(grid_estimate), math.floor(grid_estimate)

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    for i in range(n):
        player = players[i]
        image_name = image_names[player.id]
        image_path = f"{image_dir}/{image_name}"
        img = plt.imread(image_path)
        ax = axes[i // cols, i % cols]
        ax.imshow(img)
        ax.set_title(f"ID: {player.id}\nRating: {player.rating:.2f}\nDeviation: {player.deviation:.2f}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(plot_name)


def compare_rankings(
    ranking_a: Sequence[int],
    ranking_b: Sequence[int],
    k_values: Sequence[int] = (5, 10, 20)) -> dict[str, float | int | dict[int, dict[str, float | int]]]:
    """
    Compare two rankings using rank correlation and top-k set overlap metrics.
    @author: Stephen Krol
    @date: Mar 2026

    Metrics returned:
      - Spearman rank correlation and p-value
      - Kendall tau correlation and p-value
      - Top-k overlap and Jaccard similarity for each k in k_values

    :param ranking_a: First ranking as ordered item IDs (best to worst).
    :type ranking_a: Sequence[int]
    :param ranking_b: Second ranking as ordered item IDs (best to worst).
    :type ranking_b: Sequence[int]
    :param k_values: Top-k thresholds for overlap/Jaccard metrics.
    :type k_values: Sequence[int]

    :return: Dictionary of similarity metrics.
    :rtype: dict[str, float | int | dict[int, dict[str, float | int]]]
    """

    ids_a = list(ranking_a)
    ids_b = list(ranking_b)

    if len(ids_a) != len(set(ids_a)):
        raise ValueError("ranking_a contains duplicate IDs")
    if len(ids_b) != len(set(ids_b)):
        raise ValueError("ranking_b contains duplicate IDs")

    ids_b_set = set(ids_b)
    common_items = [item for item in ids_a if item in ids_b_set]

    pos_a = {item: idx + 1 for idx, item in enumerate(ids_a)}
    pos_b = {item: idx + 1 for idx, item in enumerate(ids_b)}

    if len(common_items) >= 2:
        aligned_a = [pos_a[item] for item in common_items]
        aligned_b = [pos_b[item] for item in common_items]
        spearman_rho, spearman_p = spearmanr(aligned_a, aligned_b)
        kendall_tau, kendall_p = kendalltau(aligned_a, aligned_b)
    else:
        spearman_rho, spearman_p = float("nan"), float("nan")
        kendall_tau, kendall_p = float("nan"), float("nan")

    if not k_values:
        k_values = (10,)

    # TODO: verify this implementation of top-k metrics is correct and intuitive
    top_k_metrics: dict[int, dict[str, float | int]] = {}
    for k in sorted(set(k_values)):
        if k <= 0:
            raise ValueError(f"k_values must contain positive integers, found {k}")

        top_a = set(ids_a[:k])
        top_b = set(ids_b[:k])
        intersection = len(top_a & top_b)
        union = len(top_a | top_b)

        effective_k = min(k, len(ids_a), len(ids_b))
        overlap_fraction = intersection / effective_k if effective_k > 0 else float("nan")
        jaccard = intersection / union if union > 0 else 1.0

        top_k_metrics[k] = {
            "overlap_count": intersection,
            "overlap_fraction": overlap_fraction,
            "jaccard": jaccard,
        }

    return {
        "n_ranking_a": len(ids_a),
        "n_ranking_b": len(ids_b),
        "n_common_items": len(common_items),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p_value": float(kendall_p),
        "top_k_metrics": top_k_metrics,
    }

def write_benchmark_results(results: dict,
                            output_path: str) -> None:
    """
    Write the benchmark results to a JSON file.
    @author: Stephen Krol
    @date: Mar 2026

    :param results: The dictionary containing benchmark results.
    :type results: dict
    :param output_path: The file path to write the results to.
    :type output_path: str

    :return: None
    :rtype: None
    """

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

def retrieve_winner(result: dict) -> str:
    """
    Retrieve the winner from a model prediction result.
    @author: Stephen Krol
    @date: Mar 2026

    :param result: The dictionary containing a model prediction result.
    :type result: dict

    :return: The winner ("first", "second", "draw", "unknown") extracted from the result.
    :rtype: str
    """

    try:
        outcome = int(result["result"][-1].strip())
    except ValueError:
        return "draw"

    if outcome == 1:
        return "first"
    elif outcome == 2:
        return "second"
    elif outcome == 3:
        return "draw"
    else:
        return "unknown"

def model_predictions(model_name: str,
                      prompt_filepath: str,
                      benchmark_dir: str,
                      images: list | None = None,
                      device: str = "cuda",
                      csv_filename: str="model_predictions.csv",
                      moe: bool = False) -> None:
    """
    Generate model predictions and write to csv file for benchmarking.
    @author: Stephen Krol
    @date: Mar 2026


    :param model_name: The name of the model to use for predictions.
    :type model_name: str
    :param prompt_filepath: The file path to the system prompt to use for comparisons.
    :type prompt_filepath: str
    :param benchmark_dir: Directory containing the benchmark data (images).
    :type benchmark_dir: str 
    :param images: Optional list of image filenames to use (if not using benchmark_dir).
    :type images: list | None
    :param device: The device to run the model on ("cuda" or "cpu"), defaults to "cuda".
    :type device: str
    :param csv_filename: The filename to write model predictions to, defaults to "model_predictions.csv".
    :type csv_filename: str
    :param moe: Whether to use Mixture of Experts (MoE) version of the model if available, defaults to False.
    :type moe: bool

    :return: None
    :rtype: None
    """

    device = device if torch.cuda.is_available() else "cpu"
    processor = Qwen3VLBatchProcessor(model_name=model_name,
                                      device=device,
                                      moe=moe)


    if images is None:
        images = os.listdir(benchmark_dir)
        images.sort(key=lambda x: int(x.strip(".png").split("_")[1]))

    with open(prompt_filepath, 'r') as file:
        prompt = file.read()

    jobs = build_messages(images, benchmark_dir, prompt)

    results = processor.process_batch_chunked(jobs, chunk_size=32)

    with open(csv_filename, "w") as f:
        f.write("image_a,image_b,prediction\n")
        for job, result in zip(jobs, results):
            image_a = os.path.basename(job.image1_path)
            image_b = os.path.basename(job.image2_path)
            prediction = retrieve_winner(result)
            f.write(f"{image_a},{image_b},{prediction}\n")

def CLIP_IQA_Score(image_dir: str,
                   postive_prompt: str,
                   negative_prompt: str) -> None:
    """
    Calculate the CLIP Image Quality Assessment (IQA) score for images in a directory.
    @author: Stephen Krol
    @date: Mar 2026

    :param image_dir: The directory containing the images to evaluate.
    :type image_dir: str
    :param postive_prompt: The positive prompt for CLIP IQA.
    :type postive_prompt: str
    :param negative_prompt: The negative prompt for CLIP IQA.
    :type negative_prompt: str

    :return: None
    :rtype: None
    """

    processor = CLIP_IQA()
    images = os.listdir(image_dir)
    images.sort(key=lambda x: int(x.strip(".png").split("_")[1]))

    image_paths = [f"{image_dir}/{img}" for img in images if img.endswith(".png")]

    scores = processor.compute_clip_iqa_score(image_paths, 
                                              positive_prompt=postive_prompt,
                                              negative_prompt=negative_prompt)

    for img, (pos_score, neg_score) in zip(images, scores):
        print(f"{img}: P(good)={pos_score:.6f}, P(bad)={neg_score:.6f}")

    ids = [get_id(img) for img in images if img.endswith(".png")]
    ranked_ids = [id for id, _ in sorted(zip(ids, scores), key=lambda x: x[1][0], reverse=True)]

    return ranked_ids

def get_structural_complexity(img_path, r = 3, num = 0):
    """
    Calculate the structural complexity of an image.
    @author: Ahbinav Sood
    @date: Aug 2024

    :param img_path: The path to the image file.
    :type img_path: str
    :param r: The block size for calculating local mean values, defaults to 3.
    :type r: int
    :param num: Unused parameter, defaults to 0.
    :type num: int

    :return: The structural complexity score of the image.
    :rtype: float
    """
    # load image as grayscale img and normalise values to range 0-1
    img = cv2.imread(img_path, 0)
    if img.shape != (128, 128):
        img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)

    #standardize image
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean)/std
    rows, cols = img.shape
    # uncompressed size
    img_size_uncompressed = img.size
    # initialise new np array
    # structural complexity is calculated on this new image
    new_img = np.full((rows, cols), -1.0, dtype=np.float32)
    for i in range(img.shape[0]//(r) + 1):
        y = i * (r)
        y_end = min(y + (r), img.shape[0] + 1)
        for j in range(img.shape[1]//(r) + 1):
            x = j * (r)
            x_end = min(x + (r), img.shape[1] + 1)
            # print(img[y:y_end,x:x_end])
            m = img[y:y_end, x:x_end].mean() 
            v = 0
            if m >= 1: v = 1.
            elif m >= 0 and m < 1: v=0.7
            elif m >= -1 and m < 0: v=0.35
            else: v=0
            new_img[y:y_end, x:x_end] = v
    new_img_png = Image.fromarray(np.uint8((new_img)* 255), mode='L')
    img_file = BytesIO()
    new_img_png.save(img_file, 'png')
    
    # while metadata is still is file size, it is usually constant for all images
    # thus as it doesn't affect the relative order of the attributes
    # it proabaly has little effect on the effectiveness of the attribute value
    # in our model
    img_size_compressed = img_file.tell() # - metadata_constant 
    
    return img_size_compressed/img_size_uncompressed


def calculate_structural_complexity_ranks(image_dir: str) -> list[int]:
    """
    Calculate the structural complexity ranks for images in a directory.
    @author: Stephen Krol
    @date: Mar 2026

    :param image_dir: The directory containing the images to evaluate.
    :type image_dir: str

    :return: A list of image IDs ranked by structural complexity (most complex to least).
    :rtype: list[int]
    """

    image_paths = [f"{image_dir}/{img}" for img in os.listdir(image_dir) if img.endswith(".png")]
    complexities = [(get_id(os.path.basename(path)), get_structural_complexity(path)) for path in image_paths]
    ranked_ids = [id for id, _ in sorted(complexities, key=lambda x: x[1], reverse=True)]
    return ranked_ids

def _parse_score_line(line: str) -> tuple[str, str, str]:
    """
    Parse one benchmark CSV row into (image_a, image_b, outcome).
    Supports both 5-column and 3-column score formats.
    """

    parts = line.strip().split(",")
    if len(parts) == 5:
        _, _, image_a, image_b, outcome = parts
    elif len(parts) == 3:
        image_a, image_b, outcome = parts
    else:
        raise ValueError(f"Expected 3 or 5 CSV columns, found {len(parts)} in row: {line.strip()}")

    return image_a, image_b, outcome


def _load_score_rows(scores_path: str) -> list[tuple[int, str, str, str, int, int]]:
    """
    Load score rows as (row_idx, image_a, image_b, outcome, id_a, id_b).
    """

    rows: list[tuple[int, str, str, str, int, int]] = []

    with open(scores_path, "r") as f:
        next(f)  # Skip header
        for row_idx, line in enumerate(f):
            image_a, image_b, outcome = _parse_score_line(line)
            id_a = get_id(image_a)
            id_b = get_id(image_b)
            rows.append((row_idx, image_a, image_b, outcome, id_a, id_b))

    return rows


def _score_from_rows(rows: Sequence[tuple[int, str, str, str, int, int]],
                     plot: bool = False) -> tuple[list[Player], dict[int, str]]:
    """
    Apply Glicko updates from parsed score rows.
    """

    players: dict[int, Player] = {}
    image_names: dict[int, str] = {}

    for _, image_a, image_b, outcome, id_a, id_b in rows:
        if id_a not in players:
            players[id_a] = Player(id=id_a)
            image_names[id_a] = image_a
        if id_b not in players:
            players[id_b] = Player(id=id_b)
            image_names[id_b] = image_b

        player_a = players[id_a]
        player_b = players[id_b]

        snapshot_a = (player_a.rating, player_a.deviation)
        snapshot_b = (player_b.rating, player_b.deviation)

        if outcome == "first":
            player_a.update_rating([snapshot_b[0]], [snapshot_b[1]], [1])
            player_b.update_rating([snapshot_a[0]], [snapshot_a[1]], [0])
        elif outcome == "second":
            player_a.update_rating([snapshot_b[0]], [snapshot_b[1]], [0])
            player_b.update_rating([snapshot_a[0]], [snapshot_a[1]], [1])
        elif outcome == "draw":
            player_a.update_rating([snapshot_b[0]], [snapshot_b[1]], [0.5])
            player_b.update_rating([snapshot_a[0]], [snapshot_a[1]], [0.5])

    if plot:
        sorted_players = sorted(players.values(), key=lambda p: p.rating, reverse=True)
        plot_top_players(n=12, players=sorted_players, image_names=image_names, image_dir="Data/curated", plot_name="top_players-jon.png")

        # print bottom 10 players
        sorted_players = sorted(players.values(), key=lambda p: p.rating)
        plot_top_players(n=12, players=sorted_players, image_names=image_names, image_dir="Data/curated", plot_name="bottom_players-jon.png")

    return sorted(players.values(), key=lambda p: p.rating, reverse=True), image_names


def calc_scores(scores_path: str, plot:bool = False) -> None:
    """
    Main function to evaluate Glicko scores from a CSV file containing match outcomes.
    @author: Stephen Krol
    @date: Feb 2026

    :param scores_path: The path to the CSV file containing match outcomes.
    :type scores_path: str
    :param plot: Whether to plot the top players, defaults to False.
    :type plot: bool

    :return: None
    :rtype: None
    """

    players: dict[int, Player] = {}
    image_names: dict[int, str] = {}

    with open(scores_path, "r") as f:
        next(f)  # Skip header
        for line in f:
            
            try:
                _, _, image_a, image_b, outcome = line.strip().split(",")
            except ValueError:
                image_a, image_b, outcome = line.strip().split(",")
                
            id_a = get_id(image_a)
            id_b = get_id(image_b)

            if id_a not in players:
                players[id_a] = Player(id=id_a)
                image_names[id_a] = image_a
            if id_b not in players:
                players[id_b] = Player(id=id_b)
                image_names[id_b] = image_b

            player_a = players[id_a]
            player_b = players[id_b]

            snapshot_a = (player_a.rating, player_a.deviation)
            snapshot_b = (player_b.rating, player_b.deviation)

            if outcome == "first":
                player_a.update_rating([snapshot_b[0]], [snapshot_b[1]], [1])
                player_b.update_rating([snapshot_a[0]], [snapshot_a[1]], [0])
            elif outcome == "second":
                player_a.update_rating([snapshot_b[0]], [snapshot_b[1]], [0])
                player_b.update_rating([snapshot_a[0]], [snapshot_a[1]], [1])
            elif outcome == "draw":
                player_a.update_rating([snapshot_b[0]], [snapshot_b[1]], [0.5])
                player_b.update_rating([snapshot_a[0]], [snapshot_a[1]], [0.5])

    if plot:
        sorted_players = sorted(players.values(), key=lambda p: p.rating, reverse=True)
        plot_top_players(n=12, players=sorted_players, image_names=image_names, image_dir="Data/curated", plot_name="top_players-jon.png")

        # print bottom 10 players
        sorted_players = sorted(players.values(), key=lambda p: p.rating)
        plot_top_players(n=12, players=sorted_players, image_names=image_names, image_dir="Data/curated", plot_name="bottom_players-jon.png")

    return sorted(players.values(), key=lambda p: p.rating, reverse=True), image_names


# TODO: update so n is a percentage of copmarisons rather than a fixed number, to be more robust to different dataset sizes
def calc_scores_sampled(scores_path: str,
                        n: int,
                        plot: bool = False,
                        seed: int | None = None) -> tuple[list[Player], dict[int, str]]:
    """
    Evaluate Glicko scores on a sampled subset of CSV comparisons where each design
    appears in at most n accepted comparisons.

    Sampling is randomized; pass seed for reproducible subsets.

    :param scores_path: The path to the CSV file containing match outcomes.
    :type scores_path: str
    :param n: Maximum number of sampled comparisons each design can appear in.
    :type n: int
    :param plot: Whether to plot the top players, defaults to False.
    :type plot: bool
    :param seed: Optional random seed for deterministic sampling.
    :type seed: int | None

    :return: Players sorted by rating (descending) and image filename map.
    :rtype: tuple[list[Player], dict[int, str]]
    """

    if n < 1:
        raise ValueError(f"n must be >= 1, found {n}")

    rows = _load_score_rows(scores_path)

    rng = random.Random(seed)
    shuffled_rows = list(rows)
    rng.shuffle(shuffled_rows)

    sampled_rows: list[tuple[int, str, str, str, int, int]] = []
    comparisons_per_design: dict[int, int] = {}

    for row in shuffled_rows:
        _, _, _, _, id_a, id_b = row

        # Count each design at most once per comparison.
        design_ids = {id_a, id_b}
        if any(comparisons_per_design.get(design_id, 0) >= n for design_id in design_ids):
            continue

        sampled_rows.append(row)
        for design_id in design_ids:
            comparisons_per_design[design_id] = comparisons_per_design.get(design_id, 0) + 1

    sampled_rows.sort(key=lambda row: row[0])

    return _score_from_rows(sampled_rows, plot=plot)


def get_pre_calc_ranks(csv_path: str) -> list[int]:
    """
    Get pre-calculated ranks from a CSV file.
    """

    calcs = pd.read_csv((csv_path))

    calcs = calcs.sort_values(by="S Complexity", ascending=False)
    print(calcs[["image", "S Complexity"]].head(10))
    s_complexitys_sorted_id = calcs["image"].tolist()

    calcs = calcs.sort_values(by="MC Complexity", ascending=False)
    ncomplexitys_sorted_id = calcs["image"].tolist()

    calcs = calcs.sort_values(by="Fractal Dimension", ascending=False)
    fractal_dims_sorted_id = calcs["image"].tolist()

    calcs = calcs.sort_values(by="Fractal Complexity", ascending=False)
    fractal_comps_sorted_id = calcs["image"].tolist()

    return s_complexitys_sorted_id, ncomplexitys_sorted_id, fractal_dims_sorted_id, fractal_comps_sorted_id


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate model predictions against human rankings using Glicko scores and rank correlation metrics.")
    parser.add_argument("--benchmark_dir", type=str, default="Data/curated/Images", help="Directory containing benchmark images.")
    parser.add_argument("--benchmark_scores_csv", type=str, default="jon-ranking/labels.csv", help="CSV file containing human benchmark scores.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct", help="Name of the model to generate predictions with.")
    parser.add_argument("--prompt_filepath", type=str, default="config/jon_prompt.txt", help="File path to the system prompt for model comparisons.")
    parser.add_argument("--model_predictions_csv", type=str, default="model_predictions.csv", help="CSV filename to write model predictions to.")
    parser.add_argument("--n", type=int, default=50, help="Number of comparisons each design can appear in for sampled scoring.")

    args = parser.parse_args()

    # run full prediction and evaluation pipeline once to get final scores for benchmarking
    model_predictions(benchmark_dir=args.benchmark_dir, 
                      model_name=args.model_name,
                      prompt_filepath=args.prompt_filepath,
                      csv_filename=args.model_predictions_csv,
                      device="cuda",
                      moe=False)

    players, image_names = calc_scores(args.benchmark_scores_csv)
    ids = [player.id for player in players]

    spearman = 0
    kendall = 0
    j5 = 0
    j10 = 0
    j20 = 0

    for i in range(1000):
        model_players, model_image_names = calc_scores_sampled(args.model_predictions_csv, n=args.n)
        model_ids = [player.id for player in model_players]
        benchmarks = compare_rankings(ids, model_ids)
        spearman += benchmarks["spearman_rho"]
        kendall += benchmarks["kendall_tau"]
        j5 += benchmarks["top_k_metrics"][5]["jaccard"]
        j10 += benchmarks["top_k_metrics"][10]["jaccard"]
        j20 += benchmarks["top_k_metrics"][20]["jaccard"]

    print(f"Average Spearman rho over 1000 samples @{args.n} comparisons: {spearman / 1000:.4f}")
    print(f"Average Kendall tau over 1000 samples @{args.n} comparisons: {kendall / 1000:.4f}")
    print(f"Average Jaccard index for top 5 over 1000 samples @{args.n} comparisons: {j5 / 1000:.4f}")
    print(f"Average Jaccard index for top 10 over 1000 samples @{args.n} comparisons: {j10 / 1000:.4f}")
    print(f"Average Jaccard index for top 20 over 1000 samples @{args.n} comparisons: {j20 / 1000:.4f}")

    # Compare against traditional metrics if available
    if os.path.exists("traditional-results.csv"):
        s_complexitys_sorted_id, ncomplexitys_sorted_id, fractal_dims_sorted_id, fractal_comps_sorted_id = get_pre_calc_ranks("traditional-results.csv")

        benchmarks = compare_rankings(ids, s_complexitys_sorted_id)
        write_benchmark_results(benchmarks, "Structural_Complexity_results2.json")

        benchmarks = compare_rankings(ids, ncomplexitys_sorted_id)
        write_benchmark_results(benchmarks, "NComplexity_results.json")

        benchmarks = compare_rankings(ids, fractal_dims_sorted_id)
        write_benchmark_results(benchmarks, "Fractal_Dimension_results.json")

        benchmarks = compare_rankings(ids, fractal_comps_sorted_id)
        write_benchmark_results(benchmarks, "Fractal_Complexity_results.json")


    # run clip-iqa scoring and benchmarking    
    ranked_ids = CLIP_IQA_Score(image_dir="Data/curated/Images",
                                postive_prompt="Good Design",
                                negative_prompt="Bad Design")

    benchmarks = compare_rankings(ids, ranked_ids)
    write_benchmark_results(benchmarks, "sampled-50.json")