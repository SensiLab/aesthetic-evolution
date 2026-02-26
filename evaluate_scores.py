
import math
from aesthetic_evolution.glicko import Player
import matplotlib.pyplot as plt


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

            _, _, image_a, image_b, outcome = line.strip().split(",")
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
        plot_top_players(n=12, players=sorted_players, image_names=image_names, image_dir="Data/curated", plot_name="top_players.png")

        # print bottom 10 players
        sorted_players = sorted(players.values(), key=lambda p: p.rating)
        plot_top_players(n=12, players=sorted_players, image_names=image_names, image_dir="Data/curated", plot_name="bottom_players.png")

    return sorted(players.values(), key=lambda p: p.rating, reverse=True)

if __name__ == "__main__":
    players = calc_scores("test-run-1/labels.csv", plot=True)

    print(max(players, key=lambda p: p.deviation).deviation)
    average_deviation = sum(p.deviation for p in players) / len(players)
    print(average_deviation)