

import math
import os
from utils import plot_image_grid
from CLIP_IQA import CLIP_IQA


processor = CLIP_IQA()

# retrieve image paths
image_dir = "/home/sjkro1/ARC-Discovery/aesthetic-evolution/Experiments/glicko-test-1/run2/Images"

image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.startswith("run")]

pos_prompt = "Good Harmonograph"
neg_prompt = "Messy Harmonograph"

scores = processor.compute_clip_iqa_score(
    image_path=image_paths,
    positive_prompt=pos_prompt,
    negative_prompt=neg_prompt
)

# sort scores by P(good) in descending order, sortidx
sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i][0], reverse=True)
print("Ranked Images by P(good):")
for rank, idx in enumerate(sorted_indices, start=1):
    image_path = image_paths[idx]
    p_good, p_bad = scores[idx]
    print(f"Rank {rank}: {os.path.basename(image_path)} - P(good): {p_good:.6f}, P(bad): {p_bad:.6f}")



# plot images in ranked order
grid_estimate = math.sqrt(len(image_paths))
image_names = [os.path.basename(path) for path in image_paths]

plot_image_grid([image_names[idx] for idx in sorted_indices],
                nrows=math.ceil(grid_estimate), 
                ncols=math.floor(grid_estimate),
                population_size=len(image_paths),
                filepath=image_dir,
                ranks = [scores[idx][0] for idx in sorted_indices],
                plot=False,
                image_name="CLIP_IQA_Ranking2.png")
