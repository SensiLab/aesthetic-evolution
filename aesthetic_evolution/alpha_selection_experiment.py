
import math
import numpy as np
import random

def biased_alpha(p1_score, p2_score):

    return p1_score / (p1_score + p2_score)


N = 10

alphas = np.arange(N)

scores = np.random.randint(1, math.comb(N, 2), size=N)

# randomly choose pairs and compute biased alphas
biased_alphas = []
for _ in range(10):

    p1, p2 = random.sample(range(N), 2)
    alpha = biased_alpha(scores[p1], scores[p2])
    biased_alphas.append(alpha)
    print(f"Parent 1: {p1} (score: {scores[p1]}), Parent 2: {p2} (score: {scores[p2]}), Biased alpha: {alpha:.3f}")
