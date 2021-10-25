import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import skew
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from PIL import Image
import glob
from scipy import spatial
from scipy.stats import wasserstein_distance
import os


def get_rank_with_seeds(transition_matrix, m, s_list):
    jump_factor = 0.5
    seeds = np.zeros((transition_matrix.shape[0], 1))
    for seed in s_list:
        seeds[seed - 1] = 1 / transition_matrix.shape[0]

    ranks = np.dot((np.identity(transition_matrix.shape[0]) - (1 - jump_factor) * transition_matrix), jump_factor * seeds)
    ranks = ranks.reshape(transition_matrix.shape[0])

    t_discount = np.zeros(transition_matrix.shape[0])
    for i, row in enumerate(ranks):
        if i + 1 not in s_list:
            t_discount[i] = ranks[i] / (1 - jump_factor)
        else:
            t_discount[i] = (ranks[i] - (jump_factor / transition_matrix.shape[0])) / (1 - jump_factor)

    final_ranks = []
    for i, row in enumerate(t_discount):
        final_ranks.append((i + 1, row))
    final_ranks = sorted(final_ranks, reverse=True, key=lambda x: x[1])
    return final_ranks[:m]