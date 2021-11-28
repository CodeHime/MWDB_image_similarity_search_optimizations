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
    jump_factor = 0.05
    # jump_factor = 0.5
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

import math
import numpy as np
import pickle
import os

def normalize(a):
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, np.newaxis]
    return new_matrix

def predict(similarity_m, test_array):
        num_subjects = len(similarity_m) + 1
        c = 0.5
        pagerank = np.random.uniform(low=0, high=1, size=num_subjects)
        s_vector = np.zeros(num_subjects)
        s_vector[-1] = 1
        results = []
        for i in range(len(test_array)):
                q = np.array(test_array[i])
                q = q.T
                similarity_q_m = np.vstack([similarity_m, q])
                t_matrix = similarity_q_m @ similarity_q_m.T
                similarity_m_1 = min_max_scaler.transform(t_matrix)
                pagerank = np.linalg.inv(np.identity(num_subjects) - (1-c) * similarity_m_1) @ np.atleast_2d(c * s_vector).T
                pagerank = pagerank[:-1]
                # pagerank = np.argsort(pagerank)[::-1]
                cl = np.argmax(pagerank)
                results.append(cl)
        return results
