import random
import math
import numpy as np

def ascos_score(subject_matrix, ascos, i, j):
    c = 0.9
    w_i_all = 0
    for x in range(len(subject_matrix)):
        if x != i:
            w_i_all += subject_matrix[i][x]
    score = 0
    for k in range(len(subject_matrix)):
        if k != i:
            score += c * ((subject_matrix[i][k] / w_i_all) * (1 - math.exp(-subject_matrix[i][k]))) * ascos[k][j]
    return score


def get_m_significant_subjects(ascos, m):
    subject_scores = [0] * len(ascos)
    for i in range(len(ascos)):
        for j in range(len(ascos[i])):
            subject_scores[i] += ascos[i][j]
    # print(subject_scores)
    return np.array(subject_scores).argsort()[:m]


def ascos_similarity(subject_matrix, n, m):
    ascos = [[0] * len(subject_matrix)] * len(subject_matrix)
    """Initializing the matrix with random values between 0 and 1"""
    for i in range(len(subject_matrix)):
        for j in range(len(subject_matrix)):
            ascos[i][j] = random.random()
    print(ascos)
    """TODO: Find the most n similar subjects from the subject_matrix
        and modify the subject_matrix to size of n*n"""
    score_diff = 200
    prev_score_diff = 0
    while prev_score_diff-score_diff > 2:
        # print(score_diff)
        score_diff = 0
        for i in range(len(subject_matrix)):
            for j in range(len(subject_matrix)):
                if i == j:
                    ascos[i][j] = 1
                else:
                    old_value = ascos[i][j]
                    new_value = ascos_score(subject_matrix, ascos, i, j)
                    score_diff += abs(new_value - old_value)
                    ascos[i][j] = new_value
        prev_score_diff=score_diff
        # print(ascos)
    m_subjects = get_m_significant_subjects(ascos, m)
    # print("m Most significant subjects are:", m_subjects)
    return m_subjects
