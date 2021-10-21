from scipy.linalg import svd
# from numpy.linalg import eig
from scipy.linalg import eig
from math import isclose, sqrt
import numpy as np


def pca_cust(A, k_num=-1):
    """
    Calculate PCA of the given matrix
    :param A: matrix to calculate pca for
    :param k_num: number of latent features to return
    """
    if k_num == -1:
        k_num = min(A.shape[0], A.shape[1])
    if k_num > min(A.shape[0], A.shape[1]):
        raise ValueError(k_num + " must be less than min(A.shape[0],A.shape[1]) " + min(A.shape[0], A.shape[1]))
    cov = np.cov(A)
    k, U = eig(cov)
    return U.astype(float), k.astype(float), np.transpose(U).astype(float)


def order_weights(weights, A):
    """
    Given a matrix A, return A with decresasing order of weights
    """
    pass


def svd_cust(A, k_num=-1):
    """
    Calculate SVD of the given matrix
    :param A: matrix to calculate SVD for
    :param k_num: number of latent features to return
    """
    if k_num == -1:
        k_num = min(A.shape[0], A.shape[1])
    if k_num > min(A.shape[0], A.shape[1]):
        raise ValueError(k_num + " must be less than min(A.shape[0],A.shape[1]) " + min(A.shape[0], A.shape[1]))
    transpose = A.shape[0] > A.shape[1]
    if transpose:
        A = np.transpose(A)
    data_mat = np.dot(A, np.transpose(A))
    feature_mat = np.dot(np.transpose(A), A)

    k1, U = eig(data_mat)
    k2, V = eig(feature_mat)

    V = np.transpose(V).astype(float)
    v_dict = {}
    for i in range(len(k2)):
        if k2[i] < 0:
            v_dict.update({-k2[i]: -V[i]})
        else:
            v_dict.update({k2[i]: V[i]})

    V = []
    V_sorted = sorted(v_dict.items())[::-1]
    for i in range(len(V_sorted)):
        if i < len(k1):
            V.append(V_sorted[i][1] * V_sorted[i][0] / k1[i])
        else:
            break
    if not transpose:
        return np.array(U).astype(float), np.nan_to_num(np.sqrt(k1.astype(float))), np.array(V).astype(float)
    else:
        return np.transpose(np.array(V).astype(float)), np.nan_to_num(np.sqrt(k1.astype(float))), \
               np.transpose(np.array(U).astype(float))