# DESIGN_DECISION: Use scipy as it is more accurate to calculate eigenvalues
# from numpy.linalg import eig
from scipy.linalg import eig
import numpy as np


def pca_cust(A, k_num=-1, return_order=False):
    if k_num == -1:
        k_num = min(A.shape[0], A.shape[1])
    if k_num > min(A.shape[0], A.shape[1]):
        raise ValueError(k_num + " must be less than min(A.shape[0],A.shape[1]) " + min(A.shape[0], A.shape[1]))
    cov = np.cov(A)
    k, U = eig(cov)

    U_sorted, k_order = get_sorted_matrix_on_weights(k, U, return_order=True)
    k = k[::-1]
    U = U.transpose()[::-1]
    if return_order:
        return U.transpose().astype(float), np.diag(k.astype(float)), U.astype(float), k_order
    else:
        return U.transpose().astype(float), np.diag(k.astype(float)), U.astype(float)


def svd_cust(A, k_num=-1, return_order=False):
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
    U_sorted, k1_order = get_sorted_matrix_on_weights(k1, U, return_order=True)
    k1 = np.transpose(np.array(U_sorted, dtype=object))[0]
    U = np.stack(np.transpose(np.array(U_sorted, dtype=object))[1])

    V_sorted = get_sorted_matrix_on_weights(k2, V)
    V = []
    for i in range(len(V_sorted)):
        if i < len(k1):
            V.append(V_sorted[i][1] * V_sorted[i][0] / k1[i])
        else:
            break

    if return_order:
        if not transpose:
            return np.array(U).astype(float)[:,:k_num], np.diag(np.nan_to_num(np.sqrt(k1[:k_num].astype(float)))), np.array(V).astype(
                float)[:k_num,:], k1_order
        else:
            return np.transpose(np.array(V)).astype(float)[:,:k_num], np.diag(np.nan_to_num(np.sqrt(k1[:k_num].astype(float)))), np.transpose(
                np.array(U).astype(float))[:k_num,:], k1_order
    else:
        if not transpose:
            return np.transpose(np.array(V)).astype(float)[:,:k_num], np.diag(np.nan_to_num(np.sqrt(k1[:k_num].astype(float)))), np.transpose(
                np.array(U).astype(float))[:k_num,:], k1_order
        else:
            return np.array(U).astype(float)[:,:k_num], np.diag(np.nan_to_num(np.sqrt(k1[:k_num].astype(float)))), np.array(V).astype(
                float)[:k_num,:]


def get_sorted_matrix_on_weights(weights, V, return_order=False):
    v_dict = {}
    weight_dict = {}
    V = np.array(V)
    for i in range(len(weights)):
        if weights[i] < 0:
            v_dict.update({-weights[i]: -V[i]})
            weight_dict.update({-weights[i]: i})
        else:
            v_dict.update({weights[i]: V[i]})
            weight_dict.update({weights[i]: i})

    V_sorted = sorted(v_dict.items())[::-1]
    weights_order = sorted(weight_dict.items())[::-1]
    # print(V_sorted)
    # print(weights_order)
    if return_order:
        return V_sorted, weights_order
    else:
        return V_sorted