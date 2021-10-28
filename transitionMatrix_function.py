import numpy as np
from scipy.spatial.distance import cityblock

def transition_mat(A1,n):
    A = np.array(A1)
    A2 = np.zeros(A.shape)
    srows = A.shape[0]-1
    scols = A.shape[1]-1
    temp = np.argsort(A)[::-1]
    for i in range(0,srows+1):
        for j in temp[i][:n+1]:
            A[i][j] = 0
    for i in range(0,srows+1):
        A2[i] = A[i]/np.sum(A[i])
    return A2
