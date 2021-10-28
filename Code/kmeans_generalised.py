import numpy as np
from scipy.spatial.distance import cityblock
from sklearn.cluster import KMeans


def kmeans_func(A1,k):
    A = np.array(A1)
    size = A.shape[0]
    reducedMatrix = np.zeros((size,k))
    kmeans = KMeans(n_clusters=k, random_state=0).fit(A)
    centers = kmeans.cluster_centers_
    for i in range(0,size):
        for j in range(0,k):
            reducedMatrix[i][j] = cityblock(A[i],centers[j])
    #returnMatrix = []
    #For calculating the weight associated with each subject/type vector
    #for i in range(0,size):
    #    returnMatrix.append((reducedMatrix[i],np.sum(reducedMatrix[i])))
    #For sorting the matrix according to the weights
    #returnMatrix.sort(key = lambda x:x[1],reverse=True)
    #print(returnMatrix)
    return (reducedMatrix,centers)