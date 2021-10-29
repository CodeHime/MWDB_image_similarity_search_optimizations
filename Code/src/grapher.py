import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import cityblock

"""
def index_tobe_zero(data, n):
    index = data.argsort()
    temp = len(index) - n
    return index[:temp]
"""

def create_adjacency_matrix(dataset, n):
    A = np.array(dataset)
    A2 = np.zeros(A.shape)
    srows = A.shape[0] - 1
    scols = A.shape[1] - 1
    temp = np.argsort(A)
    for i in range(0, srows + 1):
        for j in temp[i][n+1:]:
            A[i][j] = 0
    for i in range(0, srows + 1):
        A2[i] = A[i] / np.sum(A[i])
    return A2


def make_graph_with_labels_visual(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix > 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    plt.show()

# # example use case
# data = np.random.rand(5,5)
# my_labels = {i:str(i) for i in range(5)}
# make_graph_with_labels(data, my_labels)
# data