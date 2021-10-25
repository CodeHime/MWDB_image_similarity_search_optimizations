import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import cityblock


def index_tobe_zero(data, n):
    index = data.argsort()[::-1]
    temp = len(index) - n
    return index[:temp]


def create_adjacency_matrix(dataset, n):
    dlen = len(dataset[0])
    sim_mat = np.zeros((dlen, dlen))
    for i in range(0, dlen):
        for j in range(0, dlen):
            sim_mat[i][j] = cityblock(dataset[0][i][0], dataset[0][j][0])
    for i in range(0, len(sim_mat)):
        index_zero = index_tobe_zero(sim_mat[i], n)
        for j in index_zero:
            sim_mat[i][j] = 0
        total_sum = np.sum(sim_mat[i])
        k = 0
        for x in sim_mat[i]:
            if x != 0:
                sim_mat[i][k] = 1 - x / total_sum
            k = k + 1
    print(sim_mat)
    return sim_mat


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