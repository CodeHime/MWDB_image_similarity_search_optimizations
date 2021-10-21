import matplotlib.pyplot as plt
import networkx as nx

def make_graph_with_labels(adjacency_matrix, mylabels):
    rows, cols = np.where(adjacency_matrix > 0)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr, node_size=500, labels=mylabels, with_labels=True)
    plt.show()

# example use case
data = np.random.rand(5,5)
my_labels = {i:str(i) for i in range(5)}
make_graph_with_labels(data, my_labels)
data