import networkx as nx
import numpy as np


def netwrok_from_file(file_name):
        G = nx.Graph()
        with open(file_name) as f:
            for line in f:
                u, v = map(int, line.strip().split()[:2])
                G.add_edge(u, v)
        return G

def save_arrays(arrays, filename):
    with open(filename, 'wb') as f:
        for array in arrays:
            np.save(f, array)
    f.close()

def load_arrays(filename):
    arrays = []
    with open(filename, 'rb') as f:
        while True:
            try:
                arrays.append(np.load(f))
            except ValueError:
                break
    f.close()
    return arrays
     