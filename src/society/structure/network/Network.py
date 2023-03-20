import networkx as nx
import numpy as np
import copy

class Network:
    def __init__(self, network):
        self.initial_G = copy.deepcopy(network)
        self.G = network

    def rewire(self, node, connection):
        unconnected = self.get_non_neighbors(node)
        if unconnected:
            new_connection = np.random.choice(unconnected)
            self.G.add_edge(node, new_connection)
            self.G.remove_edge(node,connection)
        else:
            self.G.remove_edge(node,connection)
    
    def add_random_neighbour(self, node):
        unconnected = self.get_non_neighbors(node)
        new_connection = np.random.choice(unconnected)
        self.G.add_edge(node, new_connection)

    def get_neighbors(self, node):
        return list(nx.neighbors(self.G, node))

    def get_non_neighbors(self, node):
        return list(nx.non_neighbors(self.G, node))
    
    def nodes(self):
        return self.G.nodes()

    def reset(self):
        self.G = copy.deepcopy(self.initial_G)