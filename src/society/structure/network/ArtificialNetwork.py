import networkx as nx
import numpy as np
from society.structure.network import Network
from utils.io import netwrok_from_file

class ArtificialNetwork(Network):
    def __init__(self, size, network_type, **kwargs) -> None:
        self.network_type = network_type
        G = self.get_network(size, kwargs)
        super().__init__(G)

    def get_network(self, size, kwargs):
        match self.network_type:
            case "random_graph":
                p = 0.5
                if "p" in kwargs:
                    p = kwargs["p"]
                G = nx.erdos_renyi_graph(size, p)
            case "fully_connected":
                G = nx.complete_graph(size)
            case "small_world":
                q = 0.5
                if "q" in kwargs:
                    q = kwargs["q"]
                k = 4
                if "k" in kwargs:
                    k = kwargs["k"]
                G = nx.watts_strogatz_graph(size, k, q)
            case "stochastic_block_model":
                group1 = size // 2
                group2 = size - group1
                G = nx.stochastic_block_model([group1, group2], [[0.2, 0.05], [0.05, 0.2]]).to_undirected()   
            case "scale_free":
                G = nx.scale_free_graph(size)
            case "barabasi_albert":
                m = 4
                if "m" in kwargs:
                    m = kwargs["m"]
                G = nx.barabasi_albert_graph(size, m)
            case _:
                raise Exception("Network type not found")
        G.remove_edges_from(nx.selfloop_edges(G))
        return G