from utils.io import netwrok_from_file
import networkx as nx
from society.structure.network import Network

class RealNetwork(Network):
    ADVOGATO_PATH = "../../../data/real_networks/Advogato.txt"
    FACEBOOK_PATH = "real_networks/Facebook.txt"
    FLICKR_PATH = "real_networks/Flickr.txt"
    GOOGLE_PLUS_PATH = "real_networks/Google+.txt"

    def __init__(self, network_name) -> None:
        self.network_name = network_name
        G = self.get_network()
        super().__init__(G)
    
    def get_network(self):
        match self.network_name:
            case "Advogato":
                G = netwrok_from_file(RealNetwork.ADVOGATO_PATH)
            case "Facebook":
                G = netwrok_from_file(RealNetwork.ADVOGATO_PATH)
            case "Flickr":
                G = netwrok_from_file(RealNetwork.ADVOGATO_PATH)
            case "Google+":
                G = netwrok_from_file(RealNetwork.ADVOGATO_PATH)
            case _:
                G = nx.scale_free_graph(self.size)
                G = nx.DiGraph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
