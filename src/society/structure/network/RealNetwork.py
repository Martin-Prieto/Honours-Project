from utils.io import netwrok_from_file
import networkx as nx
from society.structure.network import Network

class RealNetwork(Network):
    ADVOGATO_PATH = "real_networks/Advogato.txt"
    FACEBOOK_PATH = "real_networks/Facebook.txt"
    FLICKR_PATH = "real_networks/Flickr.txt"
    GOOGLE_PLUS_PATH = "real_networks/Google+.txt"
    TWITTER_PATH = "real_networks/Twitter.txt"

    def __init__(self, network_name, path_to_data) -> None:
        self.network_name = network_name
        G = self.get_network(path_to_data)
        super().__init__(G)
    
    def get_network(self, path_to_data):
        match self.network_name:
            case "Advogato":
                G = netwrok_from_file(path_to_data + RealNetwork.ADVOGATO_PATH)
            case "Facebook":
                G = netwrok_from_file(path_to_data + RealNetwork.FACEBOOK_PATH)
            case "Flickr":
                G = netwrok_from_file(path_to_data + RealNetwork.FLICKR_PATH)
            case "Google+":
                G = netwrok_from_file(path_to_data + RealNetwork.GOOGLE_PLUS_PATH)
            case "Twitter":
                G = netwrok_from_file(path_to_data + RealNetwork.TWITTER_PATH)
        G.remove_edges_from(nx.selfloop_edges(G))
        return G
