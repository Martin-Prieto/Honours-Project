import networkx as nx
import numpy as np
from agents.AgentGroup import AgentGroup
import random

class AgentNetwork(AgentGroup):
    def __init__(self, size, assimilation, network_type: str):
        super().__init__(size, assimilation)
        self.network_type = network_type
        match network_type:
            case "random_graph":
                G = nx.G = nx.erdos_renyi_graph(self.size, 0.5, seed=123, directed=False).to_undirected()
            case "fully_connected":
                G = nx.complete_graph(self.size).to_undirected()
            case "scale_free":
                G = nx.scale_free_graph(self.size).to_undirected()
            case _:
                G = nx.scale_free_graph(self.size).to_undirected()
        self.adjacency_matrix = nx.to_numpy_matrix(G)

    def get_observed_agent(self, agent):
        adjacent_agents = self.adjacency_matrix[agent].nonzero()[1]
        observed_agent = np.random.choice(adjacent_agents)
        return observed_agent
    
    def rewire_network(self, agent, observed_agent, update_rule):
        if abs(self.beliefs[agent][0] - self.beliefs[observed_agent][0]) > update_rule.trust_bound:
            if (random.uniform(0, 1)  < update_rule.rewire_probability):
                self.adjacency_matrix[agent, observed_agent] = 0
                if self.network_type != "fully_connected":
                    not_linked = np.where(self.adjacency_matrix[agent] == 0)[1]
                    new_link = np.random.choice(not_linked)
                    self.adjacency_matrix[agent, new_link] = 1

    def initial_rewiring(self, trust_bound):
        for i in range(self.size):
            for agent in (self.adjacency_matrix[i].nonzero()[1]):
                if abs(self.beliefs[i][0] - self.beliefs[agent][0]) > trust_bound:
                    self.adjacency_matrix[i, agent] = 0
                    not_linked = np.where(self.adjacency_matrix[i] == 0)[1]
                    new_link = np.random.choice(not_linked)
                    self.adjacency_matrix[i, new_link] = 1