import numpy as np
from society.structure.AgentCollection import AgentCollection
from society.agents.AnalyticalAgent import AnalyticalAgent
from society.structure.network.Network import Network
from computations.insights.relative import disagreement
import copy
import random

class AgentNetwork(AgentCollection):
    def __init__(self, belief_distribution, network, agent_type = AnalyticalAgent):
        self.network = network
        agent_numbers = list(network.nodes())
        super().__init__(agent_numbers, belief_distribution, agent_type)
        
    def get_observed_agent(self, agent):
        adjacent_agents = self.network.get_neighbors(agent.number)
        if adjacent_agents:
            observed_agent = np.random.choice(adjacent_agents)
        else:
            self.network.add_random_neighbour(agent.number)
            return self.get_observed_agent(agent)
        return self.agent_lookup[observed_agent]
    
    def rewire_network(self, agent, observed_agent):
        if abs(agent.belief[0] - observed_agent.belief[0]) > agent.tolerance:
                self.network.rewire(agent.number, observed_agent.number)

    def initial_rewiring(self, tolerance, iterations):
        for _ in range(iterations):
            rewired = 0
            for agent in self.agents:
                for i in self.network.get_neighbors(agent.number):
                    observed_agent = self.agent_lookup[i]
                    if abs(observed_agent.initial_mean - agent.initial_mean) > tolerance:
                        self.network.rewire(agent.number, observed_agent.number)
                        rewired += 1
            print(disagreement(self.agent_lookup, self.get_edges()))
                        
    def get_edges(self):
        return copy.deepcopy(self.network.G.edges)

    def reset_opinions(self):
        super().initialise()
    
    def reset_network(self):
        self.network.reset()

        