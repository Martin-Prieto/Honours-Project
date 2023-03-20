import numpy as np
import copy
import random

class AgentCollection:
    def __init__(self, agent_numbers, belief_distribution, agent_type):
        self.agent_numbers = agent_numbers
        self.size = len(self.agent_numbers)
        self.agents =  self.create_agents(belief_distribution, agent_type)
        self.agent_lookup = self.create_lookup()
        self.initialise()

    def create_agents(self, belief_distribution, agent_type):
        means = belief_distribution.generate_mean(self.size)
        uncertaities = belief_distribution.generate_uncertainty(self.size)
        return [agent_type(uncertaities[i], means[i], self.agent_numbers[i]) for i in range(self.size)]
    
    def set_uncertainties(self, distribution):
        uncertainties = distribution.generate(self.size)
        for i in range(self.size):
            self.agents[i].initial_uncertainty = uncertainties[i]
            self.agents[i].trust = uncertainties[i]
    
    def create_lookup(self):
        agent_lookup = {}
        for agent in self.agents:
            agent_lookup[agent.number] = agent
        return agent_lookup

    def initialise(self):
        for agent in self.agents:
            agent.initialise()

    def set_extreme_agents(self, uncertainty, number_left, number_right):
        number = number_left + number_right
        extreme_agents = random.sample(range(self.size), number)
        self.set_sepcific_agents(extreme_agents[:number_left], uncertainty, 1)
        self.set_sepcific_agents(extreme_agents[number_left:], uncertainty, -1)

    def set_sepcific_agents(self, agent_numbers, uncertainty, belief):
        for i in agent_numbers:
            self.agents[i].initial_mean = belief
            self.agents[i].initial_uncertainty = uncertainty