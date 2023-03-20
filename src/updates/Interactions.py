import numpy as np
import copy
import random

class Interactions:
    def __init__(self, update_rule, interaction_rate=1, information_source=None, interacting_agents=False, draw_from_source=False):
        self.update_rule = update_rule
        self.interaction_rate = interaction_rate
        self.information_source = information_source
        self.interacting_agents = interacting_agents
        self.draw_from_source = draw_from_source

    def agent_interaction(self, agent_network, agent):
        observed_agent = agent_network.get_observed_agent(agent)
        agent.belief_old = copy.deepcopy(agent.belief)
        agent.belief = self.update_rule.belief_update(agent, observed_agent)
        if random.uniform(0,1) < self.update_rule.rewire_probability:
            agent_network.rewire_network(agent, observed_agent)

    def information_interaction(self, agent):
        self.info_mean = self.information_source[0]
        self.info_sd = self.information_source[1]
        information_source = self.information_source
        if self.draw_from_source:
            information_source = (np.random.normal(self.info_mean , self.info_sd), self.info_sd)
        agent.belief  = self.update_rule.belief_update(agent, information_source)

    def update(self, agent_network, agent):
        if random.uniform(0,1) < self.interaction_rate:
            if self.interacting_agents:
                self.agent_interaction(agent_network, agent)
        else:    
            if self.information_source:
                self.information_interaction(agent)