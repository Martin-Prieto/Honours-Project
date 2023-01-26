from updates.UpdateRule import UpdateRule
import random

class Interactions:
    def __init__(self, update_rule: UpdateRule, assimilation, bidirectional_updates=True, information_source=None):
        self.update_rule = update_rule
        self.assimilation = assimilation
        self.information_source = information_source
        self.bidirectional_updates = bidirectional_updates

    def agent_interaction(self, agent_network, agent):
        observed_agent = agent_network.get_observed_agent(agent)
        agent_network.beliefs[agent] = self.update_rule.belief_update(agent_network.beliefs[agent], agent_network.beliefs[observed_agent], agent_network.initial_belief_uncertainties[agent], agent_network.initial_belief_uncertainties[observed_agent])
        if self.bidirectional_updates:
            agent_network.beliefs[observed_agent] = self.update_rule.belief_update(agent_network.beliefs[observed_agent], agent_network.beliefs[agent], agent_network.initial_belief_uncertainties[observed_agent], agent_network.initial_belief_uncertainties[agent])
        agent_network.rewire_network(agent, observed_agent, self.update_rule)
        return agent_network

    def information_interaction(self, agent_network, agent):
        agent_network.beliefs[agent] = self.update_rule.belief_update(agent_network.beliefs[agent], self.information_source, agent_network.initial_belief_uncertainties[agent], self.information_source[1])
        return agent_network

    def interact(self, agent_network, agent):
        if (random.uniform(0, 1)  < self.assimilation):
            agent_network = self.agent_interaction(agent_network, agent)
        
        if self.information_source:
            agent_network = self.information_interaction(agent_network, agent)
        return agent_network