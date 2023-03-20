import numpy as np
from computations.insights.relative import disagreement, polarisation, mse, medse, wrong_agents, diversity

class Insights:
    def __init__(self, insights=[], *kwargs):
        self.belief_evolution=None
        self.mean_belief_evolution=None
        self.polarisation_evolution=None
        self.disagreement_evolution=None
        self.convergence_time = None
        self.average_opinion_change = None
        self.insights = insights
        self.mse_evolution = None
        self.medse_evolution = None
        self.wrong_agents_evolution = None
        self.diversity_evolution = None
        self.kwargs = kwargs

    def initialise_insights(self, agent_network,interactions, iterations):
        self.initialise_belief_evolution(agent_network, iterations)
        if "mean_belief" in self.insights:
            self.initialise_mean_belief_evolution(iterations)
        if "polarisation" in self.insights:
            self.initialise_polarisation_evolution(iterations)
        if "disagreement" in self.insights:
            self.initialise_disagreement_evolution(agent_network, iterations)
        if "convergence" in self.insights:
            self.initialise_convergence_time()
        if "opinion_change" in self.insights:
            self.initialise_opinion_change(iterations)
        if "mse" in self.insights:
            self.initialise_mse(agent_network, interactions, iterations)
        if "medse" in self.insights:
            self.initialise_medse(agent_network, interactions, iterations)
        if "wrong_agents"in self.insights:
            self.initialise_wrong_agents(agent_network, interactions, iterations)
        if "diversity" in self.insights:
            self.initialise_diversity(agent_network,iterations)


    def update_insights(self, agent_network, interactions, iteration):
        self.update_belief_evolution(agent_network, iteration)
        if self.mean_belief_evolution is not None:
            self.update_mean_belief_evolution(iteration)
        if self.polarisation_evolution is not None:
            self.update_polarisation_evolution(iteration)
        if self.disagreement_evolution is not None:
            self.update_disagreement_evolution(agent_network, iteration)
        if self.convergence_time is not None:
            self.update_convergence_time(agent_network, iteration)
        if self.average_opinion_change is not None:
            self.update_average_opinion_change(agent_network, iteration)
        if self.mse_evolution is not None:
            self.update_mse(agent_network, interactions, iteration)
        if self.medse_evolution is not None:
            self.update_medse(agent_network, interactions, iteration)
        if self.wrong_agents_evolution is not None:
            self.update_wrong_agents(agent_network, interactions, iteration)
        if self.diversity_evolution is not None:
            self.update_diversity_evolution(agent_network, iteration)

    
    def initialise_belief_evolution(self, agent_network, iterations):
        self.belief_evolution =  np.zeros((agent_network.size, iterations + 2))
        self.update_belief_evolution(agent_network, 1)

    def initialise_diversity(self,agent_network, iterations):
        self.diversity_evolution = np.zeros(iterations+2)
        self.update_diversity_evolution(agent_network, 1)

    def initialise_mean_belief_evolution(self, iterations):
        self.mean_belief_evolution = np.zeros(iterations+2)
        self.update_mean_belief_evolution(1)

    def initialise_polarisation_evolution(self, iterations):
        self.polarisation_evolution = np.zeros(iterations+2)
        self.update_polarisation_evolution(1)

    def initialise_disagreement_evolution(self, agent_network, iterations):
        self.disagreement_evolution = np.zeros(iterations+2)
        self.update_disagreement_evolution(agent_network, 1)

    def initialise_convergence_time(self):
        self.convergence_time = -1

    def initialise_opinion_change(self, iterations):
        self.average_opinion_change = np.zeros(iterations+2)

    def initialise_mse(self, agent_network, interatcions, iterations):
        self.mse_evolution = np.zeros(iterations+2)
        self.update_mse(agent_network, interatcions, 1)

    def initialise_medse(self, agent_network, interatcions, iterations):
        self.medse_evolution = np.zeros(iterations+2)
        self.update_medse(agent_network, interatcions, 1)

    def initialise_wrong_agents(self, agent_network, interatcions, iterations):
        self.wrong_agents_evolution = np.zeros(iterations+2)
        self.update_medse(agent_network, interatcions, 1)


    def update_belief_evolution(self, agent_network, iteration):
        for i in range(agent_network.size):
            self.belief_evolution[i][iteration] = agent_network.agents[i].get_belief_mean()

    def update_mean_belief_evolution(self, iteration):
        self.mean_belief_evolution[iteration] = np.mean(self.belief_evolution.T[iteration])

    def update_polarisation_evolution(self, iteration):
        beliefs = self.belief_evolution.T[iteration]
        self.polarisation_evolution[iteration] = polarisation(beliefs)

    def update_disagreement_evolution(self, agent_network, iteration):
        edges = agent_network.get_edges()
        agents = agent_network.agent_lookup
        self.disagreement_evolution[iteration] = disagreement(agents, edges)

    def update_convergence_time(self, agent_network, iteration):
        if self.convergence_time == -1:
            opinion_change = 0
            for agent in agent_network.agents:
                opinion_change += abs(agent.belief[0] - agent.belief_old[0])
            if round(opinion_change, 2) == 0:
                self.convergence_time = iteration
    
    def update_average_opinion_change(self, agent_network, iteration):
        total_opinion_change = 0
        for agent in agent_network.agents:
            previous_belief = agent.belief_old[0]
            current_belief = agent.belief[0]
            change = abs(previous_belief - current_belief)
            total_opinion_change += change
        self.average_opinion_change[iteration] = total_opinion_change/agent_network.size

    def update_mse(self, agent_network, interatcions, iteration):
        self.mse_evolution[iteration] = mse(agent_network, interatcions.information_source[0])
    
    def update_medse(self, agent_network, interatcions, iteration):
        self.medse_evolution[iteration] = medse(agent_network, interatcions.information_source[0])

    def update_wrong_agents(self, agent_network, interatcions, iteration):
        self.wrong_agents_evolution[iteration] = wrong_agents(agent_network, interatcions.information_source[0], 0.025)

    def update_diversity_evolution(self, agent_network, iteration):
        mean = np.mean(self.belief_evolution.T[iteration])
        self.diversity_evolution[iteration] = diversity(agent_network, mean)