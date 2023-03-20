from scipy.stats import kurtosis
import numpy as np
from statistics import median

def kurtosis(belief_evolution):
    return kurtosis(belief_evolution, axis=0)

def polarisation_evolution(belief_evolution):
    polarisation_evolution = np.apply_along_axis(polarisation, 0, belief_evolution)
    return polarisation_evolution

def disagreement_evolution(belief_evolution, edge_evolution):
    belief_evolution = belief_evolution.T
    disagreement_evolution = []
    for i in range(len(edge_evolution)):
        disagreement_evolution.append(disagreement(belief_evolution[i+1], edge_evolution[i]))
    return disagreement_evolution


def disagreement(agents, edges):
    disagreement = 0
    for edge in edges:
        node1 = edge[0]
        node2 = edge[1]
        opinion1 = agents[node1].belief[0]
        opinion2 = agents[node2].belief[0]
        disagreement += (opinion1 - opinion2)**2
    return disagreement/len(edges)


def polarisation(beliefs):
    n = len(beliefs)
    centered_beliefs = beliefs - sum(beliefs/n)
    polarisation = np.matmul(centered_beliefs.T, centered_beliefs)
    return polarisation

def mse(agent_network, true_value):
    sum_squared_error = 0
    for agent in agent_network.agents:
        belief = agent.belief[0]
        sum_squared_error += (belief - true_value)**2
    return sum_squared_error/agent_network.size

def medse(agent_network, true_value):
    absolute_errors = []
    for agent in agent_network.agents:
        belief = agent.belief[0]
        absolute_error = (belief - true_value)**2
        absolute_errors.append(absolute_error)
    return median(absolute_errors)

def wrong_agents(agent_network, true_value, threshold):
    wrong_agents = 0
    for agent in agent_network.agents:
        belief = agent.belief[0]
        absolute_error = abs(belief - true_value)
        if (absolute_error > threshold):
            wrong_agents += 1
    return wrong_agents

def diversity(agent_network, mean):
    squared_diff = 0
    for agent in agent_network.agents:
        squared_diff += (agent.belief[0] - mean)**2
    return squared_diff/agent_network.size

    
