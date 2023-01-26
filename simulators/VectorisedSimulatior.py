from helper import *
import numpy as np
import random
from matplotlib import pyplot as plt

def bounded_updating(agent_belief, information_source, bias_strength):    
        sigma = agent_belief[1]
        mu_1 = agent_belief[0]
        mu_2 = information_source[0]
        normalizing_term = 1/(sigma*math.sqrt(2 * math.pi))
        inverse_gaussian_term = 1/math.exp(((mu_1 - mu_2)**2)/(2*sigma**2))
        p = (bias_strength*normalizing_term*inverse_gaussian_term)/(bias_strength*normalizing_term*inverse_gaussian_term + (1 - bias_strength))
        new_mu = p * ((mu_1 + mu_2)/2)+(1-p)*mu_1
        return (new_mu, sigma)


def get_external_input(matrix, agent, belief_mean, sigma0):
    adjacent_agents = matrix[agent].nonzero()[1]
    i = random.choice(adjacent_agents)
    input = gaussian(belief_mean[i], sigma0[i], granularity)
    return input



class VectorisedSimulation:
  def __init__(self, iterations, number_agents, memory, assimilation):
        self.iterations = iterations
        self.number_agents = number_agents
        self.memory = memory
        self.assimilation = assimilation
        self.beliefs = np.zeros((self.number_agents, granularity))
        self.beliefs_mean = np.zeros((self.number_agents, self.iterations + 1))
        self.x0 = None
        self.sigma0 = None
        self.belief_old = None

  def set_beliefs(self, mean_low, mean_high, sigma_low, sigma_high):
        self.x0 = np.random.uniform(mean_low, mean_high, self.number_agents)
        self.sigma0 = np.random.uniform(sigma_low, sigma_high, self.number_agents)
        self.belief_old = np.array([gaussian(self.x0[i], self.sigma0[i], granularity) for i in range(self.number_agents)])

        for i in range(self.number_agents):
            self.beliefs_mean[i][0] = self.x0[i]
        

  def plot_belief_evolution(self, belief_mean):
    plt.xscale("log")
    for k in range(self.number_agents):
        plt.plot(belief_mean[k], color='green',  linewidth=0.3)
    plt.show()

  def sobkowicz_replication_vectorized(self, information_source, filter_strength = 0):
    true_seek = gaussian(information_source[0], information_source[1], granularity)
    filtered_input = true_seek
    for k in range(self.iterations):
        for i in range(self.number_agents):
            filter = self.belief_old[i]*filter_strength + (1-filter_strength) * uniform
            filtered_input = multiply(true_seek, filter)
            self.beliefs[i] = self.update_belief(filtered_input, self.belief_old[i], self.beliefs_mean[i][k], i)
            self.beliefs_mean[i][k + 1] =  round(sum(np.multiply(self.beliefs[i], granular)), 3)
        self.belief_old = np.array(self.beliefs)
    return self.beliefs_mean

  def network_belief_evolution(self, adjacency_matrix, trust = 1, filter_strength = 0):

    for k in range(self.iterations):
        for i in range(self.number_agents):
            filter = self.belief_old[i]*filter_strength + (1-filter_strength) * uniform
            external_input = get_external_input(adjacency_matrix, i, np.array(self.beliefs_mean).T[k], self.sigma0)
            filtered_input = trust*multiply(external_input, filter) + (1-trust)*uniform
            self.beliefs[i] = self.update_belief(filtered_input, self.belief_old[i], self.beliefs_mean[i][k], i) 
            self.beliefs_mean[i][k + 1] =  sum(np.multiply(self.beliefs[i], granular))
        self.belief_old = np.array(self.beliefs)
    return self.beliefs_mean

  def update_belief(self, input, belief_old, previous_belief_mean, agent):
    xtemp = np.zeros(granularity)
    if (random.uniform(0, 1)  < self.assimilation):
        belief = multiply(belief_old, input)
    else:
        xtemp = gaussian(previous_belief_mean, self.sigma0[agent], granularity)
        belief = np.array(belief_old)*self.memory + (1-self.memory)*np.array(xtemp)
    return belief

    



