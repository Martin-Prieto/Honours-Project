import numpy as np
import random
from utils.computations.analytical import filtered_gaussian_multiplication, filtered_gaussian_multiplication_unshared

class AgentGroup:
    def __init__(self, size: int, assimilation, default_uncertainty=0.2, extreme_agent_proportion=0):
        self.size = size
        self.assimilation = assimilation
        self.initial_belief_means = np.random.uniform(-1, 1, self.size)
        self.initial_belief_uncertainties = [default_uncertainty]*self.size
        self.beliefs_old = list(map(lambda x, y:(x,y), self.initial_belief_means, self.initial_belief_uncertainties))
        self.beliefs = self.beliefs_old
        
    def set_belief_means(self, mode="uniform"):
        match mode:
            case  "binormal":
                self.initial_belief_means = np.random.uniform(-1, 1, self.size)
            case _:
                self.initial_belief_means = np.random.uniform(-1, 1, self.size)

        self.set_beliefs_to_initial()

    def set_belief_uncertainties(self, sigma=0.2, mode="single", epsilon=0):
        self.initial_belief_uncertainties = [sigma]*self.size
        self.set_beliefs_to_initial()

    def set_extreme_agents(self, sigma, number, ratio=0.5):
        extreme_agents = random.sample(range(self.size), number)
        print(set(extreme_agents))
        positive_agents = int(number*ratio)
        
        for i in range(positive_agents):
            self.initial_belief_means[extreme_agents[i]] = 1
            self.initial_belief_uncertainties[extreme_agents[i]] = sigma
        
        for i in range(positive_agents, number):
            self.initial_belief_means[extreme_agents[i]] = -1
            self.initial_belief_uncertainties[extreme_agents[i]] = sigma
        self.set_beliefs_to_initial()
        

    def set_beliefs_to_initial(self):
        self.beliefs_old = list(map(lambda x, y:(x,y), self.initial_belief_means, self.initial_belief_uncertainties))
        self.beliefs = self.beliefs_old

    def set_uniform_beliefs(self, sigma_low, sigma_high, exterme_agent_proportion=0):
        self.initial_beliefs = []
        for i in range(self.size):
            if (random.uniform(0,1) < exterme_agent_proportion):
                if (random.uniform(0,1) < 0.5):
                    mean = 1
                else:
                    mean = - 1
                self.initial_belief_uncertainties[i] = sigma_low/10
            else:
                mean = np.random.uniform(-1,1)
                self.initial_belief_uncertainties[i] = np.random.uniform(sigma_low,sigma_high)
            self.initial_beliefs.append((mean, self.initial_belief_uncertainties[i]))
        self.beliefs = self.initial_beliefs
        self.beliefs_old = self.initial_beliefs
    
    def set_binormal_beliefs(self, mean_low, mean_high, sigma_low, sigma_high):
        for i in range(self.size):
            if (random.uniform(0, 1) < 0.5):
                mean = np.random.normal(mean_high,0.1)
            else: 
                mean = np.random.normal(mean_low,0.1)
            self.initial_belief_uncertainties[i] = np.random.uniform(sigma_low,sigma_high)
            self.initial_beliefs.append((mean, self.initial_belief_uncertainties[i]))
        self.beliefs = self.initial_beliefs
        self.beliefs_old = self.initial_beliefs

    def update_belief_replication(self, agent, information, bias_strength=0):
        if (random.uniform(0, 1)  < self.assimilation):
            filtered_belief = filtered_gaussian_multiplication(self.beliefs[agent], self.beliefs[agent], bias_strength)
            self.beliefs[agent] = filtered_gaussian_multiplication(filtered_belief, information, 1)

        