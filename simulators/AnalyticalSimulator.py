from utils import *
import numpy as np
from agents import AgentGroup
from agents import AgentNetwork
import random
from tqdm import tqdm
from utils.plotting import *

class AnalyticalSimulator:
  def __init__(self, iterations):
        self.iterations = iterations
        self.beliefs_mean = None

  def reset_simulation(self, agent_group):
    self.beliefs_mean = np.zeros((agent_group.size, self.iterations + 1))
    for i in range(agent_group.size):
            self.beliefs_mean[i][0] = agent_group.beliefs_old[i][0]

  def simulate(self, interactions, agent_group):
    self.reset_simulation(agent_group)
    for k in tqdm(range(1, self.iterations + 1), leave=False):
        for i in range(agent_group.size):
            agent_group = interactions.interact(agent_group, i)
            self.beliefs_mean[i][k] = agent_group.beliefs[i][0]
            if (i == 1):
              if k in [0, 1, 2]:
                plot_gaussian(agent_group.beliefs[i])
        agent_group.belief_old = agent_group.beliefs
    return self.beliefs_mean
  
  def single_information_source(self, agent_group: AgentGroup, information_source, filter_strength = 0):
    self.reset_simulation(agent_group)
    for k in range(self.iterations):
        for i in range(agent_group.size):
            agent_group.update_belief_replication(i, information_source, filter_strength)
            self.beliefs_mean[i][k + 1] =  agent_group.beliefs[i][0]
    return self.beliefs_mean