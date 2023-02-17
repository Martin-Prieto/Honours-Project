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
        self.belief_evolution = None

  def reset_simulation(self, agent_group):
    self.belief_evolution = np.zeros((agent_group.size, self.iterations + 2))
    for i in range(agent_group.size):
        self.belief_evolution[i][0] = agent_group.beliefs_old[i][0]
        self.belief_evolution[i][1] = agent_group.beliefs_old[i][0]

  def simulate(self, interactions, agent_group, show_progress=True):
    self.reset_simulation(agent_group)
    if show_progress:
      for k in tqdm(range(2, self.iterations + 2), leave=True):
          for i in range(agent_group.size):
            agent_group = interactions.interact(agent_group, i)
            self.belief_evolution[i][k] = agent_group.beliefs[i][0]
          agent_group.belief_old = agent_group.beliefs
    else:
      for k in range(2, self.iterations + 2):
          for i in range(agent_group.size):
            agent_group = interactions.interact(agent_group, i)
            self.belief_evolution[i][k] = agent_group.beliefs[i][0]
          agent_group.belief_old = agent_group.beliefs

    return self.belief_evolution
  
  def single_information_source(self, agent_group: AgentGroup, information_source, filter_strength = 0):
    self.reset_simulation(agent_group)
    for k in range(self.iterations):
        for i in range(agent_group.size):
            agent_group.update_belief_replication(i, information_source, filter_strength)
            self.belief_evolution[i][k + 1] =  agent_group.beliefs[i]
    return self.belief_evolution