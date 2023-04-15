from computations.probabilistic.iterative import *
from society.agents import Agent

class IterativeAgent(Agent):
    def __init__(self, uncertainty, mean, number):
        super().__init__(uncertainty, mean, number)
        
    def initialise(self):
        self.belief = iterative_gaussian((self.initial_mean, self.initial_uncertainty))
        self.belief_old = iterative_gaussian((self.initial_mean, self.initial_uncertainty))
        self.tolerance = 6*self.initial_uncertainty

    def get_belief_mean(self):
        return iterative_compute_mean(self.belief)