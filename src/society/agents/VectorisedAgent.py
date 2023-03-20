from computations.probabilistic.vectorised import *
from society.agents import Agent

class VectorisedAgent(Agent):
    def __init__(self, uncertainty, mean, number):
        super().__init__( uncertainty, mean, number)
        
    def initialise(self):
        self.belief = vector_gaussian((self.initial_mean, self.initial_uncertainty))
        self.belief_old = vector_gaussian((self.initial_mean, self.initial_uncertainty))
        self.tolerance = 6*self.initial_uncertainty

    def get_belief_mean(self):
        return compute_mean(self.belief)