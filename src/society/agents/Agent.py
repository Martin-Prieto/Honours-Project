from abc import ABC, abstractclassmethod

class Agent(ABC):
    def __init__(self, uncertainty, mean, number):
        self.number = number
        self.initial_uncertainty = uncertainty
        self.initial_mean = mean
        self.tolerance = 4*uncertainty
        self.belief = None
        self.belief_old = None

    @abstractclassmethod   
    def initialise(self):
        pass

    @abstractclassmethod
    def get_belief_mean(self):
        pass