from society.agents import Agent

class AnalyticalAgent(Agent):
    def __init__(self, uncertainty, mean, number):
        super().__init__(uncertainty, mean, number)
        
    def initialise(self):
        self.belief = (self.initial_mean, self.initial_uncertainty)
        self.belief_old =  (self.initial_mean, self.initial_uncertainty)
        self.tolerance = 6*self.initial_uncertainty

    def get_belief_mean(self):
        return self.belief[0]