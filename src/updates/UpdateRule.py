from computations.probabilistic.analytical import *
from computations.probabilistic.vectorised import *
from computations.probabilistic.iterative import *
from society.agents import VectorisedAgent, IterativeAgent, AnalyticalAgent, Agent

class UpdateRule:
    analytical_operation = staticmethod(filtered_gaussian_multiplication_explicit_update)
    vectorised_operation = staticmethod(vector_filtered_gaussian_multiplication)
    iterative_operation = staticmethod(iterative_filtered_gaussian_multiplication)

    def __init__(self, assimilation_bias=0, evaluation_bias=0, rewire_probability=0, filter_likelihood=False):
        self.assimilation_bias = assimilation_bias
        self.evaluation_bias = evaluation_bias
        self.rewire_probability = rewire_probability
        self.filter_likelihood = filter_likelihood

    def belief_update(self, agent, observation):
        belief = agent.belief
        input, operation = self.get_update_settings(agent, observation)
        return self.compute_update(belief, input, operation)
    
    def compute_update(self, belief, input, operation):
        if self.filter_likelihood:
            filtered_input = operation(input, belief, self.assimilation_bias)
            new_belief = operation(belief, filtered_input, 1 - self.evaluation_bias)
        else:
            filtered_belief = operation(belief, belief, self.assimilation_bias)
            new_belief = operation(filtered_belief, input, 1 - self.evaluation_bias)
        return new_belief

    def get_update_settings(self, agent, observation):
        input = None
        operation = None
        if isinstance(observation, Agent):
            if isinstance(observation, VectorisedAgent):
                mean = compute_mean(observation.belief_old)
                input = vector_gaussian((mean, agent.initial_uncertainty))
                operation = UpdateRule.vectorised_operation
            if isinstance(observation, IterativeAgent):
                mean = iterative_compute_mean(observation.belief_old)
                input = iterative_gaussian((mean, agent.initial_uncertainty))
                operation = UpdateRule.iterative_operation
            if isinstance(observation, AnalyticalAgent):
                input = (observation.belief_old[0], agent.initial_uncertainty)
                operation = UpdateRule.analytical_operation
        else:
            if isinstance(agent, VectorisedAgent):
                input = vector_gaussian(observation)
                operation = UpdateRule.vectorised_operation
            if isinstance(agent, IterativeAgent):
                input = iterative_gaussian(observation)
                operation = UpdateRule.iterative_operation
            if isinstance(agent, AnalyticalAgent):
                input = observation
                operation = UpdateRule.analytical_operation
        return input, operation

    # def get_signal_quality(self, belief, signal, trust):
    #     warnings.filterwarnings("error")
    #     distance = abs(belief[0]-signal)
    #     try:
    #         signal_uncertainty = trust*(1-(1/((1/1.8**self.evaluation)*(2**self.evaluation)))+1/((1/1.8**self.evaluation)*((2-distance)**self.evaluation)))
    #     except:
    #         signal_uncertainty =  float('inf')

    #     warnings.resetwarnings()
    #     return signal_uncertainty