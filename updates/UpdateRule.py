from utils.computations.analytical import *

class UpdateRule:
    def __init__(self, bias_strength=0, trust=1, rewire_probability=0, trust_bound=1, 
                       evolving_uncertainty=False, initial_uncertainty=False, 
                       shared_uncertainty=False, filtering="filter_input", test=False):

        self.bias_strength = bias_strength
        self.trust = trust
        self.rewire_probability = rewire_probability
        self.trust_bound = trust_bound
        self.evolving_uncertainty = evolving_uncertainty
        self.initial_uncertainty = initial_uncertainty
        self.shared_uncertainty = shared_uncertainty
        self.filtering = filtering
        self.test = test

    def testing(self, belief, input, belief_uncertainty0, input_uncertainty0):
        return filtered_gaussian_multiplication_explicit_update(belief, input, self.trust)

    def belief_update(self, belief, input, belief_uncertainty0, input_uncertainty0):

        if self.test:
            return self.testing(belief, input, belief_uncertainty0, input_uncertainty0)

        update_input = self.get_update_input(belief, input, belief_uncertainty0, input_uncertainty0)

        match self.filtering:
            case "filter_input":
                filtered_input = filtered_gaussian_multiplication_explicit_update(update_input, belief, self.bias_strength)
                if self.evolving_uncertainty == True:
                    new_belief = filtered_gaussian_multiplication_explicit_update(belief, filtered_input, self.trust)
                else:
                    new_belief = filtered_gaussian_multiplication_unshared(belief, filtered_input, self.trust)
            
            case "filter_belief":
                filtered_belief = filtered_gaussian_multiplication_explicit_update(belief, belief, self.bias_strength)
                if self.evolving_uncertainty == True:
                    new_belief = filtered_gaussian_multiplication_explicit_update(filtered_belief, update_input, self.trust)
                else:
                    new_belief = filtered_gaussian_multiplication_explicit_update(filtered_belief, update_input, self.trust, update_variance=False)

        return new_belief

  



    def get_update_input(self, belief, input, belief_uncertainty0, input_uncertainty0):
        belief_uncertainty = belief[1]
        input_mean = input[0]
        input_uncertainty = input[1]

        if self.initial_uncertainty:
            if self.shared_uncertainty:
                input_uncertainty = input_uncertainty0
            else:
                input_uncertainty = belief_uncertainty0
        else:
            if not self.shared_uncertainty:
                input_uncertainty = belief_uncertainty
        
        input = (input_mean, input_uncertainty)
        return input






