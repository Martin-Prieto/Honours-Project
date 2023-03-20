class BeliefDistribution:
    def __init__(self, uncertainty_distribution, mean_distribution):
        self.uncertainty_distribution = uncertainty_distribution
        self.mean_distribution = mean_distribution

    def generate_mean(self, size):
        return self.mean_distribution.generate(size)
    
    def generate_uncertainty(self, size):
        return self.uncertainty_distribution.generate(size)

