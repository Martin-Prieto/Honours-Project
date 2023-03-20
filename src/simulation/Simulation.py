from tqdm import trange
from computations.insights.relative import mse

class Simulation:
    def __init__(self, iterations, insights):
        self.iterations = iterations
        self.insights = insights

    def run(self, interactions, agent_network, description ="Progress: "):
        agent_network.reset_opinions()
        agent_network.reset_network()
        self.insights.initialise_insights(agent_network,interactions, self.iterations)
        for k in trange(2, self.iterations + 2, desc=description):
            for agent in agent_network.agents:
                interactions.update(agent_network, agent)
            self.insights.update_insights(agent_network, interactions, k)
            