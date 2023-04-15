from society.agents import *
from society.structure import *
from society.beliefs import *
from society.structure.network import *
from updates import *
from simulation import *
import time

def get_execution_times(numbers_of_agents, iterations, simulation_type):
    execution_times = []
    agent_type = get_agent_type(simulation_type)
    linespace = Distribution(type="linespace", range=(-1,1))
    unique = Distribution(type="unique", value=0.05)
    belief_distribution = BeliefDistribution(unique, linespace)
    update_rule = UpdateRule()
    insights = Insights()
    interactions = Interactions(update_rule, interacting_agents=True)
    simulation = Simulation(iterations, insights)

    for number_of_agents in numbers_of_agents:
        network = ArtificialNetwork(number_of_agents, "fully_connected")
        agent_network = AgentNetwork(belief_distribution, network, agent_type=agent_type)
        start_time = time.time()
        simulation.run(interactions, agent_network)
        execution_time = time.time() - start_time
        execution_times.append(execution_time)
        print(str(number_of_agents) + " agents, time: " + str(execution_time))

    return execution_times


def get_agent_type(simulation_type):
    agent_type = None
    match simulation_type:
        case "iterative":
            agent_type = IterativeAgent
        case "vectorised":
            agent_type = VectorisedAgent
        case "analytical":
            agent_type = AnalyticalAgent
    return agent_type
