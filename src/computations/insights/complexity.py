from society.agents import *
from society.structure import *
from society.beliefs import *
from updates import *
from simulation import *
import time

def get_execution_times(numbers_of_agents, iterations, simulation_type):
    execution_times = []
    agent_type = get_agent_type(simulation_type)
    unique = Distribution(type="unique", value=0.2)
    linespace = Distribution(type="linespace", range=(-1,1))
    update_rule = UpdateRule()
    interactions = Interactions(update_rule, interacting_agents=True)
    simulation = Simulation(iterations)

    for number_of_agents in numbers_of_agents:
        agent_network = AgentNetwork(number_of_agents, unique, unique, linespace, agent_type=agent_type)
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
