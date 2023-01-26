def network_simple_update(agent_network, agent, bias_strength = 0, trust = 0,  rewire_probability = 0, trust_bound=1):
    observed_agent = agent_network.get_observed_agent(agent)
    agent_network.rewire_network(agent, observed_agent, trust_bound, rewire_probability)
    agent_network.update_belief(agent, observed_agent, bias_strength, trust)
    return agent_network