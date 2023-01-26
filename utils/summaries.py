import numpy as np
from scipy.signal import find_peaks
from updates.UpdateRule import UpdateRule
from agents.AgentNetwork import AgentNetwork
from tqdm import tqdm


def get_belief_histogram(beliefs_mean, resolution=500):
    heat_map = np.zeros((len(beliefs_mean[0]), resolution))
    transposed_beliefs_mean = np.array(beliefs_mean).T
    final_opinions = transposed_beliefs_mean[len(transposed_beliefs_mean)-1]
    for k in range(len(beliefs_mean[0])):
        hist, _ = np.histogram(beliefs_mean.T[k], bins=resolution)
        heat_map[k] = hist
    hist, _ = np.histogram(final_opinions,  200, (-1,1),density=True)
    return hist

def get_number_of_peaks(beliefs_mean, resolution=500, height=1):
    found_peaks = find_peaks(get_belief_histogram(beliefs_mean, resolution), height)
    number_of_peaks = len(found_peaks[1].get('peak_heights'))
    return number_of_peaks

def number_of_opinions_summary(simulator, agent_network, sigmas, bounded_confidences):
    heat_map = np.zeros((len(sigmas), len(bounded_confidences)))
    for i in tqdm(range(len(sigmas)), desc=" outer",  position=0):
        j = 0
        while j < len(bounded_confidences):
            try:
                agent_network.set_single_belief_uncertainties(sigmas[i])
                agent_network.set_uniform_belief_means()
                agent_network.set_beliefs_to_initial()
                update_rule = UpdateRule(0,bounded_confidences[j], 0, 0)
                beliefs = simulator.simulate(update_rule.bidirectional_update, agent_network)
                heat_map[i][j] = get_number_of_peaks(beliefs)
                j += 1
            except:
                print("error")
                continue
    return heat_map

def number_of_opinions_summary_with_bias(simulator, agent_network, confirmation_biases, bounded_confidences):
    heat_map = np.zeros((len(confirmation_biases), len(bounded_confidences)))
    for i in tqdm(range(len(confirmation_biases)), desc=" outer",  position=0):
        j = 0
        while j < len(bounded_confidences):
            try:
                agent_network.set_beliefs_to_initial()
                update_rule = UpdateRule(confirmation_biases[i],bounded_confidences[j], 0, 0)
                beliefs = simulator.simulate(update_rule.bidirectional_update, agent_network)
                heat_map[i][j] = get_number_of_peaks(beliefs)
                j += 1
            except:
                print("error")
                continue
    return heat_map

