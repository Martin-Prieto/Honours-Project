from IPython.display import clear_output
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import kurtosis
from updates.UpdateRule import UpdateRule
from agents.AgentNetwork import AgentNetwork
from tqdm import tqdm
from utils.computations.vectorised import vector_gaussian, normalize
from utils.computations.analytical import polarisation
from matplotlib import pyplot as plt


def get_belief_histogram(belief_evolution, resolution=500):
    heat_map = np.zeros((len(belief_evolution[0]), resolution))
    final_opinions = get_final_opinions(belief_evolution)
    for k in range(len(belief_evolution[0])):
        hist, _ = np.histogram(belief_evolution.T[k], bins=resolution)
        heat_map[k] = hist
    hist, _ = np.histogram(final_opinions,  100, (-1,1),density=True)
    return hist

def get_belief_kernel_density(belief_evolution, kernel_bandwidth):
    final_opinions = get_final_opinions(belief_evolution)
    add_kernel = lambda x : (x, kernel_bandwidth)
    kernel_final_opinions = list(map(add_kernel, final_opinions))
    vectorised_kernel_final_opinions = np.apply_along_axis(vector_gaussian, 1, kernel_final_opinions)
    beliefs_density = normalize(np.sum(vectorised_kernel_final_opinions, axis=0))
    return beliefs_density

def get_number_of_peaks(beliefs_mean, resolution=100, height=2):
    found_peaks = find_peaks(get_belief_histogram(beliefs_mean, resolution), height)
    number_of_peaks = len(found_peaks[1].get('peak_heights'))
    return number_of_peaks

def number_of_opinions_summary(simulator, agent_network, interactions, param_iterator_x, param_iterator_y, param_x, param_y):
    heat_map = np.zeros((len(param_iterator_y), len(param_iterator_x)))
    for i in range(len(param_iterator_x)):
        j = 0
        agent_network, interactions = update_parameter(agent_network, interactions, param_iterator_x[i], param_x)
        while j < len(param_iterator_y):
            try:
                agent_network, interactions = update_parameter(agent_network, interactions, param_iterator_y[j], param_y)
                agent_network.reset()
                belief_evolution = simulator.simulate(interactions, agent_network, show_progress=False)
                print(interactions.update_rule.trust_bound)
                plot_status(belief_evolution, i, j, param_iterator_x, param_iterator_y, param_x, param_y)
                heat_map[j][i] = get_number_of_peaks(belief_evolution)
                j += 1
            except:
                print("error")
                continue       
    return heat_map

def plot_status(belief_evolution, i, j, param_iterator_x, param_iterator_y, param_x, param_y):
    from utils.plotting import plot_dynamics_sumary
    total_iterations = len(param_iterator_y)*len(param_iterator_x)
    current_iteration = i*len(param_iterator_y) + (j + 1)
    percentage = (current_iteration/total_iterations)*100
    clear_output(wait=True)
    print(str(percentage) + "%")
    print(param_x + ": " + str(param_iterator_x[i]))
    print(param_y + ": " + str(param_iterator_y[j]))
    plot_dynamics_sumary(belief_evolution)
    plt.show()


def update_parameter(agent_network, interactions, value, parameter):
    match parameter:
        case 'uncertainty':
            agent_network.set_belief_uncertainties(value)
        case 'trust':
            interactions.update_rule.trust = value
        case 'rewire_probability':
            interactions.update_rule.rewire_probability = value
        case 'bias_strength':
            interactions.update_rule.bias_strength = value
        case 'trust_bound':
            interactions.update_rule.trust_bound = value
    return agent_network, interactions


def number_of_opinions_summary_outdated(simulator, agent_network, sigmas, bounded_confidences):
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

def get_final_opinions(belief_evolution):
    transposed_beliefs_mean = np.array(belief_evolution).T
    final_opinions = transposed_beliefs_mean[len(transposed_beliefs_mean)-1]
    return final_opinions


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


def get_kurtosis(belief_evolution):
    return kurtosis(belief_evolution, axis=0)

def get_polarisation(belief_evolution):
    polarisation_evolution = np.apply_along_axis(polarisation, 0, belief_evolution)
    return polarisation_evolution

def get_disagrement(belief_evolution):
    pass