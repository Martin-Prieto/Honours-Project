from IPython.display import clear_output
import numpy as np
from scipy.signal import find_peaks
from tqdm import tqdm
from computations.probabilistic.vectorised import vector_gaussian, normalize
from matplotlib import pyplot as plt
from society.agents import *
from society.structure import *
from society import *
from updates import *
from simulation import *
from society.beliefs import Distribution
import copy


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

def get_number_of_peaks(beliefs_mean, resolution=100, height=1):
    found_peaks = find_peaks(get_belief_histogram(beliefs_mean, resolution), height)
    number_of_peaks = len(found_peaks[1].get('peak_heights'))
    return number_of_peaks


def plot_status(belief_evolution, i, j, param_iterator_x, param_iterator_y, param_x, param_y):
    from plotting import plot_dynamics_sumary
    total_iterations = len(param_iterator_y)*len(param_iterator_x)
    current_iteration = i*len(param_iterator_y) + (j + 1)
    percentage = (current_iteration/total_iterations)*100
    clear_output(wait=True)
    print(str(percentage) + "%")
    print(param_x + ": " + str(param_iterator_x[i]))
    print(param_y + ": " + str(param_iterator_y[j]))
    plot_dynamics_sumary(belief_evolution)
    plt.show()

def get_final_opinions(belief_evolution):
    transposed_beliefs_mean = np.array(belief_evolution).T
    final_opinions = transposed_beliefs_mean[len(transposed_beliefs_mean)-1]
    return final_opinions


def varying_evaluation_assimilation(evaluation, assimilation, agent_network, interactions, simulation):
    belief_evolutions = {}
    for i in range(len(evaluation)):
        for e in range(len(assimilation)):
            update_rule = UpdateRule(assimilation[e], evaluation[i])
            interactions.update_rule = update_rule
            simulation.run(interactions, agent_network)
            belief_evolutions[(i,e)] = copy.deepcopy(simulation.insights.belief_evolution)
    return belief_evolutions

def varying_evaluation_info(evaluation, information_sources, agent_network, interactions, simulation):
    belief_evolutions = {}
    for i in range(len(evaluation)):
        for e in range(len(information_sources)):
            interactions.information_source = information_sources[i]
            simulation.run(interactions, agent_network)
            belief_evolutions[(i,e)] = copy.deepcopy(simulation.insights.belief_evolution)
    return belief_evolutions

def varying_source(standard_deviations, agent_network, interactions, simulation):
    belief_evolutions = []
    mean = interactions.information_source[0]
    for i in range(len(standard_deviations)):
        sd = standard_deviations[i]
        interactions.information_source = (mean, sd)
        simulation.run(interactions, agent_network)
        belief_evolutions.append(copy.deepcopy(simulation.belief_evolution))
    return belief_evolutions

def number_of_opinions_evaluation_uncertainty(evaluation_biases, initial_uncertainties, agent_network, simulation, interactions, repetitions=1):
    uncertainties_num = len(initial_uncertainties)
    evaluation_num = len(evaluation_biases)
    heat_map = np.zeros((evaluation_num, uncertainties_num))
    for i in range(evaluation_num):
        for j in range(uncertainties_num):
            unique = Distribution(type="unique", value=initial_uncertainties[j])
            agent_network.set_uncertainties(unique)
            interactions.update_rule.evaluation_bias = evaluation_biases[i]
            number_of_peaks = []
            for _ in range(repetitions):
                simulation.run(interactions, agent_network, description=f'{i} of {evaluation_num}|{j} of {uncertainties_num}')
                number_of_peaks.append(get_number_of_peaks(simulation.insights.belief_evolution))
            heat_map[i][j] = sum(number_of_peaks)//len(number_of_peaks)
    return heat_map

def number_of_opinions_seeking_uncertainty(tolerances, initial_uncertainties, agent_network, simulation, interactions, repetitions=1):
    uncertainties_num = len(initial_uncertainties)
    evaluation_num = len(tolerances)
    heat_map = np.zeros((evaluation_num, uncertainties_num))
    for i in range(evaluation_num):
        for j in range(uncertainties_num):
            unique = Distribution(type="unique", value=initial_uncertainties[j])
            agent_network.set_uncertainties(unique)
            interactions.update_rule.tolerance = tolerances[i]
            number_of_peaks = []
            for _ in range(repetitions):
                simulation.run(interactions, agent_network, description=f'{i} of {evaluation_num}|{j} of {uncertainties_num}')
                number_of_peaks.append(get_number_of_peaks(simulation.insights.belief_evolution))
            heat_map[i][j] = sum(number_of_peaks)//len(number_of_peaks)
    return heat_map

def convergence_times_evaluation(evaluation_biases, agent_network, interactions, repetitions=1):
    all_times = []
    insights = Insights(["convergence"])
    simulation = Simulation(2000, insights)
    for evaluation_bias in evaluation_biases:
        interactions.update_rule.evaluation_bias = evaluation_bias
        computation_times = []
        for i in range(repetitions):
            simulation.run(interactions, agent_network)
            computation_times.append(simulation.insights.convergence_time)
        all_times.append(computation_times)
    return all_times


def mse_assimilation_uncertainty(assimilation_biases, initial_uncertainties, agent_network, simulation, interactions, repetitions=1):
    uncertainties_num = len(initial_uncertainties)
    evaluation_num = len(assimilation_biases)
    heat_map = np.zeros((evaluation_num, uncertainties_num))
    for i in range(evaluation_num):
        for j in range(uncertainties_num):
            unique = Distribution(type="unique", value=initial_uncertainties[j])
            agent_network.set_uncertainties(unique)
            interactions.update_rule.assimilation_bias = assimilation_biases[i]
            mses = []
            for _ in range(repetitions):
                simulation.run(interactions, agent_network, description=f'{i} of {evaluation_num}|{j} of {uncertainties_num}')
                mses.append(simulation.insights.mse_evolution[simulation.iterations])
            heat_map[i][j] = sum(mses)/len(mses)
    return heat_map

def mse_seeking_uncertainty(rewire_probability, initial_uncertainties, agent_network, simulation, interactions, repetitions=1):
    uncertainties_num = len(initial_uncertainties)
    evaluation_num = len(rewire_probability)
    heat_map = np.zeros((evaluation_num, uncertainties_num))
    for i in range(evaluation_num):
        for j in range(uncertainties_num):
            unique = Distribution(type="unique", value=initial_uncertainties[j])
            agent_network.set_uncertainties(unique)
            interactions.update_rule.rewire_probability = rewire_probability[i]
            mses = []
            for _ in range(repetitions):
                simulation.run(interactions, agent_network, description=f'{i} of {evaluation_num}|{j} of {uncertainties_num}')
                mses.append(simulation.insights.mse_evolution[simulation.iterations])
            heat_map[i][j] = sum(mses)/len(mses)
    return heat_map