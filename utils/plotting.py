from utils.summaries import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
from utils.computations.vectorised import vector_gaussian, steps, normalize

def plot_belief_evolution(belief_means):
    plt.xscale("log")
    for k in range(len(belief_means)):
        plt.plot(belief_means[k], color='green',  linewidth=0.3)

def plot_gaussian(gaussian, color='b'):
    x = np.linspace(-2, 2, 200)
    plt.plot(x, norm.pdf(x, gaussian[0], gaussian[1]), color=color)

def plot_vector_distribution(gaussian, color='b'):
    granularity = len(gaussian) - 1
    step = 4/granularity
    steps = np.array([(step*j-2) for j in range(granularity + 1)])
    plt.plot(steps, gaussian, color=color)

def plot_final_belief_distribution(belief_means):
    transposed_beliefs_mean = np.array(belief_means).T
    final_opinions = transposed_beliefs_mean[len(transposed_beliefs_mean)-1]
    plt.hist(final_opinions, 200, (-1,1), density=True, alpha=0.7, color='b')

def plot_initial_belief_distribution(belief_means):
    transposed_beliefs_mean = np.array(belief_means).T
    final_opinions = transposed_beliefs_mean[0]
    plt.hist(final_opinions, 200, (-1,1), density=True, alpha=0.7, color='b')

def plot_desity_of_opinions(belief_means, resolution=100):
    heat_map = np.zeros((len(belief_means[0]), resolution))

    for k in range(len(belief_means[0])):
        hist, _ = np.histogram(belief_means.T[k], resolution, (-1,1), density=True)
        heat_map[k] = hist
    
    plt.xscale("log")
    plt.imshow(heat_map.T, aspect="auto", origin="lower", interpolation="quadric")
    plt.yticks([0, resolution/2, resolution], [1, 0, -1])
    plt.colorbar()

def plot_final_belief_histogram(beliefs_mean, resolution=200):
    hist = get_belief_histogram(beliefs_mean, resolution)
    plt.plot(hist)
    plt.xticks([0, len(hist)/2, len(hist)], [-1, 0, 1])

def plot_number_of_opinions_summary(simulator, agent_network, interactions, param_iterator_x, param_iterator_y, param_x, param_y):
    heat_map = number_of_opinions_summary(simulator, agent_network, interactions, param_iterator_x, param_iterator_y, param_x, param_y)
    ax = sns.heatmap(heat_map, xticklabels=param_iterator_x, yticklabels=param_iterator_y)
    ax.invert_yaxis()
    plt.title('Number of final opinions', fontsize = 17)
    plt.xlabel(param_x, fontsize = 15)
    plt.ylabel(param_y, fontsize = 15)

def plot_number_of_opinions_summary_with_bias(simulator, agent_network, confirmation_biases, bounded_confidences):
    heat_map = number_of_opinions_summary_with_bias(simulator, agent_network, confirmation_biases, bounded_confidences)
    ax = sns.heatmap(heat_map, xticklabels=confirmation_biases, yticklabels=bounded_confidences)
    ax.invert_yaxis()
    plt.title('Number of final opinions', fontsize = 17)
    plt.xlabel('Integration Bias Strength', fontsize = 15)
    plt.ylabel('Bounded Confidence Parameter', fontsize = 15)

def plot_dynamics_sumary(beliefs_means):
    plt.figure(1, [20, 20])
    plt.subplot(331)
    plt.ylabel('Opinion', fontsize = 15)
    plt.xlabel('Time (iterations)', fontsize = 15)
    plt.title("Agents opinions over time")
    plot_belief_evolution(beliefs_means)
    plt.subplot(332)
    plt.ylabel('Opinion', fontsize = 15)
    plt.xlabel('Time (iterations)', fontsize = 15)
    plt.title("Opinion density over time")
    plot_desity_of_opinions(beliefs_means)
    plt.subplot(333)
    plot_final_belief_histogram(beliefs_means)
    plt.ylabel('Number of agents', fontsize = 15)
    plt.xlabel('Opinion', fontsize = 15)
    plt.title("Final opinion distribution")
    plt.show()

def plot_beliefs_kernel_density(belief_evolution, kernel_bandwidth):
    belief_kernel_density = get_belief_kernel_density(belief_evolution, kernel_bandwidth)
    plt.plot(steps, belief_kernel_density)
    plt.show()

def plot_kurtosis(belief_evolution):
    plt.xscale("log")
    plt.plot(get_kurtosis(belief_evolution))

def plot_polarisation(belief_evolution):
    plt.xscale("log")
    plt.plot(get_polarisation(belief_evolution))


    


