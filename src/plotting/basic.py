from computations.insights.opinions import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
from computations.insights.relative import polarisation_evolution, disagreement_evolution, kurtosis
from computations.probabilistic.vectorised import *
from computations.insights.opinions import get_belief_kernel_density

plt.style.use('ggplot')

def plot_belief_evolution(belief_means):
    plt.xscale("log")
    plt.xlim(1, len(belief_means[0]))
    for k in range(len(belief_means)):
        plt.plot(belief_means[k], color='royalblue',  linewidth=1.2, alpha=0.5)

def plot_belief_evolution_non_log(belief_means):
    plt.xlim(1, len(belief_means[0]))
    for k in range(len(belief_means)):
        plt.plot(belief_means[k], color='blue',  linewidth=0.3)

def plot_final_belief_distribution(belief_means):
    transposed_beliefs_mean = np.array(belief_means).T
    final_opinions = transposed_beliefs_mean[len(transposed_beliefs_mean)-1]
    plt.hist(final_opinions, 200, (-1,1), density=True, alpha=0.7, color='b')

def plot_initial_belief_distribution(belief_means):
    transposed_beliefs_mean = np.array(belief_means).T
    final_opinions = transposed_beliefs_mean[0]
    plt.hist(final_opinions, 200, (-1,1), density=True, alpha=0.7, color='b')

def plot_density_of_opinions(belief_means, resolution=50,interpolation="None", qmax=0.999):
    plt.grid(False)
    histogram = np.zeros((len(belief_means[0]), resolution))

    for k in range(len(belief_means[0])):
        hist, _ = np.histogram(belief_means.T[k], resolution, (-1,1), density=True)
        histogram[k] = hist

    plt.xscale('log')
    plt.imshow(histogram.T, aspect="auto", interpolation=interpolation , vmax=np.quantile(histogram, qmax))
    plt.yticks([0, resolution/2, resolution-1], [1, 0, -1])
    plt.colorbar()

def plot_final_belief_histogram(beliefs_mean, resolution=200):
    hist = get_belief_histogram(beliefs_mean, resolution)
    plt.plot(hist)
    plt.xticks([0, len(hist)/2, len(hist)], [-1, 0, 1])

def plot_beliefs_kernel_density(belief_evolution, kernel_bandwidth):
    belief_kernel_density = get_belief_kernel_density(belief_evolution, kernel_bandwidth)
    plt.plot(steps, belief_kernel_density)
    plt.show()

def plot_kurtosis(belief_evolution):
    plt.xscale("log")
    plt.plot(kurtosis(belief_evolution))

def plot_polarisation(belief_evolution):
    plt.xscale("log")
    plt.plot(polarisation_evolution(belief_evolution))

def plot_disagreement(belief_evolution, edge_evolution):
    plt.xscale("log")
    plt.plot(disagreement_evolution(belief_evolution, edge_evolution))

def ax_plot_belief_evolution(belief_means, ax):
    ax.set_xscale("log")
    ax.set_xlim(1, len(belief_means[0]))
    for k in range(len(belief_means)):
        ax.plot(belief_means[k],  color='royalblue',  linewidth=1, alpha=0.7)


def ax_plot_density_of_opinions(belief_means, ax, resolution=50,interpolation="None", qmax=0.999):
    ax.grid(False)
    histogram = np.zeros((len(belief_means[0]), resolution))

    for k in range(len(belief_means[0])):
        hist, _ = np.histogram(belief_means.T[k], resolution, (-1,1), density=True)
        histogram[k] = hist

    ax.set_xscale("log")
    ax.set_yticks([0, resolution/2, resolution-1], [1, 0, -1])
    return ax.imshow(histogram.T, aspect="auto", interpolation=interpolation , vmax=np.quantile(histogram, qmax))
    