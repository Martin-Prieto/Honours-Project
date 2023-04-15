from computations.insights.opinions import *
from plotting.basic import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm
import seaborn as sns
from computations.probabilistic.vectorised import vector_gaussian, steps, normalize
import warnings
import pandas as pd

plt.style.use('ggplot')

def plot_number_of_opinions_evaluation_uncertainty(heat_map, evaluation_biases, initial_uncertainties, num_ticks=20):
    yticks = np.linspace(0, len(evaluation_biases) - 1, num_ticks, dtype=np.int)
    yticklabels = [evaluation_biases[idx] for idx in yticks]
    ax = sns.heatmap(heat_map, xticklabels=initial_uncertainties, yticklabels=yticklabels)
    ax.invert_yaxis()
    ax.set_yticks(yticks)
    plt.xlabel(r'$\sigma_0$', fontsize = 15)
    plt.ylabel(r'$\delta$', fontsize = 15)
    plt.show()

def plot_number_of_opinions_tolerance_uncertainty(heat_map, tolerances, initial_uncertainties):
    ax = sns.heatmap(heat_map, xticklabels=initial_uncertainties, yticklabels=tolerances)
    ax.invert_yaxis()
    plt.xlabel(r'$\sigma_0$', fontsize = 15)
    plt.ylabel(r'$\epsilon$', fontsize = 15)

def plot_number_mse_assimilation_uncertainty(heat_map, assimilation_biases, initial_uncertainties):
    ax = sns.heatmap(heat_map, xticklabels=initial_uncertainties, yticklabels=assimilation_biases)
    ax.invert_yaxis()
    plt.xlabel(r'$\sigma_0$', fontsize = 15)
    plt.ylabel(r'$\lambda$', fontsize = 15)

def plot_dynamics_sumary(beliefs_means):
    warnings.resetwarnings()
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
    plot_density_of_opinions(beliefs_means)
    plt.subplot(333)
    plot_final_belief_histogram(beliefs_means)
    plt.ylabel('Number of agents', fontsize = 15)
    plt.xlabel('Opinion', fontsize = 15)
    plt.title("Final opinion distribution")
    plt.show()

def plot_execution_times(all_computation_times, labels):
    for i in range(len(all_computation_times)):
        numbers_of_agents = [0] +  all_computation_times[i][0]
        computation_times = [0] +  all_computation_times[i][1]
        plt.plot(numbers_of_agents, computation_times, '-o', label = labels[i],)
    plt.xlabel("Number of agents simulated")
    plt.ylabel("Simulation time (seconds)")
    plt.legend(loc="upper left")
    plt.show()

def plot_varying_parameters(parameter1, parameter2, label1, label2, belief_evolutions):
    x_num = len(parameter1)
    y_num = len(parameter2)
    fig, axs = plt.subplots(x_num, y_num, sharex=True, sharey=True)
    rows = [label1.format(row) for row in parameter1]
    cols = [label2.format(col) for col in parameter2]
    
    for i in range(x_num):
        for e in range(y_num):
            ax_plot_belief_evolution(belief_evolutions[(i, e)], axs[i,e])

    pad = 5 

    for ax, col in zip(axs[0], cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')
    
    for ax, row in zip(axs[:,0], rows):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                xycoords=ax.yaxis.label, textcoords='offset points',
                size='large', ha='right', va='center', rotation='vertical')
    
    for ax in axs.flat:
        ax.set(xlabel='Time step', ylabel='Opinion')
    
    for ax in axs.flat:
        ax.label_outer()

    fig.show()

def plot_convergence_time(convergence_times, parameter_values):
    means = [np.mean(value) for value in convergence_times]
    stds = [np.std(value) for value in convergence_times]

    plt.errorbar(range(len(means)), means, yerr=stds, fmt='-o', capsize=5)
    plt.xticks(range(len(means)), [i for i in parameter_values])
    plt.xlabel(r'$\delta$')
    plt.ylabel('Iterations until convergence')
    plt.show()


def plot_varying_uncertainty(belief_evolutions, uncertainties):
    y_num = len(belief_evolutions)
    fig, axs = plt.subplots(1, y_num, sharex=True, sharey=True, figsize=(15, 4))
    cols = [r'$\sigma_0 = {}$'.format(col) for col in uncertainties]
    for i in range(y_num):
        ax_plot_belief_evolution(belief_evolutions[i], axs[i])
    
    pad = 5 
    for ax, col in zip(axs, cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

    for ax in axs.flat:
        ax.set(xlabel='Time step', ylabel='Opinion')
    
    for ax in axs.flat:
        ax.label_outer()
    fig.show()

def plot_varying_assimilation(belief_evolutions, assimilation_biases):
    y_num = len(belief_evolutions)
    fig, axs = plt.subplots(1, y_num, sharex=True, sharey=True, figsize=(15, 4))
    cols = [r'$\lambda = {}$'.format(col) for col in assimilation_biases]
    for i in range(y_num):
        ax_plot_belief_evolution(belief_evolutions[i], axs[i])
    
    pad = 5 
    for ax, col in zip(axs, cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

    for ax in axs.flat:
        ax.set(xlabel='Time step', ylabel='Opinion')
    
    for ax in axs.flat:
        ax.label_outer()
    fig.show()

def plot_varying_source_uncertainty(belief_evolutions, source_uncertainties):
    y_num = len(belief_evolutions)
    fig, axs = plt.subplots(1, y_num, sharex=True, sharey=True, figsize=(15, 4))
    cols = [r'$\sigma_S = {}$'.format(col) for col in source_uncertainties]
    for i in range(y_num):
        ax_plot_belief_evolution(belief_evolutions[i], axs[i])
    
    pad = 5 
    for ax, col in zip(axs, cols):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                xycoords='axes fraction', textcoords='offset points',
                size='large', ha='center', va='baseline')

    for ax in axs.flat:
        ax.set(xlabel='Time step', ylabel='Opinion')
    
    for ax in axs.flat:
        ax.label_outer()
    fig.show()

def diversity_densities(df, parameter, xlim=None, bw_adjust=.5, red=False, ylabel="Density p"):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        # Initialize the FacetGrid object
    if red:
        pal = sns.cubehelix_palette(len(pd.unique(df[parameter])),start=1.2, rot=-.25, light=.7,dark=.4)
    else:
        pal = sns.cubehelix_palette(len(pd.unique(df[parameter])), rot=-.25, light=.7)
    g = sns.FacetGrid(df, row=parameter, hue=parameter, aspect=15, height=.5, palette=pal, sharey=False, xlim=xlim)

        # Draw the densities in a few steps
    g.map(sns.kdeplot,  "Diversity",
        bw_adjust=bw_adjust, clip_on=False,
        fill=True, alpha=1, linewidth=0.15)
    g.map(sns.kdeplot, "Diversity", clip_on=False, color="w", lw=2, bw_adjust=bw_adjust)

        # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    g.map(label, "Diversity")

        # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)
        # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.fig.text(0.095, 0.5, ylabel, va='center', rotation='vertical')
    g.despine(bottom=True, left=True)

   
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)
    

def disagreement_densities(df, parameter, xlim=None, bw_adjust=.5, red=False, ylabel="Density p"):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        # Initialize the FacetGrid object
    if red:
        pal = sns.cubehelix_palette(len(pd.unique(df[parameter])),start=1.2, rot=-.25, light=.7,dark=.4)
    else:
        pal = sns.cubehelix_palette(len(pd.unique(df[parameter])), rot=-.25, light=.7)
    g = sns.FacetGrid(df, row=parameter, hue=parameter, aspect=15, height=.5, palette=pal, sharey=False, xlim=xlim)

        # Draw the densities in a few steps
    g.map(sns.kdeplot,  "Disagreement",
        bw_adjust=bw_adjust, clip_on=False,
        fill=True, alpha=1, linewidth=0.15)
    g.map(sns.kdeplot, "Disagreement", clip_on=False, color="w", lw=2, bw_adjust=bw_adjust)

        # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)


    g.map(label, "Disagreement")

        # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-.25)
        # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.fig.text(0.095, 0.5, ylabel, va='center', rotation='vertical')
    g.despine(bottom=True, left=True)

   
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color,
            ha="left", va="center", transform=ax.transAxes)
