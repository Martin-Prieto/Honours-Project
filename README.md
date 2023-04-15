# Honours-Project
Here are the the materials supporting the project: Polarisation and Disagreement in Opinion
Dynamics based on Confirmation Biases

# Structure
 
 The implementation of the model, and other code required for visualisations and computations is contained in the src folder. Within this folder, there are 6 other folders:

 - computations: containing operations used in the model and operations providing insights (Diversity, Disagreement, Convergence time ...).

 - plotting: containing functions used for visualizing different properties of the opinion space (Evolution of mean beliefs, belief density, disagreement densities ...).

 - simulation: contains classes handling the simulations at a high level, controling things such as number of iterations or the insights to be computed.

 - society: contains classes used to model agents in a network:
    - agents: contains classes for agents with beliefs handeled in different ways (analytical, vextorised, iterative)
    - beliefs: contains classes for generating the initial distribution of beliefs
    - structure: contains classes for generating networks both artifitial and real using data from social media networks. Also contains classes for handling agents as a collective
    - updates: contains classes the opinion update rule, and for defining agent interactions (with other agents, or with an information source)
    - utils: contains other usefull functions for things such as reading files or storing results.

- notebooks: contains jupyter notebooks for replicating the results shown in the project organized by chapters.

- data: contains precomputed data for some results taking a long time to compute. Data for the validation section hasn't been included as it exceeds the recomended file size. Note that these results can be computed by running the notebooks. 


# Installation requirements

In order to run the notebooks, a number of external libraries need to be installed. These are listed in the requirements.txt file.
