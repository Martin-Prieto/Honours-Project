import numpy as np
from math import exp

def normalize(a):
    granularity = len(a) - 1
    step = 4/granularity
    norm = sum(a)*step
    return [x/norm for x in a]

def multiply(a,b):
    return np.array(normalize(np.multiply(a,b)))

def vector_gaussian(belief, granularity):
    mean = belief[0]
    standard_deviation = belief[1]
    step = 4/granularity
    steps = np.array([(step*j-2) for j in range(granularity + 1)])
    dist = np.array([exp(-1*(x-mean)**2/(2*(standard_deviation)**2)) for x in steps])
    return normalize(dist)


def vector_filtered_gaussian_multiplication(gaussian, gaussian_to_filter, p):
    uniform = 1/(len(gaussian) - 1)
    gaussian = np.array(gaussian)
    gaussian_to_filter = np.array(gaussian_to_filter)
    filtered_gaussian = gaussian_to_filter*p + (1-p) * uniform
    new_gaussian = multiply(gaussian, filtered_gaussian)
    return new_gaussian