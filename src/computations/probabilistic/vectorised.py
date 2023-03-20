import numpy as np
from math import exp

granularity = 400
step = 4/granularity
steps = np.array([(step*j-2) for j in range(granularity + 1)])

def normalize(a):
    norm = sum(a)
    return a * (granularity/(4*norm))

def multiply(a,b):
    return np.array(normalize(np.multiply(a,b)))

def vector_gaussian(belief):
    mean = belief[0]
    standard_deviation = belief[1]
    dist = np.array([exp(-1*(x-mean)**2/(2*(standard_deviation)**2)) for x in steps])
    return normalize(dist)

def vector_filtered_gaussian_multiplication(gaussian, gaussian_to_filter, p):
    uniform = 100/((len(gaussian) - 1))
    filtered_gaussian = gaussian_to_filter*p + (1-p) * uniform
    new_gaussian = multiply(gaussian, filtered_gaussian)
    return new_gaussian


def compute_mean(distribution):
    return round(sum(distribution * steps)/100, 4)