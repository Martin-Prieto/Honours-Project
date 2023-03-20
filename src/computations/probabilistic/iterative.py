import numpy as np
from math import exp

granularity = 400
step = 4/granularity
steps = np.array([(step*j-2) for j in range(granularity + 1)])

def normalize(a):
    norm = sum(a)
    normalized = [None for i in range(len(a))]
    for i in range(len(a)):
        normalized[i] = a[i] * (granularity/(4*norm))
    return normalized

def multiply(a,b):
    return np.array(normalize(np.multiply(a,b)))

def iterative_gaussian(belief):
    mean = belief[0]
    standard_deviation = belief[1]
    dist = [None for i in range(granularity)]
    for i in range(granularity):
        dist[i] = exp(-1*(steps[i]-mean)**2/(2*(standard_deviation)**2))
    return normalize(dist)


def iterative_filtered_gaussian_multiplication(gaussian, gaussian_to_filter, p):
    uniform = 100/(len(gaussian) - 1)
    filtered_gaussian = [None for i in range(len(gaussian_to_filter))]
    for i in range(len(gaussian_to_filter)):
        filtered_gaussian[i] = gaussian_to_filter[i]*p + (1-p)*uniform
    new_gaussian = [None for i in range(len(gaussian))]
    for i in range(len(gaussian_to_filter)):
        new_gaussian[i] = filtered_gaussian[i]*gaussian[i]
    new_gaussian = normalize(new_gaussian)
    return new_gaussian


def iterative_compute_mean(distribution):
    mean = 0
    for i in range(len(distribution)):
        mean += distribution[i]*steps[i]
    mean = mean/100

    return round(mean, 4)