import numpy as np
from math import sqrt, pi, exp
from decimal import *

granularity = 401
uniform = 1/granularity
granular = np.array([(0.01*j-2) for j in range(granularity)])

def normalize(a):
    norm = sum(a)
    return a * (granularity - 1)/(4*norm)

def multiply(a,b):
    return np.array(normalize(np.multiply(a,b)))

def gaussian(mean, standard_deviation):
    dist = np.array([exp(-1*(x-mean)**2/(2*(standard_deviation)**2)) for x in granular])
    return normalize(dist)

def bounded_updating(agent_belief, information_source, bias_strength):    
        sigma = agent_belief[1]
        mu_1 = agent_belief[0]
        mu_2 = information_source[0]
        normalizing_term = 1/(sigma*sqrt(2 * pi))
        inverse_gaussian_term = 1/exp(((mu_1 - mu_2)**2)/(2*sigma**2))
        p = (bias_strength*normalizing_term*inverse_gaussian_term)/(bias_strength*normalizing_term*inverse_gaussian_term + (1 - bias_strength))
        new_mu = p * ((mu_1 + mu_2)/2)+(1-p)*mu_1
        return (new_mu, sigma)

def multiply_gaussians(a,b):
    mean_a, sigma_a = a
    mean_b, sigma_b = b
    mean_a = float(mean_a)
    mean_b = float(mean_b)
    sigma_b = float(sigma_b)
    mean_a = float(mean_a)
    new_mean = ((mean_a * sigma_b**2) + (mean_b*sigma_a**2)) / (sigma_a**2 + sigma_b**2)
    new_sigma = sqrt((sigma_a**2*sigma_b**2) / (sigma_a**2 + sigma_b**2))
    return (new_mean, new_sigma)

def bounded_updating_extended(agent_belief, information_source, p):
    mu_1 = agent_belief[0]
    mu_2 = information_source[0]
    sd_1 = agent_belief[1]**2
    sd_2 = information_source[1]**2

    phi = 1/(exp((mu_1 - mu_2)**2)/(2*(sd_1 + sd_2))*sqrt(2*pi*(sd_1 + sd_2)))

    if p == 1:
        p_star = 1
    else:
        p_star = (p*phi)/(p*phi + (1-p))

    try:
        new_mu = p_star*((mu_1/sd_1)+(mu_2/sd_2))/((1/sd_1) + (1/sd_2)) + (1-p_star)*mu_1
        new_sigma = sqrt(sd_1*(1-p_star*(sd_1/(sd_1+sd_2)))+p_star*(1-p_star)*((mu_1-mu_2)/(1+(sd_2/sd_1)))**2)
    except:
        print("mu_1: " + str(mu_1))
        print("mu_2: " + str(mu_2))
        print("sd_1: " + str(sd_1))
        print("sd_2: " + str(sd_2))
        print("p_star: " + str(p_star))
        print("phi: " + str(phi))
        raise Exception("error")

    return (new_mu, new_sigma)

def bounded_updating_extended_replication(agent_belief, information_source, p):
    mu_1 = float(agent_belief[0])
    mu_2 = float(information_source[0])
    sd_1 = float(agent_belief[1]**2)
    sd_2 = float(information_source[1]**2)
    p = float(p)

    numerator = 1/2*pi*(sd_1 + sd_2)
    argument = ((mu_1 - mu_2)**2)/(2*(sd_1 + sd_2))
    denominator = exp(argument)     

    phi = numerator/denominator
    try:
        p_star = (p*phi)/(p*phi + (1-p))
    except:
        p_star = 1

    if (sd_2 == 0 ):
        return (mu_1, sd_1)
    
    new_mu = p_star*((mu_1/sd_1)+(mu_2/sd_2))/((1/sd_1) + (1/sd_2)) + (1-p_star)*mu_1
    new_sigma = sqrt(sd_1*(1-p_star*(sd_1/(sd_1+sd_2)))+p_star*(1-p_star)*((mu_1-mu_2)/(1+(sd_2/sd_1)))**2)

    return (new_mu, new_sigma)