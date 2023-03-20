from math import sqrt, pi, exp
import numpy as np
import warnings

def multiply_gaussians(a,b):
    warnings.filterwarnings("error")
    mean_a, sigma_a = a
    mean_b, sigma_b = b
    if sigma_b == float('inf'):
        return (mean_a, sigma_a)

    try:
        new_mean = ((mean_a * sigma_b**2) + (mean_b*sigma_a**2)) / (sigma_a**2 + sigma_b**2)
        new_sigma = sqrt((sigma_a**2*sigma_b**2) / (sigma_a**2 + sigma_b**2))
    except:
        print("sigma a: " + str(sigma_a))
        print("sigma b: " + str(sigma_b))
        new_mean = mean_a
        new_sigma = sigma_a
    warnings.resetwarnings()
    
    return (new_mean, new_sigma)

def filtered_gaussian_multiplication(gaussian, gaussian_to_filter, p, update_variance=True):
    mu_1 = gaussian[0]
    mu_2 = gaussian_to_filter[0]
    sd_1 = gaussian[1]**2
    sd_2 = gaussian_to_filter[1]**2

    if sd_1 == 0:
        return ((mu_1, sd_1))

    phi = 1/(exp((mu_1 - mu_2)**2)/(2*(sd_1 + sd_2))*sqrt(2*pi*(sd_1 + sd_2)))

    if p == 1:
        p_star = 1
    else:
        p_star = (p*phi)/(p*phi + (1-p))

    new_mu = p_star*((mu_1/sd_1)+(mu_2/sd_2))/((1/sd_1) + (1/sd_2)) + (1-p_star)*mu_1

    if update_variance:
        new_sigma = sqrt(sd_1*(1-p_star*(sd_1/(sd_1+sd_2)))+p_star*(1-p_star)*((mu_1-mu_2)/(1+(sd_2/sd_1)))**2)
        return (new_mu, new_sigma)
    
    return (new_mu, gaussian[1])

def filtered_gaussian_multiplication_unshared(gaussian, gaussian_to_filter, p):
    sd = gaussian[1]
    mu_1 = gaussian[0]
    mu_2 = gaussian_to_filter[0]
    normalizing_term = 1/(sd*sqrt(2 * pi))
    try:
        inverse_gaussian_term = 1/exp(((mu_1 - mu_2)**2)/(2*sd**2))
        p_star = (p*normalizing_term*inverse_gaussian_term)/(p*normalizing_term*inverse_gaussian_term + (1 - p))
        new_mu = p_star * ((mu_1 + mu_2)/2)+(1-p_star)*mu_1
    except:
        new_mu = gaussian[0]

    return (new_mu, sd)


def filtered_gaussian_multiplication_explicit_update(gaussian, gaussian_to_filter, p, update_variance=True):
    try:
        delta_var = (gaussian_to_filter[1]/gaussian[1])**2
    except:
        return gaussian
    
    
    delta_mu = gaussian_to_filter[0] - gaussian[0]

    
    mu_1 = gaussian[0]
    sd_1 = gaussian[1]


    term = 2*(sd_1**2)*(1+delta_var)
    phi = (1/sqrt(pi*term)) * exp(-(delta_mu**2) / term)

    if p == 1:
        p_star = 1
    else:
        p_star = (p*phi) / (p*phi + (1-p))

    h = p_star * (delta_mu/(1+delta_var))

    new_mu = mu_1 + h

    if update_variance:
        k = p_star * (1 / (1 + delta_var)) * ((1 - p_star) * ((delta_mu**2)/(1+delta_var)) - sd_1**2)
        new_sd = sqrt(sd_1**2 + k)
        return (new_mu, new_sd)
    
    return (new_mu, sd_1)