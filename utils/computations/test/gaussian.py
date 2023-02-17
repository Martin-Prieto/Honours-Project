from math import sqrt, pi, exp

def multiply_gaussian_filtered_by_prior(gaussian, gaussian_to_filter, p):
    mu_1 = gaussian[0]
    mu_2 = gaussian_to_filter[0]
    sigma_1 = gaussian[1]
    sigma_2 = gaussian_to_filter[1]
    var_1 = sigma_1**2
    var_2 = sigma_2**2

    delta_mu = (mu_1 - mu_2)**2
    N1 = exp(-delta_mu/(2*var_2+var_1))/(2*pi*sigma_1*sqrt(2*var_2 + var_1**2))

    N2 = exp(-delta_mu/(2*var_2+2*var_1))/(sqrt(2*pi*(var_1 + var_2)))
    new_mu_1 = (2*mu_1*var_2 + mu_2*var_1)/(2*var_2 + var_1)

    new_mu_2 = (mu_1*var_2 + mu_2*var_1)/(var_2 + var_1)

    new_mu = (p*new_mu_1*N1 + (1-p)*new_mu_2*N2)/(p*N1 + (1-p)*N2)

    new_sigma = sqrt((var_1*var_2)/(var_1+var_2))

    return (new_mu, sigma_1)