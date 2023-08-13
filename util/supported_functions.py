import numpy as np

# Pure power-law function
def pure_powerlaw(x, C, alpha):
    return C * x ** -(alpha)


# Alternative heavy-tailed functions

# Powerlaw with cut-off
def powerlaw_with_cutoff(x, alpha, lambda_, C):
    return C * x ** -(alpha) * np.exp(-lambda_ * x)

# Exponential
def exponential_function(x, beta, lambda_):
    return beta * np.exp(-lambda_ * x)

# Stretched Exponential
def stretched_exponential(x, beta, lambda_):
    return np.exp(-(x / lambda_) ** beta)


# Log-normal
def lognormal_function(x, mu, sigma):
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))