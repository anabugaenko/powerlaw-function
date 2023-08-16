from inspect import signature

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


# Helper Classes
class FunctionParams:
    """
    This class serves as a container for function parameters. It uses Python's introspection capabilities
    to automatically map parameters to their corresponding values for a given function.

    Parameters
    ----------
    function : callable
        The function for which the parameters are being stored. This should be a function where the first
        argument is the independent variable (commonly 'x'), followed by its parameters.

    params : list or tuple
        The parameter values for the function. These should be in the same order as in the function definition.

    Attributes
    ----------
    param_names : list
        The names of the parameters of the function, excluding the independent variable.

    Methods
    -------
    get_values():
        Returns the parameter values in the same order as `param_names`.
    """

    def __init__(self, function, params):
        self.param_names = list(signature(function).parameters.keys())[1:]  # exclude 'x'
        for name, value in zip(self.param_names, params):
            setattr(self, name, value)

    def get_values(self):
        return [getattr(self, name) for name in self.param_names]
