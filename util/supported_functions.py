from inspect import signature

import numpy as np

# Pure power-law function
def pure_powerlaw(x: float, C: float, alpha: float) -> float:
    """
    Computes the value of a pure power law function.

    Parameters
    ----------
    x : float
        Input value.
    C : float
        Scaling coefficient.
    alpha : float
        Power-law exponent. Positive values indicate a growth trend,
        while negative values indicate a decay trend.

    Returns
    -------
    float
        Computed value of the pure power law function.
    """

    return C * x ** alpha


# Alternative heavy-tailed functions

# Powerlaw with cut-off
def powerlaw_with_cutoff(x: float, alpha: float, lambda_: float, C: float) -> float:
    """
    Function representing a power law with a cut-off.
    The sign of 'alpha' determines the trend direction (positive for decay, negative for growth).

    Parameters:
    x (float): Input value.
    alpha (float): Power-law exponent.
    lambda_ (float): Cut-off parameter.
    C (float): Scaling constant.

    Returns:
    float: Computed value.
    """
    return C * x ** alpha * np.exp(-lambda_ * x)


#  Exponential
def exponential_function(x: float, beta: float, lambda_: float) -> float:
    """
    Exponential function.

    Parameters:
    x (float): Input value.
    beta (float): Scaling constant.
    lambda_ (float): Exponential decay/growth parameter.

    Returns:
    float: Computed value.
    """
    return beta * np.exp(-lambda_ * x)


#  Stretched Exponential
def stretched_exponential(x: float, beta: float, lambda_: float) -> float:
    """
    Stretched exponential function.

    Parameters:
    x (float): Input value.
    beta (float): Power-law exponent.
    lambda_ (float): Exponential decay/growth parameter.

    Returns:
    float: Computed value.
    """
    return np.exp(-((x / lambda_) ** beta))



# Log-normal
def lognormal_function(x: float, mu: float, sigma: float) -> float:
    """
    Log-normal function typically representing processes skewed towards larger values.

    Parameters:
    x (float): Input value.
    mu (float): Mean of the underlying normal distribution.
    sigma (float): Standard deviation of the underlying normal distribution.

    Returns:
    float: Computed value.
    """
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
