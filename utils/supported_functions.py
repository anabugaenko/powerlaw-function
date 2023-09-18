import numpy as np
from inspect import signature


def powerlaw(x: float, C: float, alpha: float) -> float:
    """
    Computes the value of a pure power law function.

    Parameters
    ----------
    x : float
        Input value.
    C : float
        Scaling coefficient.
    alpha : float
        Power-law exponent. Positive values indicate a growth trend, while negative values indicate a decay trend (invers power law relation).

    Returns
    -------
    float
        Computed value of the pure power law function.
    """

    return C * x**alpha


# Alternative heavy-tailed functions


def powerlaw_with_cutoff(x: float, alpha: float, lambda_: float, C: float) -> float:
    """
    Function representing a power law with a cut-off. The sign of 'alpha' determines the trend direction
    (positive for decay, negative for growth).

    Parameters:
    x (float): Input value.
    alpha (float): Power-law exponent.
    lambda_ (float): Cut-off parameter.
    C (float): Scaling constant.

    Returns:
    float: Computed value.
    """
    return C * x**alpha * np.exp(-lambda_ * x)


def powerlaw_with_exp_svf(x: float, alpha: float, beta: float, lambda_: float) -> float:
    """
    Computes the value of a power law function modified by an exponentially slowly varying function.

    The function can be expressed as:
    f(x) = x^alpha * exp(beta * x^lambda)

    Where `x^alpha` represents the power law behavior and `exp(beta * x^lambda)` represents the exponentially slowly
    varying component. a slowly varying function refers to a function that does not have a fixed, finite limit when the
    variable approaches infinity The slowly varying function L(x) essentially captures any mild variations in the tail
    behavior that the pure power-law function x^(-α) cannot account for.

    Parameters
    ----------
    x : float
        Input value.
    alpha : float
        Power-law exponent. Positive values indicate a growth trend, while negative values represent an inverse power law relation.
    beta : float
        Coefficient for the exponential function, influencing the strength of the exponential variation.
    lambda_ : float
        Exponent for the input value `x` within the exponential function. Modulates the behavior of the exponential term.

    Returns
    -------
    float
        Computed value of the power law function with exponentially slowly varying modification.
    """

    return x**alpha * exponential_function(x, beta, lambda_)


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
    return beta * np.exp(lambda_ * x)


def stretched_exponential(x: float, beta: float, lambda_: float, growth: bool = False) -> float:
    """
    Stretched exponential function that represents both growth and decay.

    Parameters:
    x (float): Input value.
    beta (float): Power-law exponent.
    lambda_ (float): Exponential growth/decay parameter.
    growth (bool): True for growth, False for decay.

    Returns:
    float: Computed value.
    """
    if growth:
        return np.exp(((x / lambda_) ** beta))
    else:
        return np.exp(-((x / lambda_) ** beta))


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
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma**2))


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
