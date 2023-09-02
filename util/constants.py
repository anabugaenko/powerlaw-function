from scipy.stats import laplace, norm, t, expon, lognorm, powerlaw, gumbel_l, gumbel_r
from util import supported_functions as sf
from util.linear_fits import linear_fit
from util.non_linear_fits import mle_fit, least_squares_fit

LINEAR_FITTING_METHODS = {
    # Mapping from name to linear fitting methods
    'OLS': lambda x, y: linear_fit(x, y, 'OLS'),
    'GeneralisedRegression': lambda x, y: linear_fit(x, y, 'GLS'),
    'RobustRegression': lambda x, y: linear_fit(x, y, 'RLM'),
}


NONLINEAR_FITTING_METHODS = {
    # Mapping from name to nonlinear fitting methods
    'MLE': mle_fit,
    'Least_squares': least_squares_fit,
}


SUPPORTED_FUNCTIONS = {
    # Pure Power law
    sf.pure_powerlaw.__name__: sf.pure_powerlaw,

    # Alternative models
    sf.powerlaw_with_cutoff.__name__: sf.powerlaw_with_cutoff,
    sf.exponential_function.__name__: sf.exponential_function,
    sf.stretched_exponential.__name__: sf.stretched_exponential,
    sf.lognormal_function.__name__: sf.lognormal_function
}
