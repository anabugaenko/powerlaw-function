from utils.linear_fits import linear_fit
from utils import supported_functions as sf
from utils.non_linear_fits import mle_fit, least_squares_fit
from scipy.stats import laplace, norm, t, expon, lognorm, powerlaw, gumbel_l, gumbel_r

LINEAR_FITTING_METHODS = {
    # Mapping from name to linear fitting methods
    "OLS": lambda x, y: linear_fit(x, y, "OLS"),
    "GeneralisedRegression": lambda x, y: linear_fit(x, y, "GLS"),
    "RobustRegression": lambda x, y: linear_fit(x, y, "RLM"),
}


NONLINEAR_FITTING_METHODS = {
    # Mapping from name to nonlinear fitting methods
    "MLE": mle_fit,
    "Least_squares": least_squares_fit,
}


SUPPORTED_FUNCTIONS = {
    # Pure Power law
    sf.powerlaw.__name__: sf.powerlaw,
    # Alternative models
    sf.powerlaw_with_cutoff.__name__: sf.powerlaw_with_cutoff,
    sf.powerlaw_with_exp_svf.__name__: sf.powerlaw_with_exp_svf,
    sf.exponential_function.__name__: sf.exponential_function,
    sf.stretched_exponential.__name__: sf.stretched_exponential,
    sf.lognormal_function.__name__: sf.lognormal_function,
}
