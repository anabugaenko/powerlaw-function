import numpy as np
from typing import List, Tuple
from scipy.stats import norm, ks_2samp



def compute_loglikelihoods(data: List[float]) -> List[float]:
    """
    Compute the log likelihood of the residuals, -0.5 * np.log(2 * np.pi * np.std(residuals) ** 2) - (residuals ** 2) / (2 * np.std(residuals) ** 2)

    Parameters:
    residuals (List[float]): The residuals for which the log likelihood is to be computed.

    Returns:
    float: The log likelihoods of the residuals.

    """
    # Compute the mean and standard deviation of the residuals
    data_mean = np.mean(data)
    data_std = np.std(data)

    # Compute the log probability density function of the residuals
    loglikelihoods = norm.logpdf(data, loc=data_mean, scale=data_std)

    return loglikelihoods


def compute_bic(log_likelihood: float, num_params: int, num_samples: int) -> float:
    """
    Compute the Bayesian Information Criterion (BIC) for a given model. Note, the BIC is computed as:

        BIC = log(n) * k - 2 * log(L)

    where n is the number of samples, k is the number of parameters, and L is the maximum likelihood.

    Parameters:
    - log_likelihood (float): The log-likelihood of the model.
    - num_params (int): The number of parameters in the model.
    - num_samples (int): The number of samples in the dataset.

    Returns:
    - float: The BIC for the model.


    """
    # Compute the BIC
    bic = np.log(num_samples) * num_params - 2 * log_likelihood

    return bic


def compute_goodness_of_fit(residuals: List[float], y_values: List[float], params: List[float],
                            model_predictions: List[float]) -> Tuple[float, float, float]:
    """
    Compute the goodness of fit of a model.

    Parameters:
    residuals (List[float]): The residuals of the model.
    y_values (List[float]): The actual observed values.
    params (List[float]): The parameters of the model.
    model_predictions (List[float]): The predicted values by the model.

    Returns:
    float: The Kolmogorov-Smirnov statistic D.
    float: The p-value of the KS test.
    float: The adjusted R-squared value.

    """
    ssr = np.sum(residuals ** 2)

    sst = np.sum((y_values - np.mean(y_values)) ** 2)

    # Compute the R-squared value
    rsquared = 1 - ssr / sst

    # Compute the Adjusted R-squared value
    n = len(y_values)
    p = len(params)
    adjusted_rsquared = 1 - (1 - rsquared) * (n - 1) / (n - p - 1)

    # Compute the KS statistic and p-value
    result = ks_2samp(y_values, model_predictions)
    ks_statistic = result.statistic

    # Compute BIC
    loglikelihoods = compute_loglikelihoods(residuals)
    loglikelihood = np.sum(loglikelihoods) # sum over all data points
    bic = compute_bic(loglikelihood, len(params), n)

    return ks_statistic, bic, adjusted_rsquared


def get_residual_loglikelihoods(first_residuals: List[float], second_residuals: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the log-likelihood of two sets of residuals.

    Parameters:
    first_residuals (List[float]): The first set of residuals.
    second_residuals (List[float]): The second set of residuals.

    Returns:
    np.ndarray: The log-likelihoods of the first set of residuals.
    np.ndarray: The log-likelihoods of the second set of residuals.

    """
    x_values = np.arange(1, len(first_residuals) + 1)

    # Trim the residuals to the same length
    if len(first_residuals) != len(second_residuals):
        min_len = min(len(first_residuals), len(second_residuals))
        first_residuals = first_residuals[-min_len:]
        second_residuals = second_residuals[-min_len:]
        x_values = x_values[-min_len:]

    # Compute the log-likelihoods
    loglikelihoods1 = compute_loglikelihoods(first_residuals)
    loglikelihoods2 = compute_loglikelihoods(second_residuals)

    return loglikelihoods1, loglikelihoods2


# Much of this function was inspired by Jeff Alstott and Aaron Clauset powerlaw code, specifically around lines 1748-1822 of
# the code at: https://github.com/jeffalstott/powerlaw/blob/master/powerlaw.py
def loglikelihood_ratio(loglikelihoods1, loglikelihoods2, nested=False, normalized_ratio=True):
    """
    This version of the function allows for both the use of normalized or unnormalized R based on the normalized_ratio
    parameter, and adjusts the calculation of p accordingly. Calculates a loglikelihood ratio and the p-value for testing
    which of two functions is more likely to have created a set of observations.

    Parameters
    ----------
    loglikelihoods1 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular functions .
    loglikelihoods2 : list or array
        The logarithms of the likelihoods of each observation, calculated from
        a particular function distribution.
    nested : bool, optional
        Whether one of the two functions that generated the
        likelihoods is a nested version of the other. False by default.
    normalized_ratio : bool, optional
        Whether to return the loglikelihood ratio, R, or the normalized
        ratio R/sqrt(n*variance). True by default.

    Returns
    -------
    R : float
        The loglikelihood ratio of the two sets of likelihoods. If positive,
        the first set of likelihoods is more likely (and so the function
        that produced them is a better fit to the data). If
        negative, the reverse is true. If normalized_ratio is True, this value
        is the normalized loglikelihood ratio, otherwise, it's the unnormalized ratio.
    p : float
        The significance of the sign of R. If below a critical value
        (typically 0.1 or .05 depending on the problem domain) the sign of R is taken to be significant. If above the
        critical value the sign of R is taken to be due to statistical
        fluctuations. p is always computed using the normalized loglikelihood ratio.
    """
    from numpy import sqrt, asarray, inf, log, mean, sum
    from scipy.special import erfc
    from scipy.stats import chi2
    from sys import float_info

    n = float(len(loglikelihoods1))

    if n == 0:
        R = 0
        p = 1
        return R, p

    loglikelihoods1 = asarray(loglikelihoods1)
    loglikelihoods2 = asarray(loglikelihoods2)

    # Clean for extreme values, if any
    min_val = log(10**float_info.min_10_exp)
    loglikelihoods1[loglikelihoods1 == -inf] = min_val
    loglikelihoods2[loglikelihoods2 == -inf] = min_val

    R = sum(loglikelihoods1 - loglikelihoods2)

    mean_diff = mean(loglikelihoods1) - mean(loglikelihoods2)
    variance = sum((loglikelihoods1 - loglikelihoods2 - mean_diff) ** 2) / n

    # Normalize R for calculating p
    R_norm = R / sqrt(n * variance)

    if nested:
        p = 1 - chi2.cdf(abs(2 * R), 1)
    else:
        p = erfc(abs(R_norm) / sqrt(2))

    # Return normalized R only if normalized_ratio is True
    if normalized_ratio:
        R = R_norm

    return R, p
