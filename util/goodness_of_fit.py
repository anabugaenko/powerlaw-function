import numpy as np
from scipy import stats
from typing import List, Tuple


from util.utils import block_print, enable_print


def loglikelihoods(data: List[float]) -> List[float]:
    """
    TODO: Issue #10.
    Compute the log likelihood of the data, hence  incorporates
    the variance of the data and assumes a certain distribution

    Parameters:
    data (List[float]): The data for which the log likelihood is to be computed.

    Returns:
    float: The log likelihoods of the data.

    """

    # Compute the standard deviation of the data as an initial parameter
    data_std = np.std(data)

    # Compute the log probability density function of the data
    loglikelihoods = stats.norm.logpdf(data, loc=0, scale=data_std)

    return loglikelihoods


def compute_bic_from_loglikelihood(log_likelihood: float, num_params: int, num_samples: int) -> float:
    """
    Compute the Bayesian Information Criterion (BIC) for a given model. Note, the BIC is computed as:

        BIC = log(n) * k - 2 * log(L)

    where n is the number of samples, k is the number of parameters, and L is the maximum likelihood
    that incorporates the variance of the data and assumes a certain distribution (usually normal).

    Parameters:
    - log_likelihood (float): The log-likelihood of the model.
    - num_params (int): The number of parameters in the model.
    - num_samples (int): The number of samples in the dataset.

    Returns:
    - float: The BIC for the model.
    """

    # Compute the BIC
    BIC = np.log(num_samples) * num_params - 2 * log_likelihood

    return BIC


def compute_bic_from_residuals(residuals: np.ndarray, num_parameters: int) -> float:
    """
    Compute the Bayesian Information Criterion (BIC) given the residuals using:

        BIC = n * log(RSS/n) + k * log(n)

    where n is the number of samples, k is the number of parameters and RSS is the residual sum of squares.
    In this way, it uses the residuals directly without making any assumptions about the
    distribution of the data (or residuals).


    Parameters:
    residuals (np.ndarray): The residuals (difference between actual and predicted values).
    num_parameters (int): The number of parameters in the model.

    Returns:
    float: The computed BIC value.

    """

    n = len(residuals)
    RSS = np.sum(residuals**2)
    BIC = n * np.log(RSS / n) + num_parameters * np.log(n)
    return BIC


def compute_rsquared(residuals: np.ndarray, y_values: np.ndarray, params: List[float]) -> Tuple[float, float]:
    """
    Compute the R-squared and adjusted R-squared values.

    Parameters:
    - residuals (np.ndarray): The residuals of the model.
    - y_values (np.ndarray): The actual observed values.
    - params (List[float]): The parameters of the model.

    Returns:
    - float: The R-squared value.
    - float: The adjusted R-squared value.
    """

    ssr = np.sum(residuals**2)
    sst = np.sum((y_values - np.mean(y_values)) ** 2)

    if sst == 0:
        raise ValueError("SST (total sum of squares) is zero, can't compute R-squared.")

    rsquared = 1 - ssr / sst

    # Compute the Adjusted R-squared value
    n = len(y_values)
    p = len(params)
    adjusted_rsquared = 1 - (1 - rsquared) * (n - 1) / (n - p - 1)

    return rsquared, adjusted_rsquared


def get_goodness_of_fit(
    residuals: List[float],
    y_values: List[float],
    params: List[float],
    model_predictions: List[float],
    bic_method="residuals",
    verbose=False,
) -> Tuple[float, float, float, float]:
    """
    Compute the goodness of fit of a model.

    Parameters:
    - residuals (List[float]): The residuals of the model.
    - y_values (List[float]): The actual observed values.
    - params (List[float]): The parameters of the model.
    - model_predictions (List[float]): The predicted values by the model.
    - bic_method (str): The method to use for BIC computation ('log_likelihood' or 'residuals'). Default is 'residuals'.

    Returns:
    - float: The Kolmogorov-Smirnov statistic, D.
    - float: The BIC.
    - float: MAPE metric.
    - float: The adjusted R-squared value.
    """
    original_stdout = block_print() if not verbose else None
    ks_statistic = None
    bic = None
    mape = None
    adjusted_rsquared = None

    try:
        # Compute the R-squared value
        rsquared, adjusted_rsquared = compute_rsquared(residuals, y_values, params)

        # MAPE metric is an error metric that is less sensitive to outliers than root Mean Squared Error (MAE).
        # from sklearn.metrics import mean_absolute_error
        # mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_values - model_predictions) / y_values)) * 100

        # Compute the KS statistic and p-value
        result = stats.ks_2samp(y_values, model_predictions, alternative="two-sided")
        ks_statistic = result.statistic

        # Compute BIC
        n = len(y_values)
        p = len(params)
        if bic_method == "log_likelihood":
            log_likelihoods = loglikelihoods(residuals)
            loglikelihood = np.sum(log_likelihoods)  # sum over all data points
            bic = compute_bic_from_loglikelihood(loglikelihood, len(params), n)
        else:
            bic = compute_bic_from_residuals(residuals, p)

    except Exception as e:
        enable_print(original_stdout)
        print(e)
    finally:
        enable_print(original_stdout)


    return ks_statistic, bic, mape, adjusted_rsquared


def get_residual_loglikelihoods(
    first_residuals: np.ndarray, second_residuals: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the log-likelihood of two sets of residuals. If the two sets
    have different lengths, they are trimmed to the length of the shorter set.

    Parameters:
    first_residuals (np.ndarray): The first set of residuals.
    second_residuals (np.ndarray): The second set of residuals.

    Returns:
    Tuple[np.ndarray, np.ndarray]: The log-likelihoods of the first and second set of residuals.
    """

    # Trim the residuals to the same length
    if len(first_residuals) != len(second_residuals):
        min_len = min(len(first_residuals), len(second_residuals))
        first_residuals = first_residuals[-min_len:]
        second_residuals = second_residuals[-min_len:]

    # Compute the log-likelihoods
    loglikelihoods1 = loglikelihoods(first_residuals)
    loglikelihoods2 = loglikelihoods(second_residuals)

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
