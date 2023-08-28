import numpy as np
from scipy.stats import norm
from typing import List, Callable, Dict
from util.non_linear_fit import least_squares_fit
from util.goodness_of_fit import compute_goodness_of_fit


def find_x_min(y_values: List[float], x_values: List[float], function: Callable) -> int:
    """
    Find the value of x_min that minimizes the KS statistic.

    Parameters:
    y_values (List[float]): The dependent variable values.
    x_values (List[float]): The independent variable values.
    function (Callable): The function to fit.

    Returns:
    int: The value of x_min that minimizes the KS statistic.

    """
    results_dict = {}

    # Don't look at last xmin, as that's also the xmax, and we want to at least have few points to fit!
    for x_min_indx in range(len(x_values) - 5):
        data = y_values[x_min_indx:]

        # Skip when data becomes empty
        if len(data) == 0:
            continue

        # Adjust lags to match the size of data
        x_adjusted = x_values[x_min_indx:]
        residuals, params, model_predictions = least_squares_fit(x_adjusted, data, function)

        D, _, _ = compute_goodness_of_fit(residuals, data, params, model_predictions)
        results_dict[x_min_indx+1] = D

    min_x_min = min(results_dict, key=results_dict.get)
    return min_x_min

