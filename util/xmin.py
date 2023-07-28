import numpy as np
from util.non_linear_fits import least_squares_fit


def find_x_min_index(y_values, x_values, function):
    results_dict = {}
    for x_min in range(len(x_values) - 3):
        data = y_values[x_min:]
        if len(data) == 0:  # Skip when data becomes empty
            continue
        x_adjusted = x_values[:len(data)]  # Adjust lags to match the size of data
        residuals, params, _ = least_squares_fit(x_adjusted, data, function)
        sse = np.sum(residuals**2)
        bic = len(data) * np.log(sse / len(data)) + len(params) * np.log(len(data))
        results_dict[x_min] = bic
    min_bic = min(list(results_dict.values()))
    return list(results_dict.keys())[list(results_dict.values()).index(min_bic)]
