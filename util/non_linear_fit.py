import warnings
import numpy as np
from scipy.optimize import least_squares
from typing import List, Callable, Union, Tuple


def least_squares_fit(x_values: List[float], y_values: List[float], function: Callable) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fits a function or curve to the data using the least squares method.

    Parameters:
    x_values (List[float]): The independent variable values.
    y_values (List[float]): The dependent variable values.
    function (Callable): The function to fit.

    Returns:
    np.ndarray: The residuals.
    np.ndarray: The optimized parameters.
    np.ndarray: The fitted values.

    """
    num_params = function.__code__.co_argcount - 1  # Exclude the 'x' parameter
    initial_guess = [0.5] * num_params  # Initialize all parameters with 0.5

    def _residuals_log(params: np.ndarray, x_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
        model_values = function(x_values, *params)
        return y_values - model_values

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = least_squares(_residuals_log, initial_guess, args=(x_values, y_values), loss='soft_l1')
            params = result.x

        fitted_values = function(x_values, *params)
        residuals = y_values - fitted_values
        return residuals, params, fitted_values
    except RuntimeError as e:
        print(f"Failed to fit curve for function {function.__name__}. Error: {e}")
        return None
