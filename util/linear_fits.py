import numpy as np
import statsmodels.api as sm
from typing import List, Union, Tuple


def linear_fit(x_values: List[float], y_values: List[float], model_type: str) -> Union[None, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Fit a linear model to the log-transformed data.

    Parameters:
    x_values (List[float]): The independent variable values.
    y_values (List[float]): The dependent variable values.
    model_type (str): The type of linear model to fit ('OLS', 'RLM', or 'GLS').

    Returns:
    np.ndarray: The residuals.
    np.ndarray: The parameters.
    np.ndarray: The fitted values.

    """
    try:
        log_x_values = np.log(x_values)
        log_y_values = np.log(y_values)

        X = sm.add_constant(log_x_values)

        if model_type == 'OLS':
            model = sm.OLS(log_y_values, X)
        elif model_type == 'RLM':
            model = sm.RLM(log_y_values, X, M=sm.robust.norms.HuberT())
        elif model_type == 'GLS':
            model = sm.GLS(log_y_values, X)
        else:
            raise ValueError(f'Unknown model type: {model_type}')

        results = model.fit()
        params = results.params  # const, slope
        fitted_values = log_x_values * params[1] + params[0]
        residuals = log_y_values - fitted_values
        return residuals, params, fitted_values
    except Exception as e:
        print(f"Failed to fit curve for model {model_type}. Error: {e}")
        return None

