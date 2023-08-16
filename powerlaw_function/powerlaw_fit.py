import numpy as np
import pandas as pd
from typing import Callable
from util.constants import SUPPORTED_FUNCTIONS

from util.xmin import find_x_min_index
from util.non_linear_fit import least_squares_fit
from util.util import block_print, enable_print
from util.supported_functions import FunctionParams
from util.goodness_of_fit import compute_goodness_of_fit, loglikelihood_ratio, get_residual_loglikelihoods


# Here we provide a functional programming approach of the methods in the powerlaw_function.py Fit Class.
def _process_powerlaw_function(x_values,y_values, function: Callable, function_name: str, xmin_index: float) -> FitResult:
    xmin_x_values = x_values[xmin_index:]
    xmin_y_values = y_values[xmin_index:]

    residuals, params, fitted_values = least_squares_fit(xmin_x_values, xmin_y_values, function)
    result = FitResult(function=function, function_name=function_name, fitting_method='Nonlinear Least-squares',
                       residuals=residuals, params=params, fitted_values=fitted_values, xmin_index=xmin_index,
                       xmin=x_values[xmin_index], data=pd.DataFrame({
            'xmin_x_values': xmin_x_values,
            'xmin_y_values': xmin_y_values
        }))

    return result


def fit_powerlaw_function(x_values, y_values, xmin, functions: dict, verbose=False):
    original_stdout = block_print() if not verbose else None

    fit_results_dict = {}

    try:

        print(f'Using Nonlinear Least-squares fitting method to directly fit {functions.keys()}. \n')

        for function_name, function in functions.items():
            xmin_indx = np.where(x_values == xmin)[0][0] if xmin is not None \
                else find_x_min_index(y_values, x_values, function)
            result = _process_powerlaw_function(x_values, y_values, function, function_name, xmin_indx)
            fit_results_dict[function_name] = result

        return fit_results_dict

    except Exception as e:
        enable_print(original_stdout)
        print(e)
    finally:
        enable_print(original_stdout)

def function_compare(fit_results_dict, func_name1: str, func_name2: str, verbose=False, **kwargs):
    """
    Using residuals.
    """

    original_stdout = block_print() if not verbose else None

    try:

        powerlaw_results = fit_results_dict.get(func_name1, None)
        if powerlaw_results is None:
            print(f'Fitting results do not exist for {func_name1}, consider calling fit_powerlaw_functions')
            return None, None

        if func_name2 not in fit_results_dict.keys():
            if func_name2 in SUPPORTED_FUNCTIONS.keys():
                print(f'Fitting {func_name2}')
                xmin_index = powerlaw_results.xmin_index
                func: Callable = SUPPORTED_FUNCTIONS.get(func_name2, None)
                result = _process_powerlaw_function(func, func_name2, xmin_index)
                fit_results_dict[func_name2] = result
            else:
                print(f'Do not recognise {func_name2}, consider calling fit_powerlaw_functions')

        # Get the residuals
        power_law_residuals = fit_results_dict[func_name1].residuals
        other_residuals = fit_results_dict[func_name2].residuals

        # Compute loglikelihood from residuals
        loglikelihoods1, loglikelihoods2 = get_residual_loglikelihoods(power_law_residuals, other_residuals)

        # Compute normalised loglikelihood ratio R and p-value
        R, p = loglikelihood_ratio(loglikelihoods1, loglikelihoods2, **kwargs)

    except Exception as e:
        enable_print(original_stdout)
        print(e)

        return e
    finally:
        enable_print(original_stdout)

    enable_print(original_stdout)

    return R, p


def return_all_fitting_results(fit_results_dict):
    results_list = []
    for func_name, results in fit_results_dict.items():
        result_dict = results.to_dictionary().copy()
        func = result_dict['function']
        xmin_index = result_dict['xmin_index']
        xmin = result_dict['xmin']
        params = result_dict['params']
        adjusted_rsquared = result_dict['adjusted_rsquared']
        bic = result_dict['bic']

        summary_dict = {
            'name': func_name,
            'fitted function': func,
            'xmin_index': xmin_index,
            'xmin': xmin,
            'fitting param': params,
            'adjusted r-squared': round(adjusted_rsquared, 4),
            'bic': round(bic, 4),
        }

        results_list.append(summary_dict)

    return pd.DataFrame(results_list)






