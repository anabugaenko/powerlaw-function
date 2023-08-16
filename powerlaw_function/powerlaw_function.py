import numpy as np
import pandas as pd
from typing import Callable
from matplotlib import pyplot as plt
from numpy import asarray, isinf, isnan
from util.constants import LINEAR_FITTING_METHODS, SUPPORTED_FUNCTIONS

from util.xmin import find_x_min_index
from util import supported_functions as sf
from util.non_linear_fit import least_squares_fit
from util.util import block_print, enable_print
from util.supported_functions import FunctionParams
from util.goodness_of_fit import compute_goodness_of_fit, loglikelihood_ratio, get_residual_loglikelihoods


class FitResult:
    """
    Represents the result of a fitting procedure.

    Attributes:
        function_name (str): Name of the fitted function.
        residuals (list): Residuals after fitting.
        fitted_values (list): Values of the fitted function.
        function (Callable): The actual function used for fitting.
        fitting_method (str): Method used for fitting.
        xmin_index (int): Index of the minimum x value.
        xmin (float): Minimum x value.
        data (DataFrame): Data used for fitting.
        params (FunctionParams): Parameters of the function.
        adjusted_rsquared (float): Adjusted R squared value.
        bic (float): BIC value.
    """
    def __init__(self, function, function_name, fitting_method, residuals, params, fitted_values, xmin_index, xmin,
                 data):
        self.function_name = function_name
        self.residuals = residuals
        self.fitted_values = fitted_values
        self.function = function
        self.fitted_function = function_name
        self.fitting_method = fitting_method
        self.xmin_index = xmin_index
        self.xmin = xmin
        self.data = data
        self.params = FunctionParams(function, params)
        self.adjusted_rsquared, self.bic = compute_goodness_of_fit(self.residuals, self.data.xmin_y_values,
                                                                   self.params.__dict__)

    def to_dictionary(self):
        """
        Converts the object attributes to a dictionary.

        Returns:
            dict: A dictionary representation of the FitResult object.
        """
        return {
            'function_name': self.function_name,
            'residuals': self.residuals,
            'params': self.params.__dict__,
            'function': self.fitted_function,
            'fitted_values': self.fitted_values,
            'xmin_index': self.xmin_index,
            'xmin': self.xmin,
            'data': self.data,
            'bic': self.bic,
            'adjusted_rsquared': self.adjusted_rsquared,
        }

    def print_fitted_results(self):
        print('')
        print(f'For {self.fitted_function} fitted using {self.fitting_method};')
        print('')
        print('Pre-fitting parameters:')
        print(f'xmin: {self.xmin}')

        print('')
        print('Fitting parameters:')
        params_dict = self.params.__dict__
        for param_name, param_value in params_dict.items():
            print(f'{param_name} = {param_value}')

        print('')
        print('Goodness of fit to data:')
        print('BIC =', self.bic)
        print('Adjusted R-squared =', self.adjusted_rsquared)
        print('\n')

    def _plot_data(self, scale='loglog', data_kwargs=None, fit_kwargs=None, figure_kwargs=None):
        """
        Plots the fitted data along with the raw data.

        Args:
            data_kwargs (dict): Arguments for plotting raw data.
            fit_kwargs (dict): Arguments for plotting fitted data.
            figure_kwargs (dict): Arguments for the figure.
            scale (str): Scale for the plot. Can be 'loglog' or 'linear'.
        """

        # Assigning relevant values
        func_name = self.fitted_function

        # Create plot with figure_kwargs
        plt.figure(**(figure_kwargs if figure_kwargs else {}))

        # Plot raw data according to the specified scale
        plot_func = plt.loglog if scale == 'loglog' else plt.plot if scale == 'linear' else None
        if plot_func is None:
            raise ValueError(f"Invalid scale value: {scale}. Must be either 'loglog' or 'linear'.")

        # Plots raw data
        plot_func(self.data.xmin_x_values, self.data.xmin_y_values, label='Raw data',
                  **(data_kwargs if data_kwargs else {}))

        # Plots fitted function on top of raw data
        plot_func(self.data.xmin_x_values, self.function(self.data.xmin_x_values, *(self.params.get_values())),
                  label=f'Fitted function: {func_name}', **(fit_kwargs if fit_kwargs else {}))

        # update legend with fit_kwargs if provided
        plt.legend(**(fit_kwargs if fit_kwargs else {'frameon': False}))

        plt.grid(False)
        plt.show()

    def plot_fit(self, data_kwargs=None, fit_kwargs=None, figure_kwargs=None, scale='loglog'):
        self._plot_data(data_kwargs=data_kwargs, fit_kwargs=fit_kwargs, figure_kwargs=figure_kwargs, scale=scale)


class Fit:
    def __init__(self, data: pd.DataFrame, xmin=None, verbose=False):
        """
        Represents a fitting process on given data.

        Attributes:
            x_values (list): X values of the data.
            y_values (list): Y values of the data.
            fit_results_dict (dict): Dictionary to store fitting results for different functions.
            verbose (bool): If True, prints additional logs.
        """

        # Ensure the DataFrame has exactly is 2D (has two columns)
        if len(data.columns) != 2:
            raise ValueError("Input data must have exactly two columns", file=sys.stderr)

        self.x_values = asarray(data.iloc[:, 0], dtype='float')
        self.y_values = asarray(data.iloc[:, 1], dtype='float')
        self.fit_results_dict = {}
        self.verbose = verbose

        # Ensure xmin is properly assigned
        if xmin and type(xmin) != tuple and type(xmin) != list:
            self.xmin = float(xmin)
            if verbose: print(f'Set xmin to {self.xmin}')
        else:
            self.xmin = None

        # Check for 0
        if 0 in self.x_values or 0 in self.y_values:
            if verbose:
                print("Values less than or equal to 0 in data. Throwing away these values.", file=sys.stderr)

            zero_x_indx = 0 in self.x_values
            zero_y_indx = 0 in self.y_values
            self.x_values = np.delete(self.x_values, zero_x_indx + zero_y_indx)
            self.y_values = np.delete(self.y_values, zero_x_indx + zero_y_indx)

        # Check for inf
        if any(isinf(self.x_values)) or any(isinf(self.y_values)):
            if verbose:
                print("Infinite values in data. Throwing away these values.", file=sys.stderr)

            inf_x_indx = isinf(self.x_values)
            inf_y_indx = isinf(self.y_values)
            self.x_values = np.delete(self.x_values, inf_x_indx + inf_y_indx)
            self.y_values = np.delete(self.y_values, inf_x_indx + inf_y_indx)

        # Check for nan
        if any(isnan(self.x_values)) or any(isnan(self.y_values)):
            if verbose:
                print("NaN values in data. Throwing away these values.", file=sys.stderr)

            nan_x_indx = isnan(self.x_values)
            nan_y_indx = isnan(self.y_values)
            self.x_values = np.delete(self.x_values, nan_x_indx + nan_y_indx)
            self.y_values = np.delete(self.y_values, nan_x_indx + nan_y_indx)

        # Check for negative values
        if any(self.x_values < 0) or any(self.y_values < 0):
            if verbose:
                print("Negative values in data. Throwing away these values.", file=sys.stderr)

            neg_x_indx = self.x_values < 0
            neg_y_indx = self.y_values < 0
            self.x_values = np.delete(self.x_values, neg_x_indx + neg_y_indx)
            self.y_values = np.delete(self.y_values, neg_x_indx + neg_y_indx)

        # Check if there are enough data points for the intended fits
        min_data_points = 20
        if len(self.x_values) < min_data_points or len(self.y_values) < min_data_points:
            raise ValueError(f"Insufficient data for fitting. At least {min_data_points} data points are required",
                             file=sys.stderr)

        # Fit power-laws
        self._fit_powerlaw_function()

    def __getattr__(self, name):
        """
        Custom attribute access method. Allows accessing fitting results as attributes.

        Args:
            name (str): Name of the attribute or fitting result to access.

        Returns:
            FitResult: If the name matches a fitting result.

        Raises:
            AttributeError: If the name does not match any attribute or fitting result.
        """
        if name in self.fit_results_dict.keys():
            return self.fit_results_dict[name]
        else:
            # Default behaviour
            print(f'Trying to access {name} on the fit object but {name} does not exist yet.'
                  f' Consider fitting {name} first.')
            raise AttributeError

    def _fit_powerlaw_function(self):
        # Private method for fitting a power-law function to the data.

        original_stdout = block_print() if not self.verbose else None

        try:
            print('Fitting pure_powerlaw function using Nonlinear Least-squares fitting method.')
            function, function_name = sf.pure_powerlaw, sf.pure_powerlaw.__name__
            xmin_indx = np.where(self.x_values == self.xmin)[0][0] if self.xmin is not None \
                else find_x_min_index(self.y_values, self.x_values, function)
            result = self._process_powerlaw_function(function, function_name, xmin_indx)
            self.fit_results_dict[function_name] = result

            print('Using Linear fitting methods to approximation pure_powerlaw fit on Loglog scale.')

            for method_name, fitting_method in LINEAR_FITTING_METHODS.items():
                xmin_indx = np.where(self.x_values == self.xmin)[0][0] if self.xmin is not None \
                    else find_x_min_index(self.y_values, self.x_values, function)
                result = self._process_linear_method(fitting_method, method_name, xmin_indx)
                self.fit_results_dict[method_name] = result

        except Exception as e:
            enable_print(original_stdout)
            print(e)
        finally:
            enable_print(original_stdout)

        return self.fit_results_dict

    def _process_powerlaw_function(self, function: Callable, function_name: str, xmin_index: float) -> FitResult:
        """
        Directly fit the data with a given power-law function and returns the results.

        Args:
            function (Callable): Function to fit.
            function_name (str): Name of the function.
            xmin_index (float): Index of the minimum x value.

        Returns:
            FitResult: Results of the fitting procedure.
        """
        xmin_x_values = self.x_values[xmin_index:]
        xmin_y_values = self.y_values[xmin_index:]

        residuals, params, fitted_values = least_squares_fit(xmin_x_values, xmin_y_values, function)
        result = FitResult(function=function, function_name=function_name, fitting_method='Nonlinear Least-squares',
                           residuals=residuals, params=params, fitted_values=fitted_values, xmin_index=xmin_index,
                           xmin=self.x_values[xmin_index], data=pd.DataFrame({
                'xmin_x_values': xmin_x_values,
                'xmin_y_values': xmin_y_values
            }))

        result.print_fitted_results()

        return result

    def _process_linear_method(self, fitting_method: Callable, method_name: str, xmin_index: float) -> FitResult:
        """
        Uses a linear method to fit power-law to the data on log-scale and returns the results.

        Args:
            fitting_method (Callable): Linear fitting method.
            method_name (str): Name of the method.
            xmin_index (float): Index of the minimum x value.

        Returns:
            FitResult: Results of the fitting procedure.
        """
        xmin_x_values = self.x_values[xmin_index:]
        xmin_y_values = self.y_values[xmin_index:]

        residuals, params, fitted_values = fitting_method(xmin_x_values, xmin_y_values)

        powerlaw_params = [np.exp(params[0]), -params[1]] # negative sign for alpha on linear scale, const
        powerlaw_fitted_values = sf.pure_powerlaw(xmin_x_values, *powerlaw_params)
        powerlaw_residuals = xmin_y_values - powerlaw_fitted_values
        result = FitResult(function= sf.pure_powerlaw, function_name='pure_powerlaw', fitting_method=method_name,
                           residuals=powerlaw_residuals, params=powerlaw_params, fitted_values=powerlaw_fitted_values,
                           xmin_index=xmin_index, xmin=self.x_values[xmin_index], data=pd.DataFrame({
                'xmin_x_values': xmin_x_values,
                'xmin_y_values': xmin_y_values
            }))

        result.print_fitted_results()

        return result

    def fit_powerlaw_function(self, functions: dict, xmin = None, verbose=False):
        """
        Fits the data using power-law functions.

        Args:
            functions (dict): Dictionary of functions to fit.
            xmin (): Minimum x value.
            verbose (bool): If True, prints additional logs.
        """
        original_stdout = block_print() if not verbose else None

        try:

            # Ensure xmin is properly assigned
            if xmin and type(xmin) != tuple and type(xmin) != list:
                xmin = float(xmin)
                if verbose: print(f'Set xmin to {xmin}')
            else:
                xmin = None

            for function_name, function in functions.items():
                print(f'Using Nonlinear Least-squares fitting method to directly fit {function_name}.')
                xmin_indx = np.where(self.x_values == xmin)[0][0] if xmin is not None \
                    else find_x_min_index(self.y_values, self.x_values, function)
                result = self._process_powerlaw_function(function, function_name, xmin_indx)
                self.fit_results_dict[function_name] = result

        except Exception as e:
            enable_print(original_stdout)
            print(e)
        finally:
            enable_print(original_stdout)

    def function_compare(self, func_name1: str, func_name2: str, verbose=False, **kwargs):
        """
        Compares two fitted functions based on their residuals.

        Args:
            func_name1 (str): Name of the first function.
            func_name2 (str): Name of the second function.
            verbose (bool): If True, prints additional logs.

        Returns:
            tuple: A tuple containing the normalized
        """

        original_stdout = block_print() if not verbose else None

        try:
            """
            Using residuals.
            """

            powerlaw_results = self.fit_results_dict.get(func_name1, None)
            if powerlaw_results is None:
                print(f'Fitting results do not exist for {func_name1}, consider calling fit_powerlaw_functions')
                return None, None

            if func_name2 not in self.fit_results_dict.keys():
                if func_name2 in SUPPORTED_FUNCTIONS.keys():
                    print(f'Fitting {func_name2}')
                    xmin_index = powerlaw_results.xmin_index
                    func: Callable = SUPPORTED_FUNCTIONS.get(func_name2, None)
                    result = self._process_powerlaw_function(func, func_name2, xmin_index)
                    self.fit_results_dict[func_name2] = result
                else:
                    print(f'Do not recognise {func_name2}, consider calling fit_powerlaw_functions')

            # Get the residuals
            power_law_residuals = self.fit_results_dict[func_name1].residuals
            other_residuals = self.fit_results_dict[func_name2].residuals

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

    def return_all_fitting_results(self):
        """
        Returns a summary of all the fitting results stored in the fit_results_dict.

        This method compiles relevant fitting metrics such as the fitted function,
        xmin_index, xmin, fitting parameters, adjusted R-squared, and BIC for each
        function in the fit_results_dict.

        Returns:
            pd.DataFrame: A DataFrame containing the summarized fitting results.
        """
        results_list = []
        for func_name, results in self.fit_results_dict.items():
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

    def plot_data(self, scale='loglog', kwargs=None):
        """
        Plots the data stored in the object based on the provided scale.

        The data can be plotted on either a log-log scale or a linear scale.

        Args:
            scale (str, optional): The type of scale on which to plot the data.
                Can be either 'loglog' or 'linear'. Defaults to 'loglog'.
            kwargs (dict, optional): Additional keyword arguments to pass to the plotting function.

        Raises:
            ValueError: If the provided scale is neither 'loglog' nor 'linear'.

        Returns:
            None: This method directly plots the data using matplotlib and doesn't return any value.
        """
        xmin_y_values = self.y_values[self.xmin:]
        xmin_x_values = self.x_values[self.xmin:]

        # Plot raw data according to the specified scale
        plot_func = plt.loglog if scale == 'loglog' else plt.plot if scale == 'linear' else None
        if plot_func is None:
            raise ValueError(f"Invalid scale value: {scale}. Must be either 'loglog' or 'linear'.")

        # Plot raw data
        plot_func(xmin_x_values, xmin_y_values, label='Raw data',
                  **(kwargs if kwargs else {}))

        plt.grid(False)
        plt.legend(frameon=False)
        plt.show()


if __name__ == '__main__':

    # Load sample data – TSLA stock trade signs.
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, '..', 'datasets', 'stock_tsla.csv')
    sample = pd.read_csv(csv_path, header=0, index_col=0)

    # Generate series from a function, in this example, we generate a series from an autocorrelation function (ACF)
    from typing import List
    def acf(series: pd.Series, lags: int) -> List:
        """
        Returns a list of autocorrelation values for each of the lags from 0 to `lags`
        """
        acl_ = []
        for i in range(lags):
            ac = series.autocorr(lag=i)
            acl_.append(ac)
        return acl_

    # Autocorrection function (ACF) of sample data
    ACF_RANGE = len(sample) + 1
    ACF_RANGE = 1001
    x = list(range(1, ACF_RANGE))
    acf_series = acf(sample['trade_sign'], ACF_RANGE)[1:]

    xy_df = pd.DataFrame({
        'x_values': x,
        'y_values': acf_series
    })

    # Basic Usage
    fit = Fit(xy_df, verbose= True)
    fit.pure_powerlaw.print_fitted_results()

    # Direct comparison against alternative models
    R, p = fit.function_compare('pure_powerlaw', 'exponential_function')
    print('Power law vs. exponential_function')
    print(f'Normalized Likelihood Ratio: {R}, p.value: {p}')
    print('\n')


    # Advanced Usage

    # Define custom power-law model: power-law with exponentially slowly varying function
    def powerlaw_with_exp_svf(x, alpha, beta, lambda_):
        return x ** -(alpha) * sf.exponential_function(x, beta, lambda_)

    custom_powerlaw_funcs = {
        'powerlaw_with_exp_svf': powerlaw_with_exp_svf
    }

    # Fit custom function
    fit.fit_powerlaw_function(custom_powerlaw_funcs)
    fit.powerlaw_with_exp_svf.print_fitted_results()


    # Direct comparison against alternative models
    print('powerlaw_with_exp_svf vs. exponential_function:')
    R, p = fit.function_compare('powerlaw_with_exp_svf', 'exponential_function', nested = True)
    print(f'Normalized Likelihood Ratio: {R}, p.value: {p}')

    # Plot
    fit.plot_data(scale='linear')
    fit.exponential_function.plot_fit()
    fit.pure_powerlaw.plot_fit()
    fit.powerlaw_with_exp_svf.plot_fit()