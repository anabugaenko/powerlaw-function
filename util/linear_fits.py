import numpy as np
import statsmodels.api as sm

def linear_fit(x_values, y_values, model_type):
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

