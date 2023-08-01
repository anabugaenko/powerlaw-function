import numpy as np
from scipy.stats import laplace, norm, t, expon, lognorm, powerlaw, gumbel_l, gumbel_r, kstest
from util.goodness_of_fit import compute_goodness_of_fit, compute_loglikelihood


def get_ic_for_model(model, y_values, lags, params):
    fitted_values = model(lags, *params)
    residuals = y_values - fitted_values
    adjusted_rsquared, bic = compute_goodness_of_fit(residuals, y_values, params)

    return bic, adjusted_rsquared


def distribution_tests(series, function_name, test='all',
                       distributions=[laplace, norm, t, expon, lognorm, powerlaw, gumbel_l, gumbel_r]):
    tests = {
        'ks': 'Kolmogorov-Smirnov test:',
        'lr': 'Likelihood ratio test:'
    }

    if test == 'all':
        test = tests.keys()

    # Base case - always do Likelihood ratio test
    print('\n', tests['lr'])
    results = []
    for distribution in distributions:
        params = distribution.fit(series)
        log_likelihood = np.sum(distribution.logpdf(series, *params))
        results.append((distribution.name, log_likelihood))
    results.sort(key=lambda x: x[1], reverse=True)
    for result in results:
        print(result)
        print("")
    print(f'The most likely distribution for {function_name} according to Likelihood ratio test: {results[0][0]}')

    # The distribution with the highest log-likelihood is considered the most plausible distribution.
    best_likelihood = results[0][0]

    # KS test
    # A smaller KS statistic indicates a better fit to the data.
    # If the KS statistic is small, we do not reject the null hypothesis that the data follows the specified distribution.
    if 'ks' in test:
        print('\n', tests['ks'])
        results = []
        for dist in distributions:
            params = dist.fit(series)
            ks_result = kstest(series, dist.name, args=params)
            results.append((dist.name, ks_result[0]))  # Store (distribution name, KS statistic)

        # Sort by KS statistic in ascending order and print the most likely distribution
        results.sort(key=lambda x: x[1])
        for result in results:
            print(f"For {result[0]} distribution: KS statistic={result[1]}")
            print("\n")
        print(f"The most likely distribution for {function_name} according to KS-test: {results[0][0]}")
        print("\n")


    return best_likelihood


def get_residuals_loglikelihoods(first_series, fitting_func1, second_series, fitting_func2):
    x_values = np.arange(1, len(first_series) + 1)

    # Trim
    if len(first_series) != len(second_series):
        min_len = min(len(first_series), len(second_series))
        first_series = first_series[-min_len:]
        second_series = second_series[-min_len:]
        x_values = x_values[-min_len:]

    params_first_model = fitting_func1.fit(first_series)
    params_second_model = fitting_func2.fit(second_series)

    loglikelihoods1 = compute_loglikelihood(fitting_func1.pdf, params_first_model, x_values, first_series)
    loglikelihoods2 = compute_loglikelihood(fitting_func2.pdf, params_second_model, x_values, second_series)

    return loglikelihoods1, loglikelihoods2


