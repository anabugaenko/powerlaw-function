from util.functions import exponential_function, logarithmic_function, periodic_function, \
    linear_function


# pure power-law function
def pure_powerlaw(x, alpha, C):
    return C * x ** alpha

def powerlaw_with_exp_svf(x, alpha, beta, lambda_):
    return x ** (-alpha) * exponential_function(x, beta, lambda_)

def powerlaw_with_log_svf(x, alpha, beta, lambda_):
    return x ** (-alpha) * logarithmic_function(x, beta, lambda_)

def powerlaw_with_lin_svf(x, alpha, beta, lambda_):
    return x ** (-alpha) * linear_function(x, beta, lambda_)

def powerlaw_with_per_svf(x, alpha, a0, A, omega):
    return x ** (-alpha) * periodic_function(x, a0, A, omega)

