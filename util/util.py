import os
import sys
from inspect import signature


def block_print():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return original_stdout

def enable_print(original_stdout):
    if original_stdout is not None:
        sys.stdout = original_stdout


# Helper Classes
class FunctionParams:
    """
    This class serves as a container for function parameters. It uses Python's introspection capabilities
    to automatically map parameters to their corresponding values for a given function.

    Parameters
    ----------
    function : callable
        The function for which the parameters are being stored. This should be a function where the first
        argument is the independent variable (commonly 'x'), followed by its parameters.

    params : list or tuple
        The parameter values for the function. These should be in the same order as in the function definition.

    Attributes
    ----------
    param_names : list
        The names of the parameters of the function, excluding the independent variable.

    Methods
    -------
    get_values():
        Returns the parameter values in the same order as `param_names`.
    """

    def __init__(self, function, params):
        self.param_names = list(signature(function).parameters.keys())[1:]  # exclude 'x'
        for name, value in zip(self.param_names, params):
            setattr(self, name, value)

    def get_values(self):
        return [getattr(self, name) for name in self.param_names]
