import json

from numpy import array as nparray


def read(path: str) -> dict:
    """ Reads the JSON config file at the provided path.

    :param path: path to config file
    :return: JSON config
    """
    with open(path, 'r') as cf:
        return json.load(cf)


def get_mcmc_parameters(path: str, run_type: str) -> dict:
    """ Returns a dictionary of parameters needed for to run a Markov
    Chain Monte Carlo (MCMC) simulation.

    :param path: path to config file
    :param run_type: 'quick' or 'full'
    :return: markov chain monte carlo dictionary
    """
    if run_type not in ['quick', 'full']:
        raise ValueError(f'Incorrect parameter type: {run_type}')

    return read(path)[run_type]


def get_slc_parameters(path: str, param: str = None) -> dict:
    """ Returns a dictionary of Synthetic Light Curve parameters and
    their values/bounds. If 'param' is specified, only the parameters
    for that type are returned.

    :param path: path to config file
    :param param: one of
        'hydrodynamic'
        'radiation'
        'observational
    :return: synthetic light curve dictionary
    """
    valid_params = ['hydrodynamic', 'radiation', 'observational']

    if param is not None and param not in valid_params:
        raise ValueError(f'Incorrect parameter type: {param}')

    parameters = read(path)

    for param_group in parameters:  # Convert bounds to numpy arrays
        for key, value in parameters[param_group]['bounds'].items():
            parameters[param_group]['bounds'][key] = nparray(value)

    return parameters[param] if param else parameters
