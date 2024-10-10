import sys
import json
from pickle import dump, HIGHEST_PROTOCOL

import numpy as np


def log_progress(delta_time: float, percent: float, action: str) -> None:
    """ Prints the progress of the MCMC run to the console.

    :param delta_time: the time in seconds since the start
    :param percent: the percent complete
    :param action: 'Running' or 'Burning'
    """
    label = "%02d m %02d s" % (delta_time / 60, delta_time % 60)
    sys.stdout.write(f'\r {action} ... %.1f%% Time=%s' % (percent, label))
    sys.stdout.flush()


def log_results(path: str, fitter, result: dict) -> None:
    """ Write results to a pickle file on disk.

    :param path: path to save the pickle file to
    :param fitter: Fitter instance
    :param result: result dictionary
    """
    output_data = {
        "result": result,
        "options": fitter.get_info(),
        "parameter_bounds": fitter.get_fitting_bounds_dict(),
        "parameter_defaults": fitter.get_params(),
        "table_info": fitter.flux_generator.table_info
    }

    with open(path, 'wb') as handle:
        dump(output_data, handle, protocol=HIGHEST_PROTOCOL)


def log_best_values(chain: np.ndarray, best_walker: tuple, params: np.ndarray, path: str) -> None:
    """ Prints the best fit values and their 1-sigma values.

    :param chain: markov chain object
    :param best_walker: the best walker indices
    :param params: list of parameter names
    :param path: path to save the JSON file to
    """
    def percentile(data, confidence_level: float = 0.68) -> (float, float):
        lower_percentile = 50 - (confidence_level / 2) * 100
        upper_percentile = 50 + (confidence_level / 2) * 100

        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)

        return np.median(data) - lower_bound, upper_bound - np.median(data)

    results = {}
    for i, param in enumerate(params):
        # All walkers
        flat_chain = chain.reshape((-1, len(params)))
        all_lower, all_upper = percentile(flat_chain[:, i])

        # Best walker
        best_lower, best_upper = percentile(chain[best_walker[0]][:, i])
        median = np.median(chain[best_walker[0]][:, i])

        results[f'{param}'] = {
            'best_walker': {
                'best_fit_value': round(chain[best_walker][i], 2),
                'median_value': round(median, 2),
                'upper_1sig_bound': round(best_upper, 2),
                'lower_1sig_bound': round(best_lower, 2)
            },
            'all_walkers': {
                'median_value': round(np.median(flat_chain[:, i]), 2),
                'upper_1sig_bound': round(all_upper, 2),
                'lower_1sig_bound': round(all_lower, 2)
            }
        }

    with open(path, 'w') as f:
        json.dump(results, f, indent=4)
