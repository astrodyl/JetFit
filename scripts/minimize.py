import json
from pathlib import Path

import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

from jetfit.ampy import Ampy
from jetfit.core import utils
from jetfit.mcmc.mcmc import log_posterior_fn
from jetfit.mcmc.priors import GaussianPrior
from jetfit.models.base import OpeningAngleModel
from scripts.plot.visualize import plot_light_curve


def safe_log_posterior_fn(theta, params, models):
    """
    Calculates the natural log of the posterior
    probability. The minimizer will crash if the
    posterior is not finite. This method wraps
    and returns a large, negative number instead
    of negative infinity.

    Parameters
    ----------
    theta : np.ndarray of float
        The MCMC sampled values.

    params : Parameters
        The MCMC parameter container.

    models : MCMCModels
        The MCMC models container.

    Returns
    -------
    float
        The natural log of the posterior.
    """
    lp = log_posterior_fn(theta, params, models)
    return lp if np.isfinite(lp) else -1e10


def run_minimizer(x0, func_args=(), bounds=(), minimizer='minimize'):
    """
    Runs the minimizer on the negative maximum a
    posteriori (MAP).

    Parameters
    ----------
    x0 : np.ndarray
        Initial positions.

    func_args : tuple
        Any args needed for the ``log_posterior_fn``.

    bounds : np.ndarray
        Sequence of ``(min, max)`` pairs for each
        element in `x`. None is used to specify no bound.

    minimizer : str
        Which minimizer to use. Must be 'minimize' or 'basinhopping'.
        Use basinhopping for difficult posteriors.

    Returns
    -------
    OptimizeResult
        The optimization result represented as a ``OptimizeResult``
        object. Important attributes are: ``x`` the solution array,
        ``success`` a Boolean flag indicating if the optimizer exited
        successfully and ``message`` which describes the cause of the
        termination.
    """
    nmap = lambda *lp_args: -safe_log_posterior_fn(*lp_args)

    if minimizer == 'minimize':
        return optimize.minimize(nmap, x0, args=func_args, bounds=bounds)

    elif minimizer == 'basinhopping':
        return optimize.basinhopping(
            nmap, x0, minimizer_kwargs={"method": "L-BFGS-B", "args": func_args, "bounds": bounds}
        )

    raise ValueError(f"Unknown minimizer '{minimizer}'.")


def plot_results(ampy, params, out_dir):
    """
    Plot the best fitting light curve from an Ampy object.

    Parameters
    ----------
    ampy : Ampy
        The Ampy object.

    params : dict
        The model parameters to plot.

    out_dir : Path, optional
        The output directory.
    """
    # Light curve plotter takes an extinction object
    ext_model = None
    if ampy.extinction_model is not None:
        ext_model = ampy.extinction_model(Rv=3.1)

    plot_light_curve(
        ampy.afterglow_model, params, ampy.obs,
        out_dir=out_dir, ext_model=ext_model
    )


def log_results(p, out_dir):
    """
    Log the minimized parameters to a JSON file.

    Parameters
    ----------
    p : dict
        The minimized parameters.

    out_dir : Path
        The output directory.
    """
    with open(out_dir / 'minimized.json', "w") as f:
        json.dump(p, f, indent=4)  # type: ignore


def main(obs_path, params_path, results_dir, initial_path):
    """
    Runs the BestFitinator and plots the light curve.

    Parameters
    ----------
    obs_path : Path
        The path to the observation CSV file.

    params_path : Path
        The path to the input parameters TOML file.

    results_dir : Path
        The output directory.

    initial_path : Path
        The path to the initial positions TOML file.

    Returns
    -------
    OptimizeResult
        See `run_minimizer` for details.
    """

    # Let the Ampy class format everything
    ampy = Ampy(obs_path, params_path)

    # Initial starting points and search bounds
    initial, bounds = [], []

    # Load the best fitting results
    with open(initial_path, "r") as f:
        results = json.load(f)

    # Set the initial search pos and bounds
    for p in ampy.mcmc.params.fitting:
        # Results are stored in linear space, but we
        # want them in their original fitting space
        initial.append(
            utils.to_scale(results[p.name], from_s='linear', to_s=p.scale)
        )

        # Use the prior bounds as the search bounds
        if isinstance(p.prior, GaussianPrior):
            # Widen Gaussian bounds in case best was outside 3-sigma
            bounds.append((p.prior.lower * 3, p.prior.upper * 3))
        else:
            bounds.append((p.prior.lower, p.prior.upper))

    # Run the minimizer
    func_args = (ampy.mcmc.params, ampy.mcmc.models)

    results = run_minimizer(
        x0=np.array(initial), func_args=func_args, bounds=np.array(bounds)
    )

    # Add some additional logging info
    min_params = ampy.mcmc.params.samples_to_dict(results.x)
    min_params['nmap'] = -2 * log_posterior_fn(results.x, *func_args)
    min_params['success'] = results.success
    min_params['message'] = results.message

    # Plot the minimized results
    # plot_results(ampy, min_params, results_dir)

    # Write the results to a JSON file
    log_results(min_params, results_dir)

    return results


if __name__ == "__main__":
    # Run the BestFitinator via the command line
    parser = argparse.ArgumentParser(description="BestFitinator Parameters")
    parser.add_argument('--obs',      help='The input observation file.')
    parser.add_argument('--params',   help='The input parameter TOML file.')
    parser.add_argument('--results',  help='The results directory.')
    parser.add_argument('--initial',  help='The initial positions TOML file.')
    args = parser.parse_args()

    # oa1 = OpeningAngleModel(1, 1, 0, 1).evaluate(10)
    # oa2 = OpeningAngleModel(1, 1, 2, 1).evaluate(10)
    # print(oa1, '\t', oa2)
    #
    # def valid(t_min, t_max, z):
    #     lower = ((1 + z) * 8.64e1 / t_min) ** 3
    #     upper = ((1 + z) * 8.64e8 / t_max) ** 3
    #     return lower, upper
    #
    # for i, e in enumerate(np.linspace(-10, 3.0, 25)):
    #     for j, n in enumerate(np.linspace(-5, 5, 25)):
    #
    #         # lo, hi = valid(796436, 69120000, 0.00973)  # noqa
    #         # lo, hi = valid(77.896427, 1396613.10932, 0.606)  # noqa
    #         # lo, hi = valid(2755.828484, 287680.272917, 3.847)  # noqa
    #         lo, hi = valid(427.997504, 2571895.78322, 0.714)  # noqa
    #
    #         if hi >= (10**n / 10**e) >= lo:  # noqa
    #             plt.scatter(e, n, color='blue', label='Valid' if i == 0 and j == 0 else None)
    #         else:
    #             plt.scatter(e, n, color='red', label='Rejected' if i == 0 and j == 0 else None)
    #
    # plt.xlabel(r'$\log_{10}E_{iso}$')
    # plt.ylabel(r'$\log_{10}n_{0}$')
    # plt.legend()
    # plt.show()
    #
    # plt.plot( [y[1] for y in x], np.linspace(-6, 3, 100))
    # plt.show()

    # Or run the BestFitinator manually
    # for event in os.listdir(rf"C:\Server\FINAL\analytic"):
    #     sub_dir = 'grbs'
    event, sub_dir = '220101A', 'grbs'
    p_obs = utils.get_input_csv_path(sub_dir, event)

    # p_params  = utils.get_event_path(sub_dir, event) / 'jetsim.toml'
    # d_results = Path(rf"C:\Server\FINAL\numerical\ism\{event}\minimized")
    # p_initial = Path(rf"C:\Server\FINAL\numerical\ism\{event}\minimized\best_fit.json")

    p_params  = utils.get_event_path(sub_dir, event) / 'parameters.toml'
    d_results = Path(rf"C:\Server\FINAL\analytic\{event}\minimized")
    p_initial = rf"C:\Server\FINAL\analytic\{event}\minimized\best_fit.json"

    response = main(
        **{
            'obs_path':     args.obs     or p_obs,
            'params_path':  args.params  or p_params,
            'results_dir':  args.results or d_results,
            'initial_path': args.initial or p_initial,
        }
    )

    print(f"Was the BestFitinator successful? {response.success}")
