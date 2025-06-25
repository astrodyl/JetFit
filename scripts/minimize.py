import json

import numpy as np
from scipy import optimize

from jetfit.ampy import Ampy
from jetfit.core import utils
from jetfit.mcmc.mcmc import log_posterior_fn
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


def run_minimizer(x0, args=(), bounds=(), minimizer='minimize'):
    """
    Runs the minimizer on the negative maximum a
    posteriori (MAP).

    Parameters
    ----------
    x0 : np.ndarray
        Initial positions.

    args : tuple
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
        return optimize.minimize(nmap, x0, args=args, bounds=bounds)

    elif minimizer == 'basinhopping':
        return optimize.basinhopping(
            nmap, x0, minimizer_kwargs={"method": "L-BFGS-B", "args": args, "bounds": bounds}
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


def main():
    """ Runs the minimizer and plots the results. """
    # Have you run the event with MCMC already? And
    # do you want to use the best parameters as the
    # starting point? If so, set to True.
    use_best_fit = False

    event = '090618'
    sub_dir = 'grbs'

    # Where does everything live?
    obs_path = utils.get_input_csv_path(sub_dir, event)
    params_path = utils.get_event_path(sub_dir, event) / 'parameters.toml'
    out_dir = utils.get_results_path() / event / 'min'

    # Let the Ampy class format everything
    ampy = Ampy(obs_path, params_path)

    # Initial starting points and search bounds
    initial, bounds = [], []

    if use_best_fit:
        # Use the best fitting results as the starting points
        best_path = utils.get_results_path() / event / 'min' / 'best_fit_min.json'

        # Load the best fitting results
        with open(best_path, "r") as f:
            results = json.load(f)

        for p in ampy.mcmc.params.fitting:
            # Results are stored in linear space, but we
            # want them in their original fitting space
            initial.append(
                utils.to_scale(results[p.name], from_s='linear', to_s=p.scale)
            )

            # Use the prior bounds as the search bounds
            bounds.append((p.prior.lower, p.prior.upper))
    else:
        for p in ampy.mcmc.params.fitting:
            # Use the middle of the prior as starting point
            initial.append((p.prior.upper + p.prior.lower) / 2)

            # Use the prior bounds as the search bounds
            bounds.append((p.prior.lower, p.prior.upper))

    # Run the minimizer
    args = (ampy.mcmc.params, ampy.mcmc.models)

    results = run_minimizer(
        x0=np.array(initial), args=args, bounds=np.array(bounds)
    )

    # Add some additional logging info
    min_params = ampy.mcmc.params.samples_to_dict(results.x)
    min_params['nmap'] = -2 * log_posterior_fn(results.x, *args)
    min_params['success'] = results.success
    min_params['message'] = results.message

    # Plot the minimized results
    plot_results(ampy, min_params, out_dir)

    # Write the results to a JSON file
    log_results(min_params, out_dir)


if __name__ == "__main__":
    main()
