import json
import os

import numpy as np

from jetfit.fit.fitter import Fitter
from jetfit.core.utils import csv, log, config
from jetfit.core.utils.plot import Plot


class JetFit:
    def __init__(self):
        self.mcmc = None        # settings, results (burn, run, best fit, ..)
        self.table = None       # path to file
        self.output = None      # directory
        self.photometry = None  # path to file
        self.parameters = None  # fitting + bounds + scale + prior, fixed


def main(**kwargs):
    inputs = kwargs.pop('inputs')
    configs = kwargs.pop('configs')

    # Synthetic Light Curve parameters
    slc_parameters = config.get_slc_parameters(configs.get('parameters'))

    parameter_bounds = {  # All in linear scale
        **slc_parameters["radiation"]['bounds'],
        **slc_parameters["hydrodynamic"]['bounds'],
        **slc_parameters['observational']['bounds']
    }

    parameter_defaults = {
        **slc_parameters["radiation"]['defaults'],
        **slc_parameters["hydrodynamic"]['defaults'],
        **slc_parameters['observational']['defaults']
    }

    # Markov Chain Monte Carlo parameters
    mcmc = config.get_mcmc_parameters(configs.get('mcmc'), kwargs.pop('run_type'))

    burn_length = mcmc.pop('burn_length')
    run_length = mcmc.pop('run_length')

    # Load the observational data from CSV
    data_frame = csv.read(inputs.get('data'))
    observational_data = csv.df_to_dict(data_frame)

    # Get Fitter object and initialize data and sampler
    input_table = inputs.get('table')
    fitter = Fitter(input_table, kwargs, parameter_bounds, parameter_defaults, explore=True)
    fitter.load_data(**observational_data)
    fitter.set_sampler(**mcmc)

    # if False:
    #     loaded_chain = np.load(rf"./jetfit/results/{kwargs.get('event')}/chain.npy", allow_pickle=True)
    #
    #     with open(rf"./jetfit/results/{kwargs.get('event')}/best_fits.json", 'r') as f:
    #         best_fits = json.load(f)
    #
    #     best_loc = (int(best_fits.get('location').get('index1')), int(best_fits.get('location').get('index2')))
    #
    #     plotter = Plot(loaded_chain, fitter, loaded_chain[best_loc], **kwargs)
    #     plotter.plot_light_curves(data_frame, parameter_defaults, f"./jetfit/results/{kwargs.get('event')}/")
    #     plotter.plot_corner_plot(parameter_bounds, f"./jetfit/results/{kwargs.get('event')}/" + 'corner.png')
    #     plotter.plot_markov_chain(f"./jetfit/results/{kwargs.get('event')}/" + 'chain.png')
    #
    #     exit()

    # Burn in and then perform actual run

    burn_result = fitter.run(iterations=burn_length, action='burning')
    mcmc_result = fitter.run(iterations=run_length, action='running')

    # Find the best fitting parameters
    result_chain = mcmc_result['Chain']
    ln_probability = mcmc_result['LnProbability']
    best_walker = np.unravel_index(np.nanargmax(ln_probability), ln_probability.shape)

    # Create a unique results directory
    results_path = f"./jetfit/results/{kwargs.pop('event')}/"

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # TEMP BURNER
    burn_chain = burn_result['Chain']
    plotter2 = Plot(burn_chain, fitter, burn_chain[best_walker], **kwargs)
    plotter2.plot_autocorrelation(results_path + 'auto_corr.png')
    plotter2.plot_markov_chain(results_path + 'burn_chain.png')

    # Plot the light curves and parameter distributions
    plotter = Plot(result_chain, fitter, result_chain[best_walker], **kwargs)
    plotter.plot_light_curves(data_frame, parameter_defaults, results_path)
    plotter.plot_corner_plot(parameter_bounds, results_path + 'corner.png')
    plotter.plot_markov_chain(results_path + 'chain.png')

    # Log the best values and medians to the console
    log.log_best_values(result_chain, best_walker, kwargs['fit'], results_path + 'best_fits.json')
