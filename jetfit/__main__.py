import os
from datetime import datetime

import numpy as np

from jetfit.fit.fitter import Fitter
from jetfit.utils import config, csv, log
from jetfit.utils.plot import Plot


def main(**kwargs):
    # Synthetic Light Curve parameters
    slc_parameters = config.get_slc_parameters('./jetfit/config/parameters.json')

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
    mcmc = config.get_mcmc_parameters('./jetfit/config/mcmc.json', kwargs.pop('run_type'))

    burn_length = mcmc.pop('burn_length')
    run_length = mcmc.pop('run_length')

    # Load the observational data from CSV
    data_frame = csv.read('./jetfit/resources/' + kwargs['input_data'])
    observational_data = csv.df_to_dict(data_frame)

    # Get Fitter object and initialize data and sampler
    input_table = './jetfit/resources/' + kwargs['input_table']
    fitter = Fitter(input_table, kwargs, parameter_bounds, parameter_defaults, explore=True)
    fitter.load_data(**observational_data)
    fitter.set_sampler(**mcmc)

    # Burn in and then perform actual run
    burn_result = fitter.run(iterations=burn_length, action='burning')
    mcmc_result = fitter.run(iterations=run_length, action='running')

    # Find the best fitting parameters
    result_chain = mcmc_result['Chain']
    ln_probability = mcmc_result['LnProbability']
    best_walker = np.unravel_index(np.nanargmax(ln_probability), ln_probability.shape)

    # Create a unique results directory
    results_path = f"./jetfit/results/{datetime.now().microsecond}/"

    if not os.path.exists(results_path):
        os.mkdir(results_path)

    # Plot the light curves and parameter distributions
    plotter = Plot(result_chain, fitter, result_chain[best_walker], **kwargs)
    plotter.plot_light_curves(data_frame, parameter_defaults, results_path + 'curves.png')
    plotter.plot_corner_plot(parameter_bounds, results_path + 'corner.png')
    plotter.plot_markov_chain(results_path + 'chain.png')

    # Log the best values and medians to the console
    log.log_best_values(result_chain, best_walker, kwargs['fit'], results_path + 'best_fits.json')


if __name__ == '__main__':

    params = {
        'run_type': 'quick',               # One of: 'quick' or 'full'
        'input_table': 'Table.h5',        # Characteristic Spectral Functions
        'input_data': 'GW170817.csv',     # Observational data
        'fit': np.array([                 # Parameters to fit
            'explosion_energy',
            'circumburst_density',
            'asymptotic_lorentz_factor',
            'boost_lorentz_factor',
            'observation_angle',
            'electron_energy_fraction',
            'magnetic_energy_fraction',
            'spectral_index'
        ]),
        'log': np.array([                 # Sets parameters in log scale
            'explosion_energy',
            'circumburst_density',
            'electron_energy_fraction',
            'magnetic_energy_fraction'
        ]),
        'log_type': 'Log10',              # One of: 'Log10' or 'Log'
        'obs_angle_prior': 'Sine',        # One of: 'Sine' or 'Uniform'
        'flux_type': 'Spectral'           # One of: 'Spectral' or 'Integrated'
    }

    main(**params)
