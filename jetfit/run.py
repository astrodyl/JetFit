import argparse
import numpy as np

from jetfit import main as run_jetfit

if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="JetFit Parameters")

    parser.add_argument('--event', help='Event Identifier')

    # Configurations
    parser.add_argument('--parameters', help='path to the parameter config file')
    parser.add_argument('--mcmc', help='path to the mcmc config file')

    # Inputs
    parser.add_argument('--table', help='path to the numerical table file')   # Table.h5
    parser.add_argument('--input', help='path to the input csv file')         #

    args = parser.parse_args()

    params = {
        'run_type': 'quick',             # One of: 'quick' or 'full'
        'event': args.event,
        'inputs': {
            'table': args.table,        # Characteristic Spectral Functions
            'data': args.input,         # Observational data
        },
        'configs': {
            'mcmc': args.mcmc,
            'parameters': args.parameters,
        },
        'fit': np.array([               # Parameters to fit
            'explosion_energy',
            'circumburst_density',
            'asymptotic_lorentz_factor',
            'boost_lorentz_factor',
            'observation_angle',
            'electron_energy_fraction',
            'magnetic_energy_fraction',
            'spectral_index'
        ]),
        'log': np.array([               # Sets parameters in log scale
            'explosion_energy',
            'circumburst_density',
            'electron_energy_fraction',
            'magnetic_energy_fraction'
        ]),
        'log_type': 'Log10',            # One of: 'Log10' or 'Log'
        'obs_angle_prior': 'Sine',      # One of: 'Sine' or 'Uniform'
        'flux_type': 'Integrated'         # One of: 'Spectral' or 'Integrated'
    }

    run_jetfit.main(**params)
