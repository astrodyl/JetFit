from pathlib import Path

import numpy as np

from jetfit.core.values.mcmc_value import MCMCFittingParameter
from jetfit.core.values.mcmc_value import MCMCFixedParameter
from jetfit.core.utilities import utils
from jetfit.core.utilities.io import toml


class MCMC:
    """

    Parameters
    ----------
    burn_length : int
        Number of iterations for each burn in.

    run_length : int
        Number of iterations for run after burn in.

    num_walkers : int
        Number of MCMC walkers.

    init_pos : list of list, with dim (num_walkers, len(fitting)), optional
        Starting positions for MCMC sampling.

    fitting : list of MCMCFittingParameter
        Parameters for MCMC fitting.

    fixed : list of MCMCFixedParameter
        Fixed parameters.
    """
    def __init__(self, burn_length: int, run_length: int, num_walkers: int,
                 fitting: list, fixed: list, init_pos: list = None):

        self.run_length = run_length
        self.burn_length = burn_length
        self.num_walkers = num_walkers

        self.fitting_parameters = fitting
        self.fixed_parameters = fixed

        self.start_burn_pos = init_pos
        self.num_dims = len(fitting) if fitting is not None else None

        if init_pos is None:
            self.set_start_positions()

        self.start_run_pos = None

    @classmethod
    def from_toml(cls, settings: str | Path, params: str | Path):
        """
        Instantiates the class from a TOML files.

        Parameters
        ----------
        settings : str | Path
            Path of the MCMC settings TOML file.

        params : str | Path
            Path of the MCMC parameters TOML file.

        Notes
        -----
        The TOML setting file requires the following:

        num_walkers : int
            Number of MCMC walkers.

        burn_length : int
            Number of iterations for each burn in.

        run_length : int
            Number of iterations for run after burn in.

        Returns
        -------
        MCMC
            Instantiated MCMC class from TOML files.
        """
        settings = toml.read(settings).get('settings')

        if not utils.is_expected_type(settings.get('burn_length', None), int):
            raise TypeError('Received unexpected or no value for'
                            'burn_length.')

        if not utils.is_expected_type(settings.get('run_length', None), int):
            raise TypeError('Received unexpected or no value for'
                            'run_length.')

        if not utils.is_expected_type(settings.get('num_walkers', None), int):
            raise TypeError('Received unexpected or no value for'
                            'num_walkers.')

        fitting = [MCMCFittingParameter.from_dict(k, v) for k, v
                   in toml.read(params).items() if 'prior' in v]

        fixed = [MCMCFixedParameter.from_dict(k, v) for k, v
                 in toml.read(params).items() if 'value' in v]

        return cls(settings.get('burn_length'), settings.get('run_length'),
                   settings.get('num_walkers'), fitting, fixed)

    def set_start_positions(self):
        """

        Returns
        -------

        """
        self.start_burn_pos = np.zeros((self.num_walkers, self.num_dims))

        for i, p in enumerate(self.fitting_parameters):
            self.start_burn_pos[:, i] = p.prior.draw(self.num_walkers)
