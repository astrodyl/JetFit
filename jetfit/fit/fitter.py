import sys
import copy

from time import time
import numpy as np
import emcee as em

from jetfit.fit.flux_generator import FluxGenerator
from jetfit.utils.log import log_progress, log_results


"""
    fitter.py

    This module contains the Fitter class for boosted fireball model.
    It depends on FluxGenerator and FluxData modules. The definitions 
    for posterior function and prior function are also included. 
    
    Performs MCMC analysis to fit boosted fireball model to observational 
    data.
"""


class Fitter:
    # Fitting Parameter
    _fitting_dimensions = 0
    _info = None
    _fitting_bounds = None
    _params = None

    # Sampler
    _sampler = None
    _sampler_type = None
    _burn_position = None
    _run_position = None

    # Interpolator and Sampler
    flux_generator = None
    table_info = None

    # Observation Data
    times = None
    time_bounds = None
    fluxes = None
    flux_errs = None
    frequencies = None

    def __init__(self, csf: str, options: dict, bounds: dict, defaults: dict,
                 explore: bool = False, log_table: bool = True, log_axis: bool = None) -> None:
        """ Initializes the Fitter object.

        For non-fitting parameters:
            Use the default values.

        For fitting parameters:
            If explore is true, fitting parameters are randomly distributed
            in whole parameter space. Else, Fitting parameters are randomly
            distributed around maximum posterior region, indicated by the
            default parameter values.

        :param csf: path to characteristic spectral functions table
        :param options: information for MCMC. E.g., set fitting parameters
        :param bounds: lower & upper bounds for fitting parameters
        :param defaults: default values for:
            - hydrodynamic: explosion energy, circumburst density,
                boost Lorentz factor, asymptotic Lorentz factor, spectral index
            - radiation: electron_energy_fraction, magnetic_energy_fraction,
                accelerated_electron_fraction
            - observational: redshift, luminosity distance, observing angle
        :param explore: whether to explore fitting parameters
        :return: Fitter object
        """
        self.flux_generator = FluxGenerator(csf, log_table, log_axis if log_axis else ['tau'])
        self._set_fit_parameter(copy.deepcopy(options), copy.deepcopy(bounds), copy.deepcopy(defaults), explore)

    def _set_fit_parameter(self, options: dict, bounds: dict, defaults: dict, explore: bool = False) -> None:
        """ Sets the fitting parameters and their bounds.

        :param options: information for MCMC. E.g., set fitting parameters
        :param bounds: bounds for parameters
        :param defaults: default values for:
            - hydrodynamic: explosion energy, circumburst density,
                boost Lorentz factor, asymptotic Lorentz factor, spectral index
            - radiation: electron_energy_fraction, magnetic_energy_fraction,
                accelerated_electron_fraction
            - observational: redshift, luminosity distance, observing angle
        :param explore: whether to explore fitting parameters
        """
        self._fitting_dimensions = len(options['fit'])
        self._info = options

        # Only consider bounds for fitting parameters. Set proper scales.
        temp = []
        for key in options['fit']:
            if key in options['log']:
                func = np.log10 if options['log_type'] == 'Log10' else np.log
                temp.append(func(bounds[key]))
            else:
                temp.append(bounds[key])

        self._fitting_bounds = np.array(temp)
        self._fitting_bounds_dict = bounds

        # Set initial regions for walkers
        if explore:
            self._initial_bound = self._fitting_bounds
        else:
            temp = []
            for key in options['fit']:
                if key in options['log']:
                    func = np.log10 if options['log_type'] == 'Log10' else np.log
                    temp.append([func(defaults[key]) * 0.98, func(defaults[key]) * 1.02])
                else:
                    temp.append([defaults[key] * 0.98, defaults[key] * 1.02])

            self._initial_bound = np.array(temp)

        self._params = defaults.copy()

    def load_data(self, **kwargs) -> None:
        """ Stores the observational data to the class.

        :param kwargs: Accepted keyword arguments include:
            - times (ndarray): observation time in second
            - time_bounds (ndarray): observation time bounds; Current MCMC will not use this information
            - fluxes (ndarray): fluxes in mJy
            - flux_errors (ndarray): flux errors
            - frequencies (ndarray): frequencies
        """
        self.times = kwargs['times']
        self.time_bounds = kwargs['time_bounds']
        self.fluxes = kwargs['fluxes']
        self.flux_errs = kwargs['flux_errors']
        self.frequencies = kwargs['frequencies']

    def set_sampler(self, sampler: str, num_temps: int, num_walkers: int, threads: int) -> None:
        """ Sets up the sampler and initial position for burn-in.

        :param sampler: 'Ensemble' or 'ParallelTempered'
        :param num_temps: only valid for Parallel-Tempering
        :param num_walkers: number of walkers
        :param threads: number of threads
        """
        self._sampler_type = sampler

        if sampler == "Ensemble":
            self._sampler = em.EnsembleSampler(num_walkers, self._fitting_dimensions, log_posterior,
                                               args=[self._fitting_bounds, self._info, self._params, self.flux_generator,
                                                     self.times, self.frequencies, self.fluxes, self.flux_errs],
                                               threads=threads)

            self._burn_position = (self._initial_bound[:, 0] + (self._initial_bound[:, 1]-self._initial_bound[:, 0]) *
                                   np.random.rand(num_walkers, self._fitting_dimensions))
        else:
            self._sampler = em.PTSampler(num_temps, num_walkers, self._fitting_dimensions, log_like, log_prior,
                                         loglargs=[self._info, self._params, self.flux_generator, self.times,
                                                   self.frequencies, self.fluxes, self.flux_errs],
                                         logpargs=[self._fitting_bounds, self._info], threads=threads)

            self._burn_position = (self._initial_bound[:, 0] + (self._initial_bound[:, 1]-self._initial_bound[:, 0]) *
                                   np.random.rand(num_temps, num_walkers, self._fitting_dimensions))

    def run(self, iterations: int, action: str, output: str = None) -> dict:
        """ Performs MCMC analysis to fit boosted fireball model to
        observational data.

        :param iterations: number of MCMC iterations
        :param action: 'burning' or 'run'
        :param output: optional output file name
        :return: run results as a dictionary
        """
        if action == 'burning':
            starting_pos = self._burn_position
        elif action == 'running':
            self._sampler.reset()
            starting_pos = self._run_position
        else:
            raise ValueError(f'Invalid run action provided: {action}')

        time_start, i = time(), 1
        for step_result in self._sampler.sample(starting_pos, iterations=iterations, storechain=True):
            log_progress(time() - time_start, (100.0 * i) / iterations, action.capitalize())
            i += 1

        sys.stdout.write('\n')

        if action == 'burning':
            self._run_position = step_result[0]

        if output is not None:
            log_results(output, self, self.get_results())

        return self.get_results()

    def get_sampler(self) -> em.Sampler:
        return self._sampler

    def get_info(self):
        return self._info

    def get_fitting_bounds_dict(self):
        return self._fitting_bounds_dict

    def get_params(self):
        return self._params

    def get_results(self) -> dict:
        """ Returns a dictionary of results from the sampler. If the
        sampler is Parallel-Tempered, the results will contain the
        lowest temperature results.
        """
        if self._sampler_type == 'Ensemble':
            return {
                'Chain': self._sampler.chain,
                'LnProbability': self._sampler.lnprobability,
                'AcceptanceFraction': self._sampler.acceptance_fraction
            }
        else:
            return {
                'Chain': self._sampler.chain[0],
                'LnProbability': self._sampler.lnprobability[0],
                'AcceptanceFraction': self._sampler.acceptance_fraction[0]
            }


"""
We need to explicitly define the prior, likelihood and posterior.
To run emcee in parallel, the definitions for prior and likelihood are tricky. 
Please check the emcee document: https://dfm.io/emcee/current/user/advanced/#multiprocessing
"""


def log_prior(fit_params: np.ndarray, bounds, info: dict) -> float:
    """ This method is for PTSampler

    :param fit_params: values for fitting parameters
    :param bounds: lower & upper bounds for fitting parameters: shape = (len(FitParameter, 2))
    :param info: prior information
    :return: if within bounds, log(prior); else, -inf;
    """

    if ((fit_params[:] > bounds[:, 0]) * (fit_params[:] < bounds[:, 1])).all():
        if 'observation_angle' in info['fit']:
            if info['obs_angle_prior'] == 'Sine':
                i = np.argwhere(info['fit'] == 'observation_angle')[0][0]
                return np.log(np.sin(fit_params[i]))
            elif info['obs_angle_prior'] == 'Uniform':
                return 0.0
            else:
                raise ValueError("Cannot recognize Info['obs_angle_prior'].")
        else:
            return 0.0
    else:
        return -np.inf


def log_like(fit_params: np.ndarray, info: dict, params: dict, flux_generator,
             times: np.ndarray, frequencies: np.ndarray, flux: np.ndarray, flux_errs: np.ndarray) -> float:
    """ This method is for PTSampler

    :param fit_params: values for fitting parameters
    :param info: prior information
    :param params: contains all parameters {z, dL, E, n, p, epse, epsb, xiN, Eta0, GammaB, theta_obs}
    :param flux_generator: flux generator
    :param times: observational time in second
    :param frequencies: frequencies. The length should be the same as times
    :param flux: flux in mJy
    :param flux_errs: flux error in mJy
    :return: calculate Chi^2 and return -0.5*Chi^2 (details see Ryan+ 2014)
    """
    for i, key in enumerate(info['fit']):
        if key in info['log']:
            if info['log_type'] == 'Log10':
                params[key] = np.power(10.0, fit_params[i])
            else:
                params[key] = np.exp(fit_params[i])
        else:
            params[key] = fit_params[i]

    if info['flux_type'] == 'Spectral':
        flux_model = flux_generator.get_spectral(times, frequencies, params)
    elif info['fluxType'] == 'Integrated':
        flux_model = flux_generator.get_integrated_flux(times, frequencies, params)
    else:
        raise ValueError("Integrated Flux has not implemented!")

    if np.isnan(flux_model[0]):
        return -np.inf

    chi_square = np.sum(((flux - flux_model) / flux_errs)**2)
    return -0.5 * chi_square


def log_posterior(fit_params: np.ndarray, bounds, info: dict, params: dict, flux_generator,
                  times: np.ndarray, frequencies: np.ndarray, flux: np.ndarray, flux_errs: np.ndarray) -> float:
    """ This method is for EnsembleSampler.

    :param fit_params: values for fitting parameters
    :param bounds: lower & upper bounds for fitting parameters: shape = (len(FitParameter, 2))
    :param info: prior information
    :param params: contains all parameters {z, dL, E, n, p, epse, epsb, xiN, Eta0, GammaB, theta_obs}
    :param flux_generator: flux generator
    :param times: observational time in second
    :param frequencies: frequencies. The length should be the same as times
    :param flux: flux in mJy
    :param flux_errs: flux error in mJy
    :return: log(prior) + log(likelihood)
    """
    log_prior_function = log_prior(fit_params, bounds, info)
    log_like_function = log_like(fit_params, info, params, flux_generator, times, frequencies, flux, flux_errs)

    # Log Posterior
    if np.isfinite(log_prior_function) and np.isfinite(log_like_function):
        return log_prior_function + log_like_function

    return -np.inf
